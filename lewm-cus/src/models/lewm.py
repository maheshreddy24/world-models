"""LeWM: a joint-embedding predictive world model.

The model is four parts and one idea:

    encoder        observation  -> latent
    action_encoder action block -> latent
    predictor      (latents, actions) -> next latent
    projector(s)   optional heads around the encoder / predictor output

The idea is that prediction happens entirely in latent space — nothing is ever
decoded back to pixels.  What stops a trainable encoder from collapsing the
latent to a constant is not a stop-gradient or an EMA target but the SIGReg
term in `loss()`, which keeps the embedding distribution isotropic Gaussian.
A frozen encoder (DINO-WM) cannot collapse, so there the term is left out.

Conventions used throughout:

    B  batch          T  latent steps      D  embed_dim
    P  tokens per frame (1 for pooled encoders, 196 for DINO patches)
    S  action samples (planning only)      A  action_dim * frameskip

Latent step `t` is the observation at env step `t * frameskip`; action `t` is
the block of `frameskip` raw actions that carries step `t` to step `t + 1`.
So `predict(emb[:, :k+1], act[:, :k+1])[:, -1]` is the model's guess at `emb[k+1]`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from .blocks import MLP
from .encoder import build_encoder, build_oracle_encoder
from .predictor import ARPredictor, ActionEncoder
from .sigreg import SIGReg


def latent_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared L2 per token, averaged over tokens: (..., P, D) -> (...).

    Averaging over P keeps a patch-grid latent on the scale of a single pooled
    vector (identical when P = 1), so costs and MPPI temperatures mean the same
    thing for every encoder.
    """
    return (a - b).pow(2).sum(-1).mean(-1)


def _per_token(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply an (N, D) -> (N, D) head to every token of (..., D).

    The heads are 2-d on purpose: their BatchNorm treats each token as a sample.
    """
    return head(x.reshape(-1, x.size(-1))).reshape(x.shape)


class LeWM(nn.Module):
    """Args:
        encoder: observation encoder, `(N, ...) -> (N, P, D)`.
        action_encoder: action-block encoder, `(B, T, A) -> (B, T, D)`.
        predictor: autoregressive latent dynamics model.
        projector: head applied to encoder output (identity if None).
        pred_proj: head applied to predictor output (identity if None).
        obs_key: which batch key holds the observation.
        context_len: max latent steps fed to the predictor at inference.
        cost: planning cost, one of "final_mse" | "mean_mse" | "final_cosine".
        freeze_encoder: keep the encoder fixed: no gradients, always in eval mode.
        oracle_encoder: optional `(N, S) -> (N, 1, D)` encoder of the oracle state
            (`batch[oracle_key]`), whose token is appended to each frame's
            encoder tokens. It is trained even when the encoder is frozen.
    """

    oracle_key = "observation"

    def __init__(
        self,
        encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
        obs_key: str = "pixels",
        context_len: int = 3,
        cost: str = "final_mse",
        freeze_encoder: bool = False,
        oracle_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.obs_key = obs_key
        self.context_len = context_len # this is the history length used for the predictor at inference time
        self.cost = cost
        self.freeze_encoder = freeze_encoder
        self.oracle_encoder = oracle_encoder
        if freeze_encoder:
            # No grads: the optimiser skips it and no activations are kept for backward.
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True) -> "LeWM":
        """A frozen encoder stays in eval mode, so dropout / BatchNorm cannot move it either."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------ #
    #  Encoding
    # ------------------------------------------------------------------ #
    def encode(self, obs: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """obs: (B, T, ...), state: (B, T, S) -> (B, T, P, D). Time is folded into the batch.

        `state` is the oracle state, read only when the model has an oracle
        encoder; its token comes last in each frame, so P counts it.
        """
        b = obs.size(0)
        tokens = self.encoder(rearrange(obs.float(), "b t ... -> (b t) ..."))  # (B*T, P, D)
        if self.oracle_encoder is not None:
            if state is None:
                raise ValueError(
                    f"this model reads the oracle state next to the observation; "
                    f"pass {self.oracle_key!r} to encode()"
                )
            oracle = self.oracle_encoder(rearrange(state.float(), "b t s -> (b t) s"))  # (B*T, 1, D)
            tokens = torch.cat([tokens, oracle], dim=1)  # (B*T, P + 1, D)
        tokens = _per_token(self.projector, tokens)
        return rearrange(tokens, "(b t) p d -> b t p d", b=b)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, T, A) -> (B, T, D). NaNs (sequence padding) become zeros."""
        return self.action_encoder(torch.nan_to_num(action, 0.0))

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """One causal pass: frame `t` predicts the embedding at `t + 1`.

        emb: (B, T, P, D), act_emb: (B, T, D) -> (B, T, P, D)
        """
        return _per_token(self.pred_proj, self.predictor(emb, act_emb))

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def loss(self, batch: dict, sigreg: SIGReg | None = None, sigreg_weight: float = 0.0) -> dict:
        """Teacher-forced next-embedding prediction, plus SIGReg when one is given.

        Every step of the sequence except the last predicts its successor, so a
        sequence of T steps gives T-1 supervised predictions.  Pass `sigreg=None`
        only when the target cannot collapse (frozen encoder, no projector).
        """
        emb = self.encode(batch[self.obs_key], batch.get(self.oracle_key))  # (B, T, P, D)
        act_emb = self.encode_action(batch["action"])  # (B, T, D)

        pred = self.predict(emb[:, :-1], act_emb[:, :-1])  # (B, T-1, P, D)
        target = emb[:, 1:]  # (B, T-1, P, D)

        pred_loss = (pred - target).pow(2).mean()
        out = {"loss": pred_loss, "pred_loss": pred_loss.detach()}
        if sigreg is not None:
            # (T, B*P, D): each timestep's token marginal is tested against N(0, I)
            sigreg_loss = sigreg(rearrange(emb, "b t p d -> t (b p) d"))
            out["loss"] = pred_loss + sigreg_weight * sigreg_loss
            out["sigreg_loss"] = sigreg_loss.detach()

        with torch.no_grad():
            # Collapse announces itself as emb_std -> 0 long before the loss moves.
            out["emb_std"] = emb.std(dim=(0, 1, 2)).mean()
            out["emb_norm"] = emb.norm(dim=-1).mean()
            # How much better than "the world never changes" the prediction is;
            # < 1 means the predictor has learned something about dynamics.
            baseline = (emb[:, :-1] - target).pow(2).mean()
            out["pred_vs_static"] = pred_loss / baseline.clamp_min(1e-8)
        return out

    # ------------------------------------------------------------------ #
    #  Inference: latent rollout
    # ------------------------------------------------------------------ #
    def rollout(self, ctx_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll the predictor forward on its own output.

        Args:
            ctx_emb: (B, H, P, D) embeddings of the H observed steps.
            actions: (B, T, A) action blocks for steps 0..T-1, with T >= H.

        Returns:
            (B, T - H + 1, P, D) predictions for steps H .. T. The last one is
            the latent reached after executing every action block.
        """
        h, t = ctx_emb.size(1), actions.size(1)
        if t < h:
            raise ValueError(f"need at least {h} action blocks for a {h}-step context, got {t}")

        act_emb = self.encode_action(actions)
        emb = ctx_emb
        preds = []
        for k in range(h - 1, t):
            lo = max(0, k - self.context_len + 1)  # slide a fixed-width window
            nxt = self.predict(emb[:, lo : k + 1], act_emb[:, lo : k + 1])[:, -1:]
            emb = torch.cat([emb, nxt], dim=1)
            preds.append(nxt)
        return torch.cat(preds, dim=1)

    # ------------------------------------------------------------------ #
    #  Inference: planning cost
    # ------------------------------------------------------------------ #
    def goal_cost(self, preds: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        """Distance from a rollout to the goal latent.

        preds: (N, T, P, D), goal_emb: (N, P, D) -> (N,)
        """
        if self.cost == "final_mse":
            return latent_sq_dist(preds[:, -1], goal_emb)
        if self.cost == "mean_mse":
            return latent_sq_dist(preds, goal_emb.unsqueeze(1)).mean(dim=1)
        if self.cost == "final_cosine":
            return 1.0 - F.cosine_similarity(preds[:, -1], goal_emb, dim=-1).mean(dim=-1)
        raise ValueError(f"unknown cost {self.cost!r}")

    @torch.no_grad()
    def plan_cost(
        self,
        ctx_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        candidates: torch.Tensor,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Score action candidates against the goal — the function the solver minimises.

        The context is encoded once by the caller and merely expanded here, so a
        300-sample CEM population costs one encoder pass, not 300.

        Args:
            ctx_emb: (B, H, P, D) encoded observation history.
            goal_emb: (B, P, D) encoded goal.
            candidates: (B, S, T, A) action blocks to score.
            chunk_size: samples per forward pass (0 = all at once). Lower it if
                the rollout does not fit in memory.

        Returns:
            (B, S) cost per candidate.
        """
        b, s, t, _ = candidates.shape
        chunk = chunk_size if chunk_size > 0 else s
        costs = []
        for start in range(0, s, chunk):
            block = candidates[:, start : start + chunk]
            n = block.size(1)
            # Repeat context and goal for every candidate: (B, ...) -> (B*S, ...)
            emb = ctx_emb.unsqueeze(1).expand(b, n, *ctx_emb.shape[1:]).flatten(0, 1)
            goal = goal_emb.unsqueeze(1).expand(b, n, *goal_emb.shape[1:]).flatten(0, 1)
            preds = self.rollout(emb, block.flatten(0, 1))
            costs.append(self.goal_cost(preds, goal).view(b, n))
        return torch.cat(costs, dim=1)


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def build_model(cfg, action_dim: int, state_dim: int | None = None) -> LeWM:
    """Assemble a LeWM from the config.

    Args:
        action_dim: raw env action dimension (the model sees action_dim * frameskip).
        state_dim: observation width, required only for the oracle encoder.
    """
    mcfg, dim = cfg.model, cfg.model.embed_dim
    encoder = build_encoder(cfg, input_dim=state_dim)

    # its just a simple mlp to encode the action variables to embeddings
    action_encoder = ActionEncoder(
        input_dim=action_dim * cfg.data.frameskip,
        emb_dim=dim,
        smoothed_dim=mcfg.act_smooth_dim,
        mlp_scale=mcfg.act_mlp_scale,
    )

    # Privileged state next to the pixels (`model.oracle_obs`): one more token per frame.
    oracle_encoder = build_oracle_encoder(cfg)

    # Training feeds seq_len-1 positions; rollouts feed at most `history`.
    num_frames = max(cfg.data.history, cfg.data.seq_len - 1)
    predictor = ARPredictor(
        num_frames=num_frames,
        num_patches=encoder.num_patches + (oracle_encoder is not None),
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        depth=mcfg.depth,
        heads=mcfg.heads,
        dim_head=mcfg.dim_head,
        mlp_dim=mcfg.mlp_dim,
        dropout=mcfg.dropout,
    )

    def head():
        return MLP(dim, mcfg.proj_hidden, dim, norm=mcfg.proj_norm) if mcfg.use_projector else None

    return LeWM(
        encoder=encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        projector=head(),
        pred_proj=head(),
        obs_key=cfg.data.obs_key, # number of obs dim  
        context_len=cfg.data.history, # number of data points for dynamics predictor, 
        cost=mcfg.cost,
        freeze_encoder=mcfg.freeze_encoder, # if in case frozen backbone
        oracle_encoder=oracle_encoder,
    )
