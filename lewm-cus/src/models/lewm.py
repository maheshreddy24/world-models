"""LeWM: a joint-embedding predictive world model.

The model is four parts and one idea:

    encoder        observation  -> latent
    action_encoder action block -> latent
    predictor      (latents, actions) -> next latent
    projector(s)   optional heads around the encoder / predictor output

The idea is that prediction happens entirely in latent space — nothing is ever
decoded back to pixels.  What stops the latent from collapsing to a constant is
not a stop-gradient or an EMA target but the SIGReg term in `loss()`, which
keeps the embedding distribution isotropic Gaussian.

Conventions used throughout:

    B  batch          T  latent steps      D  embed_dim
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
from .encoder import build_encoder
from .predictor import ARPredictor, ActionEncoder
from .sigreg import SIGReg


class LeWM(nn.Module):
    """Args:
        encoder: observation encoder, `(N, ...) -> (N, D)`.
        action_encoder: action-block encoder, `(B, T, A) -> (B, T, D)`.
        predictor: autoregressive latent dynamics model.
        projector: head applied to encoder output (identity if None).
        pred_proj: head applied to predictor output (identity if None).
        obs_key: which batch key holds the observation.
        context_len: max latent steps fed to the predictor at inference.
        cost: planning cost, one of "final_mse" | "mean_mse" | "final_cosine".
    """

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

    # ------------------------------------------------------------------ #
    #  Encoding
    # ------------------------------------------------------------------ #
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, ...) -> (B, T, D). Time is folded into the batch."""
        b = obs.size(0)
        flat = rearrange(obs.float(), "b t ... -> (b t) ...")
        emb = self.projector(self.encoder(flat))
        return rearrange(emb, "(b t) d -> b t d", b=b)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, T, A) -> (B, T, D). NaNs (sequence padding) become zeros."""
        return self.action_encoder(torch.nan_to_num(action, 0.0))

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """One causal pass: slot `t` predicts the embedding at `t + 1`."""
        b = emb.size(0)
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=b)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def loss(self, batch: dict, sigreg: SIGReg, sigreg_weight: float) -> dict:
        """Teacher-forced next-embedding prediction + the collapse regulariser.

        Every step of the sequence except the last predicts its successor, so a
        sequence of T steps gives T-1 supervised predictions.
        """
        emb = self.encode(batch[self.obs_key])  # (B, T, D)
        act_emb = self.encode_action(batch["action"])  # (B, T, D)

        pred = self.predict(emb[:, :-1], act_emb[:, :-1])  # (B, T-1, D)
        target = emb[:, 1:]  # (B, T-1, D)

        pred_loss = (pred - target).pow(2).mean()
        # (T, B, D): each timestep's marginal is tested against N(0, I) separately
        sigreg_loss = sigreg(emb.transpose(0, 1))

        out = {
            "loss": pred_loss + sigreg_weight * sigreg_loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }
        with torch.no_grad():
            # Collapse announces itself as emb_std -> 0 long before the loss moves.
            out["emb_std"] = emb.std(dim=(0, 1)).mean()
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
            ctx_emb: (B, H, D) embeddings of the H observed steps.
            actions: (B, T, A) action blocks for steps 0..T-1, with T >= H.

        Returns:
            (B, T - H + 1, D) predictions for steps H .. T. The last one is the
            latent reached after executing every action block.
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

        preds: (N, T, D), goal_emb: (N, D) -> (N,)
        """
        goal = goal_emb.unsqueeze(1)
        if self.cost == "final_mse":
            return (preds[:, -1:] - goal).pow(2).sum(dim=-1).squeeze(1)
        if self.cost == "mean_mse":
            return (preds - goal).pow(2).sum(dim=-1).mean(dim=1)
        if self.cost == "final_cosine":
            return 1.0 - F.cosine_similarity(preds[:, -1], goal_emb, dim=-1)
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
            ctx_emb: (B, H, D) encoded observation history.
            goal_emb: (B, D) encoded goal.
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
            emb = ctx_emb.unsqueeze(1).expand(b, n, *ctx_emb.shape[1:])
            emb = rearrange(emb, "b s h d -> (b s) h d")
            acts = rearrange(block, "b s t a -> (b s) t a")
            preds = self.rollout(emb, acts)
            goal = goal_emb.unsqueeze(1).expand(b, n, -1).reshape(b * n, -1)
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

    action_encoder = ActionEncoder(
        input_dim=action_dim * cfg.data.frameskip,
        emb_dim=dim,
        smoothed_dim=mcfg.act_smooth_dim,
        mlp_scale=mcfg.act_mlp_scale,
    )

    # Training feeds seq_len-1 positions; rollouts feed at most `history`.
    num_frames = max(cfg.data.history, cfg.data.seq_len - 1)
    predictor = ARPredictor(
        num_frames=num_frames,
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
        obs_key=cfg.data.obs_key,
        context_len=cfg.data.history,
        cost=mcfg.cost,
    )
