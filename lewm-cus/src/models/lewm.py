"""LeWM: a joint-embedding predictive world model.

Four parts:

    encoder         observation  -> latent tokens
    action_encoder  action block -> latent
    predictor       (latents, actions) -> next latent
    projector(s)    optional MLP heads after the encoder and after the predictor

Prediction happens entirely in latent space; nothing is decoded back to
pixels. What stops a trainable encoder from collapsing to a constant is either
the SIGReg term in `loss()`, which keeps the embedding distribution isotropic
Gaussian, or an EMA target (`ema=True`): the targets come from a slowly moving
copy of encoder + projector with no gradient (BYOL / I-JEPA). A frozen encoder
(DINO-WM) cannot collapse, so it runs with neither.

With an EMA target, `encode()` (what rollouts, probe and decoder see) uses the
target branch: the predictor's outputs live in that space.

Shapes:

    B  batch       T  latent steps      P  tokens per frame (1 pooled, 196 DINO patches)
    D  embed_dim   A  action_dim * frameskip

Latent step t is the frame at row t * frameskip; action t is the block of
`frameskip` raw actions that carries step t to step t + 1. So
`predict(emb[:, :k+1], act[:, :k+1])[:, -1]` is the model's guess at `emb[:, k+1]`.
"""

from __future__ import annotations

import copy

import torch
from einops import rearrange
from torch import nn

from ..data.transforms import ImageTransform
from .blocks import MLP
from .encoder import build_encoder
from .predictor import ActionEncoder, ARPredictor
from .sigreg import SIGReg


def latent_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared L2 per token, averaged over tokens: (..., P, D) -> (...).

    Averaging over P keeps a patch-grid latent on the scale of a single pooled
    vector (identical when P = 1).
    """
    return (a - b).pow(2).sum(-1).mean(-1)


def _per_token(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply an (N, D) -> (N, D) head to every token of (..., D); BatchNorm sees tokens as samples."""
    return head(x.reshape(-1, x.size(-1))).reshape(x.shape)


class LeWM(nn.Module):
    """Args:
        encoder: `(N, ...) -> (N, P, D)`.
        action_encoder: `(B, T, A) -> (B, T, D)`.
        predictor: causal transformer, `(B, T, P, D), (B, T, D) -> (B, T, P, D)`.
        projector, pred_proj: heads after the encoder / predictor (identity if None).
        obs_key: the batch key holding the observation.
        context_len: latent frames the predictor sees at inference (= data.history).
        freeze_encoder: no gradients for the encoder, and it always stays in eval mode.
        ema: keep an EMA copy of encoder + projector as the prediction target
            (updated by `update_target`).
        pixel_transform: applied on the model's device to uint8 frames (B, T, H, W, 3),
            as the training loader ships them; float input is taken as already transformed.
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
        freeze_encoder: bool = False,
        ema: bool = False,
        pixel_transform: ImageTransform | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.obs_key = obs_key
        self.context_len = context_len
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.ema = ema
        self.pixel_transform = pixel_transform
        if ema:
            self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
            self.target_projector = copy.deepcopy(self.projector).requires_grad_(False)

    def train(self, mode: bool = True) -> "LeWM":
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------ #
    #  Building blocks
    # ------------------------------------------------------------------ #
    def _encode_with(self, encoder: nn.Module, projector: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        b = obs.size(0)
        x = rearrange(obs, "b t ... -> (b t) ...")
        if obs.dtype == torch.uint8 and self.pixel_transform is not None:
            x = self.pixel_transform(x)  # (B*T, H, W, 3) uint8 -> (B*T, 3, S, S)
        tokens = encoder(x.float())  # (B*T, P, D)
        tokens = _per_token(projector, tokens)
        return rearrange(tokens, "(b t) p d -> b t p d", b=b)

    def encode_online(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, ...) -> (B, T, P, D) through the trained encoder (the predictor's input in training)."""
        return self._encode_with(self.encoder, self.projector, obs)

    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, ...) -> (B, T, P, D) in the space the predictor predicts (the EMA branch, if any)."""
        if not self.ema:
            return self.encode_online(obs)
        with torch.no_grad():
            return self._encode_with(self.target_encoder, self.target_projector, obs)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, T, ...) -> (B, T, P, D), the latent used at inference."""
        return self.encode_target(obs)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """target <- momentum * target + (1 - momentum) * online; buffers (BatchNorm stats) copied."""
        if not self.ema:
            return
        for online, target in ((self.encoder, self.target_encoder), (self.projector, self.target_projector)):
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.lerp_(po.detach(), 1.0 - momentum)
            for bo, bt in zip(online.buffers(), target.buffers()):
                bt.copy_(bo)

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, T, A) -> (B, T, D). NaNs become zeros."""
        return self.action_encoder(torch.nan_to_num(action, 0.0))

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """One causal pass; output frame t is the prediction of frame t + 1.

        emb: (B, T, P, D), act_emb: (B, T, D) -> (B, T, P, D)
        """
        return _per_token(self.pred_proj, self.predictor(emb, act_emb))

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def loss(self, batch: dict, sigreg: nn.Module | None = None, sigreg_weight: float = 0.0,
             pred_weight: float = 1.0) -> dict:
        """Teacher-forced next-embedding prediction, plus SIGReg (or another regulariser) when one is given.

        loss = pred_weight * pred_loss + sigreg_weight * regulariser; `pred_weight` < 1 is
        VISReg's convex form, (1 - lambda) * pred + lambda * reg.

        A sequence of T steps gives T - 1 supervised predictions in one pass.
        With an EMA target, the predictor reads the online embeddings and
        predicts the (gradient-free) EMA ones.
        """
        emb = self.encode_online(batch[self.obs_key])  # (B, T, P, D)
        act_emb = self.encode_action(batch["action"])  # (B, T, D)

        pred = self.predict(emb[:, :-1], act_emb[:, :-1])
        target_emb = self.encode_target(batch[self.obs_key]) if self.ema else emb
        target = target_emb[:, 1:]
        pred_loss = (pred - target).pow(2).mean()
        out = {"loss": pred_loss, "pred_loss": pred_loss.detach()}

        if sigreg is not None:
            # (T, B*P, D): each timestep's token distribution is tested against N(0, I)
            sigreg_loss = sigreg(rearrange(emb, "b t p d -> t (b p) d"))
            out["loss"] = pred_weight * pred_loss + sigreg_weight * sigreg_loss
            out["sigreg_loss"] = sigreg_loss.detach()

        with torch.no_grad():
            out["emb_std"] = emb.std(dim=(0, 1, 2)).mean()  # heads to 0 on collapse
            out["emb_norm"] = emb.norm(dim=-1).mean()
            # < 1: the predictor beats "nothing moves"
            static = (target_emb[:, :-1] - target).pow(2).mean()
            out["pred_vs_static"] = pred_loss / static.clamp_min(1e-8)
        return out

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def rollout(self, ctx_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll the predictor forward on its own output.

        Every step feeds the last `context_len` latents (real ones first, then
        predictions) plus their action blocks, and appends the prediction.

        Args:
            ctx_emb: (B, H, P, D) latents of the H observed frames.
            actions: (B, T, A) action blocks 0..T-1, with T >= H.

        Returns:
            (B, T - H + 1, P, D) predictions of frames H .. T.
        """
        h, t = ctx_emb.size(1), actions.size(1)
        if t < h:
            raise ValueError(f"need at least {h} action blocks for a {h}-frame context, got {t}")

        act_emb = self.encode_action(actions)
        emb = ctx_emb
        for k in range(h - 1, t):
            lo = max(0, k - self.context_len + 1)
            nxt = self.predict(emb[:, lo : k + 1], act_emb[:, lo : k + 1])[:, -1:]
            emb = torch.cat([emb, nxt], dim=1)
        return emb[:, h:]


def build_model(cfg, action_dim: int, state_dim: int | None = None) -> LeWM:
    """Assemble a LeWM from the config.

    Args:
        action_dim: raw action width (the model sees action_dim * frameskip per step).
        state_dim: oracle-state width, needed only by the mlp encoder.
    """
    mcfg, dim = cfg.model, cfg.model.embed_dim
    encoder = build_encoder(cfg, input_dim=state_dim)
    action_encoder = ActionEncoder(
        input_dim=action_dim * cfg.data.frameskip,
        emb_dim=dim,
        smoothed_dim=mcfg.act_smooth_dim,
        mlp_scale=mcfg.act_mlp_scale,
    )
    predictor = ARPredictor(
        num_frames=max(cfg.data.history, cfg.data.seq_len - 1),  # training feeds seq_len - 1 frames
        num_patches=encoder.num_patches,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        depth=mcfg.depth,
        heads=mcfg.heads,
        dim_head=mcfg.dim_head,
        mlp_dim=mcfg.mlp_dim,
        dropout=mcfg.dropout,
        pos_init_std=getattr(mcfg, "pos_init_std", 0.02),
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
        freeze_encoder=mcfg.freeze_encoder,
        ema=cfg.loss.ema,
        pixel_transform=ImageTransform(cfg.data.img_size) if cfg.data.obs_key == "pixels" else None,
    )
