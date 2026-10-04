"""Action encoder and the autoregressive latent dynamics predictor."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .blocks import Transformer


class ActionEncoder(nn.Module):
    """Embed a block of raw actions into the latent width.

    One "step" of the world model covers `frameskip` env steps, so the input is
    the flattened block of those raw actions.  The 1x1 convolution mixes action
    dimensions before the MLP lifts the block to `emb_dim`.
    """

    def __init__(self, input_dim: int, emb_dim: int = 192, smoothed_dim: int = 64, mlp_scale: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, T, input_dim) -> (B, T, emb_dim)"""
        x = action.float().transpose(1, 2)  # conv1d wants channels first
        x = self.patch_embed(x).transpose(1, 2)
        return self.embed(x)


def frame_causal_mask(frames: int, patches: int, device=None) -> torch.Tensor | None:
    """(T*P, T*P) bool mask: a token attends to every token of its own and earlier frames.

    This is the DINO-WM mask. With one token per frame it is plain causal
    attention, so None is returned and SDPA's faster built-in causal path is used.
    """
    if patches == 1:
        return None
    frame = torch.arange(frames, device=device).repeat_interleave(patches)
    return frame[:, None] >= frame[None, :]


class ARPredictor(nn.Module):
    """Predict the next latent from the latent history and the actions taken.

    Transformer over `(B, T, P, D)` latents (T frames of P tokens each), flattened
    to T*P tokens with attention causal between frames, so one pass trains all
    T next-frame predictions.  The action embedding of each frame reaches every
    token of that frame through AdaLN-zero.  Output frame `t` is the prediction
    of frame `t+1`.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        num_patches: int = 1,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        depth: int = 6,
        heads: int = 6,
        dim_head: int = 64,
        mlp_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim) * 0.02)  # which frame
        # Which token within a frame. Only created for P > 1, so checkpoints of
        # pooled encoders keep exactly the parameters they were saved with.
        self.patch_pos_embedding = (
            nn.Parameter(torch.randn(1, 1, num_patches, input_dim) * 0.02) if num_patches > 1 else None
        )
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            conditional=True,
        )

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """emb: (B, T, P, D), act_emb: (B, T, D) -> (B, T, P, D) next-frame predictions."""
        _, t, p, _ = emb.shape
        if t > self.num_frames:
            raise ValueError(
                f"context of {t} steps exceeds the {self.num_frames} position "
                "embeddings; raise data.history or shorten the window"
            )
        if p != self.num_patches:
            raise ValueError(f"expected {self.num_patches} tokens per frame, got {p}")
        if act_emb.size(1) != t:
            raise ValueError(f"got {t} embeddings but {act_emb.size(1)} actions")

        x = emb + self.pos_embedding[:, :t, None]
        if self.patch_pos_embedding is not None:
            x = x + self.patch_pos_embedding
        cond = act_emb[:, :, None].expand(-1, -1, p, -1)  # a frame's action conditions all its tokens

        x = rearrange(x, "b t p d -> b (t p) d")
        cond = rearrange(cond, "b t p d -> b (t p) d")
        out = self.transformer(x, cond, attn_mask=frame_causal_mask(t, p, x.device))
        return rearrange(out, "b (t p) d -> b t p d", t=t)
