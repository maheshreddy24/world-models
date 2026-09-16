"""Action encoder and the autoregressive latent dynamics predictor."""

from __future__ import annotations

import torch
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


class ARPredictor(nn.Module):
    """Predict the next latent from the latent history and the actions taken.

    Causal transformer over `(B, T, D)` embeddings, with the action embedding of
    each step injected into that step's block through AdaLN-zero.  Output slot
    `t` is the prediction of embedding `t+1`.
    """

    def __init__(
        self,
        *,
        num_frames: int,
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
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim) * 0.02)
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
        """emb: (B, T, D), act_emb: (B, T, D) -> (B, T, D) next-step predictions."""
        t = emb.size(1)
        if t > self.num_frames:
            raise ValueError(
                f"context of {t} steps exceeds the {self.num_frames} position "
                "embeddings; raise data.history or shorten the window"
            )
        if act_emb.size(1) != t:
            raise ValueError(f"got {t} embeddings but {act_emb.size(1)} actions")
        return self.transformer(emb + self.pos_embedding[:, :t], act_emb)
