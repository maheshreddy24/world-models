"""Observation encoders: pixels -> latent, or oracle state -> latent.

Both expose the same contract — `(N, ...) -> (N, embed_dim)` on flattened
batch*time input — so the world model does not care which one it holds.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from .blocks import MLP


class ViTEncoder(nn.Module):
    """timm Vision Transformer trunk, pooled to one vector per frame.

    vit-tiny is ~5.5M parameters at width 192, which is what makes the whole
    world model trainable on a single GPU.

    Args:
        name: any timm ViT (`vit_tiny_patch16_224`, `vit_small_patch16_224`, ...).
        img_size: input resolution; position embeddings are interpolated to fit.
        pretrained: LeWM trains from scratch, so this is False by default.
        pool: "cls" takes the class token, "mean" averages the patch tokens.
    """

    def __init__(
        self,
        name: str = "vit_tiny_patch16_224",
        img_size: int = 224,
        pretrained: bool = False,
        pool: str = "cls",
    ):
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean', got {pool!r}")
        self.pool = pool
        self.vit = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,  # tolerate resolutions the checkpoint never saw
        )
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        self.embed_dim = self.vit.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> (N, D)"""
        tokens = self.vit.forward_features(x)  # (N, prefix + patches, D)
        if self.pool == "cls" and self.num_prefix > 0:
            return tokens[:, 0]
        return tokens[:, self.num_prefix :].mean(dim=1)


class StateEncoder(nn.Module):
    """MLP over a low-dimensional state — the oracle ablation.

    Same interface as the ViT, so swapping `model.encoder` between "vit" and
    "mlp" is the only change needed to train on states instead of pixels.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, embed_dim: int = 192):
        super().__init__()
        self.net = MLP(input_dim, hidden_dim, embed_dim, norm="batch")
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, input_dim) -> (N, D)"""
        return self.net(x.float())


def build_encoder(cfg, input_dim: int | None = None) -> nn.Module:
    """Instantiate the encoder named by `cfg.model.encoder`."""
    mcfg = cfg.model
    if mcfg.encoder == "vit":
        enc = ViTEncoder(
            name=mcfg.vit_name,
            img_size=cfg.data.img_size,
            pretrained=mcfg.vit_pretrained,
            pool=mcfg.vit_pool,
        )
    elif mcfg.encoder == "mlp":
        if input_dim is None:
            raise ValueError("the mlp encoder needs the state dimension")
        enc = StateEncoder(input_dim, mcfg.mlp_hidden, mcfg.embed_dim)
    else:
        raise ValueError(f"unknown encoder {mcfg.encoder!r}")

    if enc.embed_dim != mcfg.embed_dim:
        raise ValueError(
            f"encoder width {enc.embed_dim} != model.embed_dim {mcfg.embed_dim}; "
            f"set model.embed_dim={enc.embed_dim} for {mcfg.vit_name}"
        )
    return enc
