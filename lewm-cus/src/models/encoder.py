"""Observation encoders: pixels -> latent tokens, or oracle state -> latent token.

Every encoder has the same contract on flattened batch*time input:

    (N, ...) -> (N, P, D)

where P is the number of latent tokens per observation.  Pooled encoders (the
ViT class token, the state MLP, DINO's class token) give P = 1; DINO patch
features give one token per image patch.  The world model handles any P, so
swapping encoders is a config change only.
"""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from .blocks import MLP


class ViTEncoder(nn.Module):
    """timm Vision Transformer trunk, pooled to one vector per frame.

    vit-tiny is ~5.5M parameters at width 192.

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
        self.num_patches = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> (N, 1, D)"""
        tokens = self.vit.forward_features(x)  # (N, prefix + patches, D)
        if self.pool == "cls" and self.num_prefix > 0:
            return tokens[:, :1]
        return tokens[:, self.num_prefix :].mean(dim=1, keepdim=True)


class DinoEncoder(nn.Module):
    """Pretrained DINOv2 trunk, the observation model of DINO-WM.

    DINO-WM keeps it frozen and predicts its patch features directly; fixed
    targets cannot collapse, so no anti-collapse term is needed. Freezing is
    decided by `model.freeze_encoder`, not here.

    Args:
        name: torch.hub DINOv2 model (`dinov2_vits14` is 384 wide, `dinov2_vitb14` 768).
        tokens: "patch" keeps the spatial grid (DINO-WM), "cls" pools to one vector.
        img_size: frames are resized to this before the trunk.  It must be a
            multiple of the 14px patch; DINO-WM uses 196, a 14x14 grid.
    """

    def __init__(self, name: str = "dinov2_vits14", tokens: str = "patch", img_size: int = 196):
        super().__init__()
        if tokens not in ("patch", "cls"):
            raise ValueError(f"tokens must be 'patch' or 'cls', got {tokens!r}")
        self.dino = torch.hub.load("facebookresearch/dinov2", name)
        patch = self.dino.patch_size
        if img_size % patch:
            raise ValueError(f"dino img_size {img_size} is not a multiple of the {patch}px patch")
        self.tokens = tokens
        self.img_size = img_size
        self.embed_dim = self.dino.embed_dim
        self.num_patches = (img_size // patch) ** 2 if tokens == "patch" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) ImageNet-normalised -> (N, P, D)"""
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False, antialias=True
            )
        features = self.dino.forward_features(x)
        if self.tokens == "patch":
            return features["x_norm_patchtokens"]  # (N, P, D)
        return features["x_norm_clstoken"].unsqueeze(1)  # (N, 1, D)


class StateEncoder(nn.Module):
    """MLP over the low-dimensional oracle state: (N, S) -> (N, 1, D).

    Same interface as the pixel encoders, so `model.encoder=mlp` is the only
    change needed to train on states instead of frames.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, embed_dim: int = 192):
        super().__init__()
        self.net = MLP(input_dim, hidden_dim, embed_dim, norm="batch")
        self.embed_dim = embed_dim
        self.num_patches = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float()).unsqueeze(1)


def build_encoder(cfg, input_dim: int | None = None) -> nn.Module:
    """The encoder named by `cfg.model.encoder`; `input_dim` is the state width (mlp only)."""
    mcfg = cfg.model
    if mcfg.encoder == "vit":
        enc = ViTEncoder(mcfg.vit_name, img_size=cfg.data.img_size, pretrained=mcfg.vit_pretrained, pool=mcfg.vit_pool)
    elif mcfg.encoder == "dino":
        enc = DinoEncoder(mcfg.dino_name, tokens=mcfg.dino_tokens, img_size=mcfg.dino_img_size)
    elif mcfg.encoder == "mlp":
        if input_dim is None:
            raise ValueError("the mlp encoder needs the state dimension")
        enc = StateEncoder(input_dim, mcfg.mlp_hidden, mcfg.embed_dim)
    else:
        raise ValueError(f"unknown encoder {mcfg.encoder!r}")

    if enc.embed_dim != mcfg.embed_dim:
        raise ValueError(f"{mcfg.encoder} encoder width {enc.embed_dim} != model.embed_dim {mcfg.embed_dim}")
    return enc
