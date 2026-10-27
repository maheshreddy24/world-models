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
        patch_size: int | None = None,
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
            **({"patch_size": patch_size} if patch_size else {}),
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


class HFViTEncoder(nn.Module):
    """transformers ViTModel, built as stable_pretraining.backbone.utils.vit_hf builds it for LeWM / Sub-JEPA.

    Same sizes as timm's (tiny: width 192, 12 layers, 3 heads, MLP 4x), no pooling layer, no
    mask token, position encodings interpolated for other input sizes. Its CLS token after the
    final LayerNorm is the frame's latent, as JEPA.encode takes `last_hidden_state[:, 0]`.
    """

    SIZES = {"tiny": (192, 12, 3), "small": (384, 12, 6), "base": (768, 12, 12)}

    def __init__(self, size: str = "tiny", patch_size: int = 14, img_size: int = 224):
        super().__init__()
        from transformers import ViTConfig, ViTModel

        width, depth, heads = self.SIZES[size]
        config = ViTConfig(hidden_size=width, num_hidden_layers=depth, num_attention_heads=heads,
                           intermediate_size=4 * width, image_size=img_size, patch_size=patch_size)
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=False)
        self.embed_dim = width
        self.num_patches = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) -> (N, 1, D)"""
        return self.vit(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, :1]


class DinoEncoder(nn.Module):
    """Pretrained DINOv2 trunk, the observation model of DINO-WM.

    DINO-WM keeps it frozen and predicts its patch features directly; fixed
    targets cannot collapse, so no anti-collapse term is needed. Freezing is
    decided by `model.freeze_encoder`, not here.

    Weights come from the Hugging Face hub (`transformers.Dinov2Model`). Both
    outputs are taken after the final LayerNorm, as DINO-WM does.

    Args:
        name: Hugging Face model id (`facebook/dinov2-small` is 384 wide, `-base` 768).
            The old torch.hub names (`dinov2_vits14`, ...) are mapped to it.
        tokens: "patch" keeps the spatial grid (DINO-WM), "cls" pools to one vector.
        img_size: frames are resized to this before the trunk.  It must be a
            multiple of the 14px patch; DINO-WM uses 196, a 14x14 grid.
    """

    HUB_TO_HF = {
        "dinov2_vits14": "facebook/dinov2-small",
        "dinov2_vitb14": "facebook/dinov2-base",
        "dinov2_vitl14": "facebook/dinov2-large",
        "dinov2_vitg14": "facebook/dinov2-giant",
    }

    def __init__(self, name: str = "facebook/dinov2-small", tokens: str = "patch", img_size: int = 196):
        super().__init__()
        from transformers import Dinov2Model

        if tokens not in ("patch", "cls"):
            raise ValueError(f"tokens must be 'patch' or 'cls', got {tokens!r}")
        self.dino = Dinov2Model.from_pretrained(self.HUB_TO_HF.get(name, name))
        patch = self.dino.config.patch_size
        if img_size % patch:
            raise ValueError(f"dino img_size {img_size} is not a multiple of the {patch}px patch")
        self.tokens = tokens
        self.img_size = img_size
        self.embed_dim = self.dino.config.hidden_size
        self.num_patches = (img_size // patch) ** 2 if tokens == "patch" else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) ImageNet-normalised -> (N, P, D)"""
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(
                x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False, antialias=True
            )
        tokens = self.dino(pixel_values=x).last_hidden_state  # (N, 1 + P, D), layer-normed
        if self.tokens == "patch":
            return tokens[:, 1:]  # (N, P, D)
        return tokens[:, :1]  # (N, 1, D)


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
    if mcfg.encoder == "vit" and getattr(mcfg, "vit_impl", "timm") == "hf":
        size = next(k for k in ("tiny", "small", "base") if k in mcfg.vit_name)
        enc = HFViTEncoder(size, patch_size=mcfg.vit_patch or 16, img_size=cfg.data.img_size)
    elif mcfg.encoder == "vit":
        enc = ViTEncoder(mcfg.vit_name, img_size=cfg.data.img_size, pretrained=mcfg.vit_pretrained, pool=mcfg.vit_pool,
                         patch_size=getattr(mcfg, "vit_patch", None))
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
