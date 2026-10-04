"""Transformer primitives shared by the predictor.

Nothing here is task specific: a causal attention block, an MLP, and a
transformer stack whose blocks can be conditioned on a per-token vector
through AdaLN-zero modulation (the mechanism DiT uses to inject a condition
without destabilising the residual stream at init).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: scale/shift a normalised activation."""
    return x * (1 + scale) + shift


class MLP(nn.Module):
    """Two-layer MLP with optional norm, used for projectors and state encoders."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm: str = "batch",
        act=nn.GELU,
    ):
        super().__init__()
        norms = {"batch": nn.BatchNorm1d, "layer": nn.LayerNorm, "none": None}
        if norm not in norms:
            raise ValueError(f"norm must be one of {list(norms)}, got {norm!r}")
        norm_layer = norms[norm](hidden_dim) if norms[norm] is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) — flatten any time dimension before calling (BatchNorm is 2-d)."""
        return self.net(x)


class FeedForward(nn.Module):
    """Pre-norm position-wise feed-forward."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention. Causal unless given a mask — the predictor is autoregressive."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, D). attn_mask: (N, N) bool, True where a query may attend a key.

        Without a mask, attention is plain causal over the N tokens.
        """
        x = self.norm(x)
        q, k, v = (
            rearrange(t, "b t (h d) -> b h t d", h=self.heads)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        # A length-1 sequence has nothing to attend back to; SDPA's causal mask
        # is fine with it, but skip the flag to keep the fast path predictable.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None and x.size(1) > 1,
        )
        return self.to_out(rearrange(out, "b h t d -> b t (h d)"))


class Block(nn.Module):
    """Plain pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        return x + self.mlp(self.norm2(x))


class ConditionalBlock(nn.Module):
    """Transformer block conditioned on a per-token vector via AdaLN-zero.

    The modulation head is zero-initialised, so at init the block is the
    identity and the condition (here: the action) is eased in during training.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, D) tokens, c: (B, N, D) conditions."""
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(c).chunk(6, dim=-1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a), attn_mask)
        return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class Transformer(nn.Module):
    """Stack of blocks with optional input/output projections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        conditional: bool = True,
    ):
        super().__init__()
        block_cls = ConditionalBlock if conditional else Block
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList(
            block_cls(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c, attn_mask)
        return self.output_proj(self.norm(x))
