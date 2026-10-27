"""Anti-collapse regularisers that drop in where SIGReg goes (`loss.reg_type`).

All of them take the online embedding as (T, N, D), one distribution per
timestep, the way LeWM applies SIGReg, and return a scalar.

    sigreg   SIGReg (src/models/sigreg.py): every random 1-d projection of the
             embedding should look standard normal (Epps-Pulley statistic).
    subjepa  Sub-JEPA (Zhao et al. 2026, arXiv:2605.09241): SIGReg inside K fixed
             random subspaces of dimension D / K instead of the full space; a
             port of sub-jepa/subjepa.py (MultiSubspaceSIGReg, orthogonal_frozen).
    visreg   VISReg (arXiv:2606.02572, Algorithm 1): scale (per-dimension std
             -> 1) + shape (sliced Wasserstein distance of the std-normalised
             embedding to N(0, 1) quantiles, std detached) + centre (mean -> 0).
"""

from __future__ import annotations

import torch
from torch import nn

from .sigreg import SIGReg


class SubspaceSIGReg(nn.Module):
    """SIGReg averaged over K random subspaces (Sub-JEPA).

    Args:
        embed_dim: D.
        num_subspaces: K; each subspace has D / K dimensions (must divide).
        knots, num_proj: as SIGReg; `num_proj` directions are drawn per subspace.
        init: "orthogonal_frozen" (the paper's setting: orthonormal rows, fixed)
            or "random_frozen" (Gaussian rows, fixed).
    """

    def __init__(self, embed_dim: int, num_subspaces: int, knots: int = 17, num_proj: int = 1024,
                 init: str = "orthogonal_frozen"):
        super().__init__()
        if embed_dim % num_subspaces:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by {num_subspaces} subspaces")
        sub = embed_dim // num_subspaces
        mats = []
        for _ in range(num_subspaces):
            if init == "orthogonal_frozen":
                q, _ = torch.linalg.qr(torch.randn(embed_dim, sub), mode="reduced")
                mats.append(q.T)
            elif init == "random_frozen":
                mats.append(torch.randn(sub, embed_dim))
            else:
                raise ValueError(f"unknown subspace init {init!r}")
        self.register_buffer("projections", torch.stack(mats))  # (K, d', D)
        self.sigreg = SIGReg(knots, num_proj)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (T, N, D) -> scalar."""
        proj = torch.einsum("tnd,ked->ktne", emb, self.projections.to(emb.dtype))  # (K, T, N, d')
        k, d_sub = proj.size(0), proj.size(-1)
        # independent random directions in every subspace, as subjepa.py
        a = torch.randn(k, d_sub, self.sigreg.num_proj, device=emb.device, dtype=emb.dtype)
        a = a / a.norm(p=2, dim=1, keepdim=True)
        x_t = torch.einsum("ktne,kep->ktnp", proj, a).unsqueeze(-1) * self.sigreg.t  # (K, T, N, P, knots)
        err = (x_t.cos().mean(2) - self.sigreg.phi).square() + x_t.sin().mean(2).square()
        statistic = (err @ self.sigreg.weights) * proj.size(2)
        return statistic.mean()


class VISReg(nn.Module):
    """VISReg = scale + shape + centre (each weight 1, the paper's default).

    Computed in float32: sorting and quantile matching are not bf16-safe.

    Args:
        num_proj: random directions for the sliced Wasserstein term.
        eps: added to the detached std before normalising.
    """

    def __init__(self, num_proj: int = 2048, eps: float = 1e-4):
        super().__init__()
        self.num_proj = num_proj
        self.eps = eps

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (T, N, D) -> scalar, averaged over T."""
        with torch.autocast(emb.device.type, enabled=False):
            z = emb.float()
            n, d = z.size(-2), z.size(-1)
            mu = z.mean(-2, keepdim=True)  # (T, 1, D)
            center = mu.square().mean()
            zc = z - mu
            std = zc.std(-2, unbiased=False, keepdim=True)  # (T, 1, D)
            scale = (1.0 - std).square().mean()

            zn = zc / (std.detach() + self.eps)
            w = torch.randn(d, self.num_proj, device=z.device)
            w = w / w.norm(p=2, dim=0, keepdim=True)
            p = torch.sort(zn @ w, dim=-2).values  # (T, N, P)
            u = torch.arange(1, n + 1, device=z.device, dtype=torch.float32) / (n + 1)
            target = torch.distributions.Normal(0.0, 1.0).icdf(u)[:, None]  # (N, 1)
            shape = (p - target).square().mean()
        return scale + shape + center


def build_regularizer(cfg) -> nn.Module | None:
    """The anti-collapse term named by `cfg.loss.reg_type`, or None when `loss.use_sigreg` is off."""
    lcfg = cfg.loss
    if not lcfg.use_sigreg:
        return None
    if lcfg.reg_type == "sigreg":
        return SIGReg(lcfg.sigreg_knots, lcfg.sigreg_proj)
    if lcfg.reg_type == "subjepa":
        return SubspaceSIGReg(cfg.model.embed_dim, lcfg.subspaces, lcfg.sigreg_knots, lcfg.sigreg_proj,
                              lcfg.subspace_init)
    if lcfg.reg_type == "visreg":
        return VISReg(lcfg.visreg_proj)
    raise ValueError(f"unknown loss.reg_type {lcfg.reg_type!r}")
