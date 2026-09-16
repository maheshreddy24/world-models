"""Sketched Isotropic Gaussian Regulariser.

This is the single term that keeps LeWM from collapsing, replacing EMA targets,
stop-gradients and the usual pile of auxiliary losses.  It pushes the embedding
distribution toward an isotropic Gaussian by testing random 1-d projections of
the batch with the Epps-Pulley statistic: the squared distance between the
empirical characteristic function of a projection and that of a standard normal,
integrated over frequencies with a Gaussian window.

Zero loss means every projection looks standard-normal, which for a distribution
is exactly the isotropic Gaussian — so the embeddings keep full rank and a
constant scale without ever being told what to encode.
"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Args:
        knots: quadrature points on the frequency grid [0, t_max].
        num_proj: random directions sketched per call.
        t_max: highest frequency tested.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        # trapezoidal weights, folded together with the Gaussian window
        weights = torch.full((knots,), 2 * dt)
        weights[0] = weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)  # characteristic fn of N(0,1)
        self.register_buffer("weights", weights * window)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (..., N, D) — the statistic is averaged over any leading dims.

        Pass (T, B, D) to test each timestep's marginal separately, or (N, D)
        to test one pooled batch.
        """
        directions = torch.randn(emb.size(-1), self.num_proj, device=emb.device, dtype=emb.dtype)
        directions = directions / directions.norm(p=2, dim=0, keepdim=True)

        # (..., N, P) projections evaluated at every frequency -> (..., N, P, K)
        x_t = (emb @ directions).unsqueeze(-1) * self.t
        # empirical characteristic function, averaged over the N samples
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * emb.size(-2)
        return statistic.mean()
