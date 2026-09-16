"""Sampling-based trajectory optimisers.

Each solver minimises a black-box cost over action sequences.  They know
nothing about the world model — they are handed a `cost_fn` that maps
`(B, S, H, A)` action candidates to `(B, S)` costs, which is what makes them
reusable with any model and easy to swap from the config (`plan.solver`).

    B  environments planned in parallel
    S  candidate sequences per environment
    H  planning horizon in latent steps
    A  action_dim * frameskip (one latent step commits to a block of actions)
"""

from __future__ import annotations

from typing import Callable

import torch

CostFn = Callable[[torch.Tensor], torch.Tensor]


class Solver:
    """Common sampling/bookkeeping shared by CEM and MPPI.

    Args:
        horizon: latent steps to optimise.
        action_dim: width of one action block.
        num_samples: candidates drawn per iteration.
        n_iters: refinement iterations per solve.
        var_scale: initial standard deviation of the search distribution.
        min_std: floor on the std, so the search never fully collapses.
        momentum: how much of the previous mean to keep, in [0, 1).
        bounds: `(low, high)` clamp applied to candidates, in the same
            (normalised) space the model was trained on.
        device, seed, dtype: self-explanatory.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 300,
        n_iters: int = 30,
        var_scale: float = 1.0,
        min_std: float = 0.01,
        momentum: float = 0.0,
        bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        device: str | torch.device = "cuda",
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.var_scale = var_scale
        self.min_std = min_std
        self.momentum = momentum
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.bounds = None
        if bounds is not None:
            low, high = bounds
            self.bounds = (
                torch.as_tensor(low, device=self.device, dtype=dtype),
                torch.as_tensor(high, device=self.device, dtype=dtype),
            )

    # ---- helpers -------------------------------------------------------- #
    def _init(self, batch_size: int, init_mean: torch.Tensor | None):
        shape = (batch_size, self.horizon, self.action_dim)
        mean = torch.zeros(shape, device=self.device, dtype=self.dtype)
        if init_mean is not None:
            init_mean = init_mean.to(self.device, self.dtype)
            steps = min(init_mean.size(1), self.horizon)
            mean[:, :steps] = init_mean[:, :steps]
        std = torch.full(shape, self.var_scale, device=self.device, dtype=self.dtype)
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            mean.size(0), self.num_samples, self.horizon, self.action_dim,
            generator=self.generator, device=self.device, dtype=self.dtype,
        )
        candidates = mean.unsqueeze(1) + std.unsqueeze(1) * noise
        # Always keep the incumbent, so a solve can never make things worse.
        candidates[:, 0] = mean
        if self.bounds is not None:
            candidates = candidates.clamp(self.bounds[0], self.bounds[1])
        return candidates

    @torch.no_grad()
    def solve(self, cost_fn: CostFn, batch_size: int, init_mean=None) -> dict:
        raise NotImplementedError


class CEMSolver(Solver):
    """Cross-entropy method: refit the sampling distribution to the elites.

    Args:
        topk: number of elite candidates kept per iteration.
    """

    def __init__(self, *args, topk: int = 30, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk

    @torch.no_grad()
    def solve(self, cost_fn: CostFn, batch_size: int, init_mean=None) -> dict:
        mean, std = self._init(batch_size, init_mean)
        k = min(self.topk, self.num_samples)
        cost = None

        for _ in range(self.n_iters):
            candidates = self._sample(mean, std)
            cost = cost_fn(candidates)  # (B, S)

            _, elite_idx = torch.topk(cost, k, dim=1, largest=False)
            elites = torch.take_along_dim(candidates, elite_idx[..., None, None], dim=1)

            new_mean = elites.mean(dim=1)
            new_std = elites.std(dim=1).clamp_min(self.min_std)
            mean = self.momentum * mean + (1 - self.momentum) * new_mean
            std = self.momentum * std + (1 - self.momentum) * new_std

        best_cost = cost.min(dim=1).values if cost is not None else None
        return {"actions": mean, "std": std, "cost": best_cost}


class MPPISolver(Solver):
    """Model-predictive path integral: exponentially-weighted average of samples.

    Softer than CEM's hard elite cut — every candidate contributes, weighted by
    `exp(-cost / temperature)`.

    Args:
        temperature: lower is greedier; costs are shifted by their per-env
            minimum first, so the scale is relative and numerically safe.
    """

    def __init__(self, *args, temperature: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    @torch.no_grad()
    def solve(self, cost_fn: CostFn, batch_size: int, init_mean=None) -> dict:
        mean, std = self._init(batch_size, init_mean)
        cost = None

        for _ in range(self.n_iters):
            candidates = self._sample(mean, std)
            cost = cost_fn(candidates)

            advantage = cost - cost.min(dim=1, keepdim=True).values
            weights = torch.softmax(-advantage / self.temperature, dim=1)  # (B, S)
            w = weights[..., None, None]

            mean = (w * candidates).sum(dim=1)
            var = (w * (candidates - mean.unsqueeze(1)).pow(2)).sum(dim=1)
            std = var.sqrt().clamp_min(self.min_std)

        best_cost = cost.min(dim=1).values if cost is not None else None
        return {"actions": mean, "std": std, "cost": best_cost}


def build_solver(cfg, action_dim: int, bounds=None, device="cuda", dtype=torch.float32) -> Solver:
    """Instantiate the solver named by `plan.solver`."""
    p = cfg.plan
    common = dict(
        horizon=p.horizon,
        action_dim=action_dim,
        num_samples=p.num_samples,
        n_iters=p.n_iters,
        var_scale=p.var_scale,
        min_std=p.min_std,
        momentum=p.momentum,
        bounds=bounds,
        device=device,
        seed=cfg.seed,
        dtype=dtype,
    )
    if p.solver == "cem":
        return CEMSolver(**common, topk=p.topk)
    if p.solver == "mppi":
        return MPPISolver(**common, temperature=p.temperature)
    raise ValueError(f"unknown solver {p.solver!r}; expected 'cem' or 'mppi'")
