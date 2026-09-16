"""Receding-horizon control on top of a latent world model.

Each call to `act()` either pops a pending action or, when the buffer runs dry,
replans: encode the observation history and the goal, let the solver minimise
the model's latent distance to the goal over action blocks, keep the first
`receding_horizon` blocks and queue the raw env actions they unpack into.

Two things are worth knowing about the spaces involved:

* the solver works in *normalised* action space (the same z-scoring the model
  was trained with), and actions are denormalised and clipped to the env's Box
  only on the way out;
* one solver step is a *block* of `action_block` raw env steps, so a horizon of
  5 with a block of 5 plans 25 env steps ahead.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .solvers import Solver


class MPCPlanner:
    """Args:
        model: a `LeWM` in eval mode.
        solver: the trajectory optimiser.
        cfg: the full config (uses `plan`, `data`, `device`).
        normalizer: the z-score statistics used during training.
        image_transform: pixel preprocessing, identical to training.
        action_space_bounds: `(low, high)` of the raw env action space.
    """

    def __init__(
        self,
        model,
        solver: Solver,
        cfg,
        normalizer,
        image_transform=None,
        action_space_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.model = model
        self.solver = solver
        self.cfg = cfg
        self.plan = cfg.plan
        self.normalizer = normalizer
        self.image_transform = image_transform
        self.device = torch.device(cfg.device)
        self.obs_key = cfg.data.obs_key
        self.action_dim = normalizer.dim("action")
        self.bounds = action_space_bounds

        self.num_envs = 0
        self.history: list[deque] = []
        self.queue: list[deque] = []
        self.warm_start: torch.Tensor | None = None
        self.goal_emb: torch.Tensor | None = None
        self.last_cost: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, num_envs: int) -> None:
        """Clear all per-episode state for `num_envs` parallel environments."""
        self.num_envs = num_envs
        self.history = [deque(maxlen=self.plan.history_len) for _ in range(num_envs)]
        self.queue = [deque() for _ in range(num_envs)]
        self.warm_start = None
        self.goal_emb = None
        self.last_cost = None

    @torch.no_grad()
    def set_goal(self, goal_obs) -> None:
        """Encode the goal once per episode — it never changes during a rollout."""
        obs = self._to_model_input(goal_obs)  # (N, ...)
        self.goal_emb = self.model.encode(obs.unsqueeze(1))[:, 0]  # (N, D)

    # ------------------------------------------------------------------ #
    #  Acting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, active: np.ndarray | None = None) -> np.ndarray:
        """Return one raw env action per environment.

        Args:
            obs: current observation for every env, batched on axis 0.
            active: bool mask of envs still running; inactive ones get zeros
                and are never planned for.

        Returns:
            (num_envs, action_dim) float32 actions inside the env's action space.
        """
        if self.goal_emb is None:
            raise RuntimeError("call set_goal() before act()")

        active = np.ones(self.num_envs, dtype=bool) if active is None else np.asarray(active, bool)
        processed = self._to_model_input(obs)  # (N, ...)
        for i in range(self.num_envs):
            self.history[i].append(processed[i])

        replan = [i for i in range(self.num_envs) if active[i] and not self.queue[i]]
        if replan:
            self._replan(replan)

        actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        for i in range(self.num_envs):
            if active[i] and self.queue[i]:
                actions[i] = self.queue[i].popleft()
        return actions

    def _replan(self, envs: list[int]) -> None:
        idx = torch.as_tensor(envs, dtype=torch.long)
        ctx = self._context(envs)  # (n, H0, ...)
        ctx_emb = self.model.encode(ctx)  # (n, H0, D)
        goal_emb = self.goal_emb[idx.to(self.goal_emb.device)]

        def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
            return self.model.plan_cost(ctx_emb, goal_emb, candidates, self.plan.chunk_size)

        init = self.warm_start[idx] if (self.plan.warm_start and self.warm_start is not None) else None
        out = self.solver.solve(cost_fn, batch_size=len(envs), init_mean=init)
        plan = out["actions"]  # (n, horizon, action_dim * block), normalised

        if out["cost"] is not None:
            self.last_cost = np.full(self.num_envs, np.nan, dtype=np.float32)
            self.last_cost[envs] = out["cost"].float().cpu().numpy()

        self._store_warm_start(idx, plan)

        # Unpack the executed part of the plan into raw env actions.
        keep = plan[:, : self.plan.receding_horizon]
        raw = keep.reshape(len(envs), -1, self.action_dim).float().cpu().numpy()
        raw = self.normalizer.denormalize("action", raw)
        if self.bounds is not None:
            raw = np.clip(raw, self.bounds[0], self.bounds[1])
        for row, env_i in enumerate(envs):
            self.queue[env_i].extend(raw[row])

    def _store_warm_start(self, idx: torch.Tensor, plan: torch.Tensor) -> None:
        """Shift the unexecuted tail of the plan forward to seed the next solve."""
        if not self.plan.warm_start:
            return
        if self.warm_start is None:
            self.warm_start = torch.zeros(
                self.num_envs, self.plan.horizon, plan.size(-1), dtype=plan.dtype, device=plan.device
            )
        tail = plan[:, self.plan.receding_horizon :]
        shifted = torch.zeros_like(self.warm_start[idx])
        if tail.size(1):
            shifted[:, : tail.size(1)] = tail
        self.warm_start[idx] = shifted

    # ------------------------------------------------------------------ #
    #  Observation plumbing
    # ------------------------------------------------------------------ #
    def _to_model_input(self, obs) -> torch.Tensor:
        """Raw env observation -> the exact tensor the encoder saw in training."""
        if self.obs_key == "pixels":
            x = self.image_transform(np.asarray(obs))
        else:
            x = torch.as_tensor(self.normalizer.normalize(self.obs_key, np.asarray(obs, np.float32)))
        return x.to(self.device, torch.float32)

    def _context(self, envs: list[int]) -> torch.Tensor:
        """Stack each env's observation history into (n, H0, ...).

        Early in an episode there are fewer frames than `history_len`; the
        oldest available frame is repeated so the shape stays fixed.
        """
        rows = []
        for i in envs:
            frames = list(self.history[i])
            pad = [frames[0]] * (self.plan.history_len - len(frames))
            rows.append(torch.stack(pad + frames))
        return torch.stack(rows)


class RandomPlanner:
    """Uniform action sampling — the baseline every planning number is judged against."""

    def __init__(self, action_space_bounds, action_dim: int, seed: int = 0):
        self.low, self.high = action_space_bounds
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.num_envs = 0
        self.last_cost = None

    def reset(self, num_envs: int) -> None:
        self.num_envs = num_envs

    def set_goal(self, goal_obs) -> None:  # noqa: D102 - nothing to do
        pass

    def act(self, obs, active=None) -> np.ndarray:
        actions = self.rng.uniform(self.low, self.high, size=(self.num_envs, self.action_dim))
        return actions.astype(np.float32)
