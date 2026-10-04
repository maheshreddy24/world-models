"""OGBench scene environments, stepped in lockstep for the v1 evaluation.

The scene counterpart of `CubeVecEnv`: `num_envs` MuJoCo environments in a list,
driven together.  Rendering dominates the cost and MuJoCo's EGL contexts are not
thread-safe, so the loop is sequential on purpose.

Two things differ from the cube setup:

* an episode is seeded from a `v1_pairs.npz` row, which carries the full start
  state — `qpos`, `qvel` *and* `button_states`, because the button colours are
  env state that lives outside `qpos`;
* success is not the env's own check.  The v1 tasks are "make the scene look
  like this", so `src.data.scene_tasks.is_success` compares the final cube pose,
  drawer, window and buttons against the goal state and ignores the arm.  The
  env's built-in task is never set, and `terminate_at_goal` is therefore off.
"""

from __future__ import annotations

import os

# MuJoCo needs a GL backend chosen before it is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import ogbench  # noqa: F401  — registers the OGBench env ids

from ..data.scene_state import STATE, SceneState
from ..data.scene_tasks import is_success
from ._render import close_env

SCENE_ENV_ID = "visual-scene-v0"


class SceneVecEnv:
    """Args:
        num_envs: environments driven in lockstep.
        env_id: gym id; `visual-scene-v0` is the env behind `visual-scene-play-v0`.
        max_episode_steps: truncation cap per env; keep it above the eval budget
            so the budget is what ends an episode.
        cube_tol, slide_tol: success tolerances, see `scene_tasks.is_success`.
        with_state: also report `observation`, the 39-d oracle state, built by the
            same `SceneState` function the training column was built with.
        seed: base seed; env `i` gets `seed + i`.

    Every observation dict holds `pixels` (N, 64, 64, 3) uint8 — the same frames
    the recording stores — plus the simulator state and `proprio`, the gripper
    position and its per-step displacement.
    """

    def __init__(
        self,
        num_envs: int,
        env_id: str = SCENE_ENV_ID,
        max_episode_steps: int = 1000,
        cube_tol: float = 0.04,
        slide_tol: float = 0.03,
        with_state: bool = False,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self._state_fn = SceneState() if with_state else None
        self.seed = seed
        self.cube_tol, self.slide_tol = cube_tol, slide_tol
        self.envs = [
            gym.make(env_id, max_episode_steps=max_episode_steps, render_mode="rgb_array")
            for _ in range(num_envs)
        ]
        self.action_space = self.envs[0].action_space
        self.action_dim = int(np.prod(self.action_space.shape))
        self.bounds = (self.action_space.low, self.action_space.high)

        self.terminated = np.zeros(num_envs, dtype=bool)
        self.truncated = np.zeros(num_envs, dtype=bool)
        self.success = np.zeros(num_envs, dtype=bool)
        self.goal_qpos: np.ndarray | None = None
        self.goal_btn: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Episode setup
    # ------------------------------------------------------------------ #
    def reset_to(self, qpos: np.ndarray, qvel: np.ndarray, button_states: np.ndarray) -> dict:
        """Reset every env to a recorded start state from `v1_pairs.npz`.

        Args:
            qpos: (num_envs, 25), qvel: (num_envs, 24), button_states: (num_envs, 2).

        Returns:
            the observation dict; `proprio` starts at zero velocity, exactly as
            the first row of a recorded episode does.
        """
        frames = []
        for i, env in enumerate(self.envs):
            env.reset(seed=self.seed + i)
            u = env.unwrapped
            u.set_state(
                np.asarray(qpos[i], np.float64),
                np.asarray(qvel[i], np.float64),
                np.asarray(button_states[i]).astype(int),
            )
            frames.append(np.asarray(u.compute_observation(), np.uint8))

        self.terminated[:] = False
        self.truncated[:] = False
        self.success[:] = False
        self._frames = np.stack(frames)
        self._info = [env.unwrapped.compute_ob_info() for env in self.envs]
        self._prev_effector = np.stack([i["proprio/effector_pos"] for i in self._info]).astype(np.float32)
        return self._observation()

    def freeze(self, mask: np.ndarray) -> None:
        """Park envs so they are never stepped and never counted as solved.

        Used when a batch of tasks is smaller than the env pool: the spare envs
        are reset to a copy of a real task (so every array stays rectangular)
        and then frozen here, which is cheaper than building a smaller pool.
        """
        self.terminated[np.asarray(mask, bool)] = True

    def set_goals(self, goal_qpos: np.ndarray, goal_btn: np.ndarray) -> None:
        """The scene states success is measured against, one per env."""
        self.goal_qpos = np.asarray(goal_qpos, np.float64)
        self.goal_btn = np.asarray(goal_btn).astype(int)

    # ------------------------------------------------------------------ #
    #  Stepping
    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray) -> dict:
        """Advance every still-running env by one raw step.

        Envs that already finished are frozen: their last frame is repeated so
        arrays stay rectangular and videos stay in sync.
        """
        actions = np.asarray(actions, np.float32).reshape(self.num_envs, self.action_dim)
        prev_effector = np.stack([i["proprio/effector_pos"] for i in self._info]).astype(np.float32)
        stepped = ~self.done  # which envs this call actually advances

        for i, env in enumerate(self.envs):
            if not stepped[i]:
                continue
            obs, _, term, trunc, _ = env.step(np.clip(actions[i], *self.bounds))
            self._frames[i] = np.asarray(obs, np.uint8)
            self._info[i] = env.unwrapped.compute_ob_info()
            # The env's own task is never set, so `term` only fires on a real
            # simulator failure; truncation is the TimeLimit wrapper.
            self.terminated[i] = bool(term)
            self.truncated[i] = bool(trunc)

        self._prev_effector = prev_effector
        if self.goal_qpos is not None:
            reached = np.array([
                is_success(i["qpos"], i["button_states"], self.goal_qpos[k], self.goal_btn[k],
                           self.cube_tol, self.slide_tol)
                for k, i in enumerate(self._info)
            ])
            # `success` latches: reaching the goal once counts, even if the scene
            # drifts out of tolerance afterwards. Frozen envs are not re-judged.
            self.success |= reached & stepped
        return self._observation()

    def _observation(self) -> dict:
        effector = np.stack([i["proprio/effector_pos"] for i in self._info]).astype(np.float32)
        extra = {}
        if self._state_fn is not None:
            extra[STATE] = np.stack(
                [self._state_fn(i["qpos"], i["qvel"], i["button_states"]) for i in self._info]
            )
        return {
            **extra,
            "pixels": self._frames.copy(),
            "qpos": np.stack([i["qpos"] for i in self._info]).astype(np.float32),
            "qvel": np.stack([i["qvel"] for i in self._info]).astype(np.float32),
            "button_states": np.stack([i["button_states"] for i in self._info]).astype(np.int64),
            # Same definition as the dataset's proprio: position, then the
            # displacement over the last env step.
            "proprio": np.concatenate([effector, effector - self._prev_effector], axis=-1),
            "success": self.success.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
        }

    @property
    def done(self) -> np.ndarray:
        """Envs that have stopped: finished, truncated, or already successful."""
        return self.terminated | self.truncated | self.success

    def render(self) -> np.ndarray:
        """(num_envs, 64, 64, 3) uint8 — the frames the policy is looking at."""
        return self._frames.copy()

    def close(self) -> None:
        for env in self.envs:
            close_env(env)
        if self._state_fn is not None:
            self._state_fn.close()

    def __enter__(self) -> "SceneVecEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
