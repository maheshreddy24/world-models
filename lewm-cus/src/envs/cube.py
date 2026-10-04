"""OGBench cube manipulation environments, stepped in lockstep.

A thin, readable alternative to a full vectorised-env stack: `num_envs` MuJoCo
environments held in a list and driven together.  Rendering dominates the cost
here, and MuJoCo's EGL contexts are not thread-safe, so the loop is sequential
on purpose.

Evaluation episodes are seeded *from the dataset*: each env is reset to a
recorded `(qpos, qvel)` and its cube target is moved to the pose the recorded
trajectory reaches `goal_offset` steps later.  The agent gets the goal *image*
from the same recording, so the task is "reproduce this outcome from pixels"
and success is the simulator's own 4cm distance check.
"""

from __future__ import annotations

import os

# MuJoCo needs a GL backend chosen before it is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401  — registers the `swm/...` env ids

from ..data.ogbench import CONTACT, cube_proprio
from ._render import close_env


class CubeVecEnv:
    """Args:
        num_envs: environments driven in lockstep.
        env_id: gym id, `swm/OGBCube-v0` for the cube task.
        env_type: `single` (1 cube) through `octuple` (8).
        img_size: render resolution, matched to the model's input.
        max_episode_steps: truncation cap per env.
        terminate_at_goal: stop an env the moment the simulator calls it solved.
        seed: base seed; env `i` gets `seed + i`.
        proprio: which `cube_proprio` the observation dict carries (`policy.proprio`).
    """

    def __init__(
        self,
        num_envs: int,
        env_id: str = "swm/OGBCube-v0",
        env_type: str = "single",
        img_size: int = 224,
        max_episode_steps: int = 200,
        terminate_at_goal: bool = True,
        seed: int = 0,
        proprio: str = "arm",
    ):
        self.num_envs = num_envs
        self.img_size = img_size
        self.seed = seed
        self.proprio = proprio
        self.envs = [
            gym.make(
                env_id,
                env_type=env_type,
                ob_type="states",
                width=img_size,
                height=img_size,
                terminate_at_goal=terminate_at_goal,
                max_episode_steps=max_episode_steps,
                render_mode="rgb_array",
            )
            for _ in range(num_envs)
        ]
        self.action_space = self.envs[0].action_space
        self.action_dim = int(np.prod(self.action_space.shape))
        self.bounds = (self.action_space.low, self.action_space.high)
        self.terminated = np.zeros(num_envs, dtype=bool)
        self.truncated = np.zeros(num_envs, dtype=bool)
        self.success = np.zeros(num_envs, dtype=bool)

    # ------------------------------------------------------------------ #
    #  Episode setup
    # ------------------------------------------------------------------ #
    def reset_to(self, qpos: np.ndarray, qvel: np.ndarray, prev_observation: np.ndarray | None = None) -> dict:
        """Reset every env to a recorded simulator state.

        Args:
            qpos: (num_envs, nq) recorded positions.
            qvel: (num_envs, nv) recorded velocities.
            prev_observation: (num_envs, 28) recorded state one step earlier,
                so a first "ee" `proprio` carries the true end-effector velocity.
                Without it that velocity starts at zero. "arm" ignores it.

        Returns:
            dict with `pixels` (N, H, W, 3) uint8, `observation` (N, 28),
            `proprio` (N, 13 | 6) of kind `self.proprio` and `contact` (N, 1).
        """
        observations = []
        for i, env in enumerate(self.envs):
            state = np.concatenate([np.asarray(qpos[i], np.float64), np.asarray(qvel[i], np.float64)])
            obs, _ = env.reset(seed=self.seed + i, options={"state": state})
            observations.append(np.asarray(obs, np.float32))

        self.terminated[:] = False
        self.truncated[:] = False
        self.success[:] = False
        self._last_obs = np.stack(observations)
        prev = self._last_obs if prev_observation is None else np.asarray(prev_observation, np.float32)
        return {
            "pixels": self.render(),
            "observation": self._last_obs,
            **self._policy_inputs(prev),
            **self._bodies(),
        }

    def set_targets(self, target_pos: np.ndarray, target_quat: np.ndarray | None = None, cube_id: int = 0) -> None:
        """Move each env's goal marker, which is also what success is measured against."""
        for i, env in enumerate(self.envs):
            quat = None if target_quat is None else np.asarray(target_quat[i], np.float64)
            env.unwrapped.set_target_pos(cube_id, np.asarray(target_pos[i], np.float64), quat)

    # ------------------------------------------------------------------ #
    #  Stepping
    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray) -> dict:
        """Advance every still-running env by one raw step.

        Envs that already finished are frozen: their last frame is repeated so
        arrays stay rectangular and videos stay in sync.
        """
        actions = np.asarray(actions, np.float32).reshape(self.num_envs, self.action_dim)
        prev = self._last_obs
        observations = []
        for i, env in enumerate(self.envs):
            if self.done[i]:
                observations.append(self._last_obs[i])
                continue
            obs, _, term, trunc, info = env.step(np.clip(actions[i], *self.bounds))
            observations.append(np.asarray(obs, np.float32))
            self.terminated[i] = bool(term)
            self.truncated[i] = bool(trunc)
            # `success` latches: reaching the goal once counts, even if the cube
            # is nudged out of tolerance afterwards.
            self.success[i] |= bool(info.get("success", False))

        self._last_obs = np.stack(observations)
        return {
            "pixels": self.render(),
            "observation": self._last_obs,
            **self._policy_inputs(prev),
            **self._bodies(),
            "success": self.success.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
        }

    def _bodies(self) -> dict:
        """True cube and end-effector positions, metres — for grasp diagnostics.

        Read straight off the simulator rather than the 28-d observation, whose
        copies are centred and scaled.  A frozen env keeps reporting its last
        pose, like every other field here.
        """
        cube, effector = [], []
        for env in self.envs:
            base = env.unwrapped
            cube.append(base._data.joint("object_joint_0").qpos[:3].copy())
            effector.append(base._data.site_xpos[base._pinch_site_id].copy())
        return {"cube_pos": np.stack(cube).astype(np.float32), "effector_pos": np.stack(effector).astype(np.float32)}

    def _policy_inputs(self, prev: np.ndarray) -> dict:
        """The diffusion policy's side inputs, defined exactly as in its training data."""
        return {
            "proprio": cube_proprio(self._last_obs, prev, self.proprio),
            "contact": self._last_obs[:, CONTACT].copy(),
        }

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated

    def render(self) -> np.ndarray:
        """(num_envs, H, W, 3) uint8 frames from the front camera."""
        frames = [np.asarray(env.unwrapped.render(), np.uint8) for env in self.envs]
        self._frames = np.stack(frames)
        return self._frames

    def close(self) -> None:
        for env in self.envs:
            close_env(env)

    def __enter__(self) -> "CubeVecEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
