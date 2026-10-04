"""OGBench cube-double environments, stepped in lockstep.

The env behind `visual-cube-double-play`, built with the same `mode` and render
size the recording was made with, so a recorded `(qpos, qvel)` renders back to
the recorded frame (mean gap ~0.1/255).

Tasks are seeded from the recording: each env is reset to a recorded state and
must reach the cube positions the recording reaches some steps later.  Success
is OGBench's own criterion — every cube within `cube_tol` of its goal position —
computed here from `qpos`, since the env's built-in check only scores the one
target block it draws in data-collection mode.
"""

from __future__ import annotations

import os

# MuJoCo needs a GL backend chosen before it is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np

import ogbench  # noqa: F401  — registers the OGBench env ids

from ._render import close_env

CUBE_DOUBLE_ENV_ID = "visual-cube-double-v0"


def cube_positions(qpos: np.ndarray, qpos_adr: list[int]) -> np.ndarray:
    """(..., nq) -> (..., num_cubes, 3): each cube's xyz from the free-joint qpos."""
    return np.stack([qpos[..., a : a + 3] for a in qpos_adr], axis=-2)


class CubeDoubleVecEnv:
    """Args:
        num_envs: environments driven in lockstep.
        env_id: gym id; `visual-cube-double-v0` is the env behind the recording.
        img_size: render resolution; must match the recording (124 here).
        max_episode_steps: truncation cap per env; keep it above the eval budget.
        cube_tol: metres each cube may be from its goal (OGBench uses 0.04).
        seed: base seed; env `i` gets `seed + i`.
    """

    def __init__(
        self,
        num_envs: int,
        env_id: str = CUBE_DOUBLE_ENV_ID,
        img_size: int = 124,
        max_episode_steps: int = 1000,
        cube_tol: float = 0.04,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.seed = seed
        self.cube_tol = cube_tol
        self.envs = [
            gym.make(
                env_id,
                width=img_size,
                height=img_size,
                mode="data_collection",  # as recorded: no task, no goal pre-roll on reset
                terminate_at_goal=False,
                max_episode_steps=max_episode_steps,
            )
            for _ in range(num_envs)
        ]
        self.envs[0].reset(seed=seed)
        model = self.envs[0].unwrapped._model
        self.num_cubes = self.envs[0].unwrapped._num_cubes
        self.qpos_adr = [int(model.jnt_qposadr[model.joint(f"object_joint_{i}").id]) for i in range(self.num_cubes)]

        self.action_space = self.envs[0].action_space
        self.action_dim = int(np.prod(self.action_space.shape))
        self.bounds = (self.action_space.low, self.action_space.high)
        self.terminated = np.zeros(num_envs, dtype=bool)
        self.truncated = np.zeros(num_envs, dtype=bool)
        self.success = np.zeros(num_envs, dtype=bool)
        self.goal_cubes: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Episode setup
    # ------------------------------------------------------------------ #
    def reset_to(self, qpos: np.ndarray, qvel: np.ndarray) -> dict:
        """Reset every env to a recorded state: qpos (num_envs, 28), qvel (num_envs, 26)."""
        frames = []
        for i, env in enumerate(self.envs):
            env.reset(seed=self.seed + i)
            u = env.unwrapped
            u.set_state(np.asarray(qpos[i], np.float64), np.asarray(qvel[i], np.float64))
            frames.append(np.asarray(u.compute_observation(), np.uint8))

        self.terminated[:] = False
        self.truncated[:] = False
        self.success[:] = False
        self._frames = np.stack(frames)
        self._qpos = np.stack([env.unwrapped._data.qpos.copy() for env in self.envs])
        return self._observation()

    def set_goals(self, goal_qpos: np.ndarray) -> None:
        """The cube positions success is measured against, from each goal's qpos."""
        self.goal_cubes = cube_positions(np.asarray(goal_qpos, np.float64), self.qpos_adr)

    def freeze(self, mask: np.ndarray) -> None:
        """Park spare envs of a short batch: never stepped, never counted."""
        self.terminated[np.asarray(mask, bool)] = True

    # ------------------------------------------------------------------ #
    #  Stepping
    # ------------------------------------------------------------------ #
    def step(self, actions: np.ndarray) -> dict:
        """Advance every still-running env by one raw step; finished envs hold their last frame."""
        actions = np.asarray(actions, np.float32).reshape(self.num_envs, self.action_dim)
        stepped = ~self.done
        for i, env in enumerate(self.envs):
            if not stepped[i]:
                continue
            obs, _, term, trunc, _ = env.step(np.clip(actions[i], *self.bounds))
            self._frames[i] = np.asarray(obs, np.uint8)
            self._qpos[i] = env.unwrapped._data.qpos
            self.terminated[i] = bool(term)
            self.truncated[i] = bool(trunc)

        if self.goal_cubes is not None:
            # Latched: reaching the goal once counts. Frozen envs are not re-judged.
            self.success |= self.reached() & stepped
        return self._observation()

    def cube_distances(self) -> np.ndarray:
        """(num_envs, num_cubes) metres from each cube to its goal."""
        return np.linalg.norm(cube_positions(self._qpos, self.qpos_adr) - self.goal_cubes, axis=-1)

    def reached(self) -> np.ndarray:
        return (self.cube_distances() <= self.cube_tol).all(axis=1)

    def _observation(self) -> dict:
        return {
            "pixels": self._frames.copy(),
            "qpos": self._qpos.astype(np.float32),
            "success": self.success.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
        }

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated | self.success

    def close(self) -> None:
        for env in self.envs:
            close_env(env)

    def __enter__(self) -> "CubeDoubleVecEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
