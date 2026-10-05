"""What each task's state means: the quantities a probe reads off a latent.

A task turns its oracle state (the `observation` column, see
datasets/prepare_mmbench.py) into probe *targets*, and reads physical
*quantities* back out of targets, true or predicted:

    targets(obs)  (..., obs_dim) -> (..., K)   what the linear probe regresses
    read(targets) (..., K)       -> (..., Q)   radians for angles, metres for positions

Angles are regressed as cos/sin, never raw: a raw angle jumps at +-180 degrees
(and reacher's shoulder winds past it). Errors are reported in degrees for
angles and centimetres for positions (`errors`).

Observation layouts (DMControl, checked against the data):

    acrobot   [cos upper, cos lower, sin upper, sin lower, 2 joint velocities]
    cartpole  [cart x, cos pole, sin pole, cart velocity, pole angular velocity]
    pendulum  [cos pole, sin pole, angular velocity]
    reacher   [shoulder angle, wrist angle, target - finger (x, y), 2 joint velocities]
    cube      28-d: joint pos (6), joint vel (6), effector xyz (12:15), effector yaw
              cos/sin, gripper opening, gripper contact, cube xyz (19:22), cube quat,
              cube yaw cos/sin; positions stored as 10 x metres (x offset by 0.425)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def wrap(a: torch.Tensor) -> torch.Tensor:
    """Angle into (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def cos_sin(a: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(a), torch.sin(a)], -1)


@dataclass(frozen=True)
class Quantity:
    name: str
    # "angle" (radians, error in degrees), "position" (metres, error in cm),
    # or "pixel" (PushT workspace pixels, error in the same pixels)
    kind: str

    @property
    def unit(self) -> str:
        return {"angle": "deg", "position": "cm", "pixel": "px"}[self.kind]


class Task:
    name: str
    quantities: tuple[Quantity, ...]
    state_key = "observation"  # the recording's column holding the true state

    def targets(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def read(self, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def errors(self, pred_targets: torch.Tensor, true_targets: torch.Tensor) -> torch.Tensor:
        """Absolute error per quantity, in its display unit: (..., K), (..., K) -> (..., Q)."""
        diff = self.read(pred_targets) - self.read(true_targets)
        out = []
        for i, q in enumerate(self.quantities):
            if q.kind == "angle":
                out.append(wrap(diff[..., i]).abs() * (180.0 / math.pi))
            elif q.kind == "position":
                out.append(diff[..., i].abs() * 100.0)
            else:
                out.append(diff[..., i].abs())
        return torch.stack(out, -1)


class Acrobot(Task):
    name = "acrobot-swingup"
    # shoulder: angle of the upper link; elbow: angle between the two links
    quantities = (Quantity("shoulder", "angle"), Quantity("elbow", "angle"))

    def targets(self, obs):
        return obs[..., [0, 2, 1, 3]]  # cos/sin upper, cos/sin lower

    def read(self, t):
        upper = torch.atan2(t[..., 1], t[..., 0])
        lower = torch.atan2(t[..., 3], t[..., 2])
        return torch.stack([upper, wrap(lower - upper)], -1)


class Cartpole(Task):
    name = "cartpole-swingup"
    quantities = (Quantity("cart", "position"), Quantity("pole", "angle"))

    def targets(self, obs):
        return obs[..., [0, 1, 2]]  # cart x, cos/sin pole

    def read(self, t):
        return torch.stack([t[..., 0], torch.atan2(t[..., 2], t[..., 1])], -1)


class Pendulum(Task):
    name = "pendulum-swingup"
    quantities = (Quantity("pole", "angle"),)

    def targets(self, obs):
        return obs[..., [0, 1]]  # cos/sin pole

    def read(self, t):
        return torch.atan2(t[..., 1], t[..., 0])[..., None]


class Reacher(Task):
    name = "reacher-easy"
    # the target sits at a random spot each episode; to_target says where, from the finger
    quantities = (Quantity("shoulder", "angle"), Quantity("wrist", "angle"),
                  Quantity("to_target_x", "position"), Quantity("to_target_y", "position"))

    def targets(self, obs):
        return torch.cat([cos_sin(obs[..., 0]), cos_sin(obs[..., 1]), obs[..., 2:4]], -1)

    def read(self, t):
        shoulder = torch.atan2(t[..., 1], t[..., 0])
        wrist = torch.atan2(t[..., 3], t[..., 2])
        return torch.stack([shoulder, wrist, t[..., 4], t[..., 5]], -1)


class Cube(Task):
    """OGBench cube-single: where the cube and the gripper are, as 3-D distances.

    `read` returns both positions in metres, (..., 6) = cube xyz, effector xyz,
    and `errors` scores each as the Euclidean distance between prediction and
    truth, in cm.
    """

    name = "cube-single"
    quantities = (Quantity("cube", "position"), Quantity("effector", "position"))
    scale = 10.0  # observation positions are 10 x metres

    def targets(self, obs):
        return obs[..., [19, 20, 21, 12, 13, 14]]

    def read(self, t):
        return t / self.scale

    def errors(self, pred_targets, true_targets):
        diff = (self.read(pred_targets) - self.read(true_targets)).unflatten(-1, (2, 3))
        return diff.norm(dim=-1) * 100.0


class PushT(Task):
    """PushT (LeWM's recording): where the agent and the T-block are, and how the T is turned.

    State: agent x, y; block x, y; block angle (0..2pi, no symmetry); agent
    velocity x, y; positions in pixels of the 512 x 512 workspace. `read`
    returns (..., 5) = agent xy, block xy, block angle; `errors` scores the two
    positions as 2-D distances (px) and the angle in degrees.
    """

    name = "pusht"
    quantities = (Quantity("agent", "pixel"), Quantity("block", "pixel"), Quantity("block_angle", "angle"))
    state_key = "state"

    def targets(self, obs):
        return torch.cat([obs[..., 0:4], cos_sin(obs[..., 4])], -1)  # agent xy, block xy, cos/sin angle

    def read(self, t):
        return torch.cat([t[..., 0:4], torch.atan2(t[..., 5], t[..., 4])[..., None]], -1)

    def errors(self, pred_targets, true_targets):
        p, t = self.read(pred_targets), self.read(true_targets)
        agent = (p[..., 0:2] - t[..., 0:2]).norm(dim=-1)
        block = (p[..., 2:4] - t[..., 2:4]).norm(dim=-1)
        angle = wrap(p[..., 4] - t[..., 4]).abs() * (180.0 / math.pi)
        return torch.stack([agent, block, angle], -1)


TASKS: dict[str, Task] = {t.name: t for t in (Acrobot(), Cartpole(), Pendulum(), Reacher(), Cube(), PushT())}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"no probe spec for task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]
