"""BallCatch: a 2D side-view catching task in pymunk.

A ball is launched from the left wall with a random height, speed and angle;
a U-shaped basket slides along the floor and the agent sets its horizontal
velocity. The episode is a success when the ball's first landing is
on the basket's floor rather than on the ground.

    world      4 m x 4 m box: floor, left and right walls, open top
    view       the whole box, fixed camera, 128 x 128 RGB (~30 px per metre)
    control    20 Hz, 12 physics substeps per step (240 Hz)
    action     (1,) basket x-velocity in [-1, 1], times MAX_SPEED m/s
    oracle     (6,) ball x, y, vx, vy, basket x, basket vx

Everything the simulator needs to resume is in `SimState` (12 floats), so a
recorded row can be replayed exactly: at a launch frame nothing is in contact,
and the basket is kinematic, so no contact cache is lost on reset.

`BallCatchSim` is one simulator; `BallVecEnv` drives several in lockstep with
the same interface the evaluation loop uses for the cube (`reset_to`, `step`,
`success`, `done`, `bounds`).

Nothing in the frame is fixed per episode except through the ball and basket
themselves. v1 drew a launch tube at the episode's launch height and angle for
the whole episode, and LeWM encoded that instead of the ball: it differs across
episodes (SIGReg is happy) and never changes in time (prediction is free).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pymunk

# -- world ------------------------------------------------------------------ #
WORLD_W = 4.0  # metres between the walls
WORLD_H = 4.0  # nominal box height; the top is open
GRAVITY = 9.81
CONTROL_HZ = 20
SUBSTEPS = 12
DT = 1.0 / (CONTROL_HZ * SUBSTEPS)

# -- bodies ----------------------------------------------------------------- #
BALL_R = 0.2  # ~12 px across at 128 px, close to one 16 px ViT patch
BALL_MASS = 0.2
BASKET_HALF = 0.4  # wall centre-lines at +-0.4 m: a 0.66 m opening for a 0.4 m ball (v1 slack, 0.26 m)
BASKET_WALL_H = 0.4
SEG_R = 0.07  # basket segments are 0.14 m thick
BASKET_X_RANGE = (0.6, WORLD_W - BASKET_HALF - SEG_R - 0.01)
MAX_SPEED = 3.0  # m/s at |action| = 1

# -- launcher --------------------------------------------------------------- #
LAUNCH_X = 0.3  # ball centre at the launch frame
# Low enough to fly over the basket's left wall when the basket starts at the wall.
LAUNCH_H = (0.8, 1.8)
LAUNCH_SPEED = (4.0, 8.0)
LAUNCH_ANGLE_DEG = (20.0, 70.0)
LAND_X = (1.0, 3.5)  # where the analytic landing point may fall
MIN_FLIGHT_S = 0.7
MAX_APEX = 3.7  # ball centre; keeps the whole ball inside the view

# Height of the ball centre when it rests on the basket floor, the plane the
# expert aims at.
CATCH_Y = 2 * SEG_R + BALL_R

# -- rendering -------------------------------------------------------------- #
IMG_SIZE = 128
SUPERSAMPLE = 4
VIEW_X = (-0.1, WORLD_W + 0.1)  # walls are drawn 0.1 m thick outside the box
VIEW_Y = (-0.2, WORLD_H)  # 0.2 m of floor below y = 0
COLORS = {
    "bg": (236, 236, 230),
    "ground": (90, 90, 96),
    "wall": (120, 120, 126),
    "ball": (245, 130, 20),
    "basket": (35, 85, 200),
}

# -- simulator state -------------------------------------------------------- #
# ball x, y, vx, vy, angle, angular velocity, basket x, basket vx,
# launch height, launch angle (rad; bookkeeping only, not drawn), landed, caught
STATE_DIM = 12
OBS_DIM = 6
ACTION_DIM = 1

# collision types
_BALL, _GROUND, _BASKET_FLOOR, _OTHER = 1, 2, 3, 4


def sample_launch(rng: np.random.Generator) -> tuple[float, float, float]:
    """(height, speed, angle in rad) whose ballistic arc stays in view and lands on the floor range."""
    while True:
        h = rng.uniform(*LAUNCH_H)
        speed = rng.uniform(*LAUNCH_SPEED)
        angle = math.radians(rng.uniform(*LAUNCH_ANGLE_DEG))
        vx, vy = speed * math.cos(angle), speed * math.sin(angle)
        if h + vy * vy / (2 * GRAVITY) > MAX_APEX:
            continue
        t = time_to_height(h, vy, CATCH_Y)
        if t is None or t < MIN_FLIGHT_S:
            continue
        if LAND_X[0] <= LAUNCH_X + vx * t <= LAND_X[1]:
            return h, speed, angle


def time_to_height(y: float, vy: float, target: float) -> float | None:
    """Time until a ballistic body at height `y` rising at `vy` descends through `target`."""
    disc = vy * vy + 2 * GRAVITY * (y - target)
    if disc < 0:
        return None
    return (vy + math.sqrt(disc)) / GRAVITY


def predicted_landing_x(obs: np.ndarray) -> float:
    """Where the ball crosses the catch plane on its way down, ignoring walls.

    Once the ball is below the plane (it has landed), its current x is returned.
    """
    x, y, vx, vy = (float(v) for v in obs[:4])
    t = time_to_height(y, vy, CATCH_Y)
    if t is None or y < CATCH_Y:
        return x
    # Reflect off the walls so the prediction stays inside the box.
    x = x + vx * t
    lo, hi = BALL_R, WORLD_W - BALL_R
    span = hi - lo
    x = (x - lo) % (2 * span)
    return lo + (x if x <= span else 2 * span - x)


class BallCatchSim:
    """One BallCatch simulator.

    Args:
        img_size: render resolution.
    """

    def __init__(self, img_size: int = IMG_SIZE):
        self.img_size = img_size
        self._build()
        self.landed = False
        self.caught = False
        self.launch_h = 1.0
        self.launch_angle = 0.5

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        space = pymunk.Space()
        space.gravity = (0.0, -GRAVITY)
        space.iterations = 20
        static = space.static_body

        ground = pymunk.Segment(static, (-1.0, -0.5), (WORLD_W + 1.0, -0.5), 0.5)  # top at y = 0
        left = pymunk.Segment(static, (-0.5, 0.0), (-0.5, 10.0), 0.5)  # face at x = 0
        right = pymunk.Segment(static, (WORLD_W + 0.5, 0.0), (WORLD_W + 0.5, 10.0), 0.5)
        ground.collision_type = _GROUND
        for s in (left, right):
            s.collision_type = _OTHER
        for s, e in ((ground, 0.8), (left, 0.8), (right, 0.8)):
            s.elasticity, s.friction = e, 0.6
        space.add(ground, left, right)

        basket = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        floor = pymunk.Segment(basket, (-BASKET_HALF, SEG_R), (BASKET_HALF, SEG_R), SEG_R)
        wall_l = pymunk.Segment(basket, (-BASKET_HALF, SEG_R), (-BASKET_HALF, BASKET_WALL_H), SEG_R)
        wall_r = pymunk.Segment(basket, (BASKET_HALF, SEG_R), (BASKET_HALF, BASKET_WALL_H), SEG_R)
        floor.collision_type = _BASKET_FLOOR
        floor.elasticity, floor.friction = 0.3, 0.9  # a soft floor keeps a caught ball in
        for w in (wall_l, wall_r):
            w.collision_type = _OTHER
            w.elasticity, w.friction = 0.6, 0.5
        space.add(basket, floor, wall_l, wall_r)

        ball = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, BALL_R))
        ball_shape = pymunk.Circle(ball, BALL_R)
        ball_shape.collision_type = _BALL
        ball_shape.elasticity, ball_shape.friction = 0.8, 0.5
        space.add(ball, ball_shape)

        self.space, self.ball, self.basket = space, ball, basket
        self._basket_cmd = 0.0

    # ------------------------------------------------------------------ #
    #  State
    # ------------------------------------------------------------------ #
    def reset_launch(self, rng: np.random.Generator, basket_x: float | None = None) -> np.ndarray:
        """A fresh episode: ball at the left wall with a sampled launch velocity."""
        h, speed, angle = sample_launch(rng)
        if basket_x is None:
            basket_x = rng.uniform(*BASKET_X_RANGE)
        state = np.array(
            [LAUNCH_X, h, speed * math.cos(angle), speed * math.sin(angle), 0.0, 0.0,
             basket_x, 0.0, h, angle, 0.0, 0.0],
            dtype=np.float64,
        )
        self.set_state(state)
        return self.observation()

    def set_state(self, state: np.ndarray) -> None:
        """Restore a recorded `SimState`. A fresh space means no stale contacts."""
        s = np.asarray(state, np.float64)
        self._build()
        self.ball.position = (s[0], s[1])
        self.ball.velocity = (s[2], s[3])
        self.ball.angle = s[4]
        self.ball.angular_velocity = s[5]
        self.basket.position = (s[6], 0.0)
        self._basket_cmd = float(s[7])
        self.basket.velocity = (self._basket_cmd, 0.0)
        self.launch_h, self.launch_angle = float(s[8]), float(s[9])
        self.landed, self.caught = bool(s[10]), bool(s[11])
        self.space.reindex_shapes_for_body(self.basket)

    def get_state(self) -> np.ndarray:
        b = self.ball
        return np.array(
            [b.position.x, b.position.y, b.velocity.x, b.velocity.y, b.angle, b.angular_velocity,
             self.basket.position.x, self._basket_cmd, self.launch_h, self.launch_angle,
             float(self.landed), float(self.caught)],
            dtype=np.float64,
        )

    def observation(self) -> np.ndarray:
        """The 6-d oracle state: ball x, y, vx, vy, basket x, basket vx."""
        b = self.ball
        return np.array(
            [b.position.x, b.position.y, b.velocity.x, b.velocity.y, self.basket.position.x, self._basket_cmd],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Dynamics
    # ------------------------------------------------------------------ #
    def step(self, action) -> np.ndarray:
        """Advance one control step (1/20 s) with basket velocity `action * MAX_SPEED`."""
        a = float(np.clip(np.asarray(action, np.float64).reshape(-1)[0], -1.0, 1.0))
        cmd = a * MAX_SPEED
        lo, hi = BASKET_X_RANGE
        for _ in range(SUBSTEPS):
            x = self.basket.position.x
            # A kinematic body ignores the static walls, so clamp it by hand:
            # never command a velocity that would carry it past a bound this substep.
            v = min(max(cmd, (lo - x) / DT), (hi - x) / DT)
            self.basket.velocity = (v, 0.0)
            self.space.step(DT)
            if not self.landed:
                self._check_landing()
        # The commanded velocity after clamping at a bound, as the oracle reports it.
        x = self.basket.position.x
        self._basket_cmd = 0.0 if (x <= lo + 1e-6 and cmd < 0) or (x >= hi - 1e-6 and cmd > 0) else cmd
        return self.observation()

    def _check_landing(self) -> None:
        """Latch the first touch of the basket floor (caught) or the ground (missed)."""
        hits = []
        self.ball.each_arbiter(lambda arb: hits.extend(s.collision_type for s in arb.shapes))
        if _BASKET_FLOOR in hits:
            self.landed = self.caught = True
        elif _GROUND in hits:
            self.landed = True

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #
    def render(self) -> np.ndarray:
        """(img_size, img_size, 3) uint8, anti-aliased by supersampling."""
        n = self.img_size * SUPERSAMPLE
        sx = n / (VIEW_X[1] - VIEW_X[0])
        sy = n / (VIEW_Y[1] - VIEW_Y[0])

        def px(x, y):
            return int(round((x - VIEW_X[0]) * sx)), int(round((VIEW_Y[1] - y) * sy))

        img = np.empty((n, n, 3), np.uint8)
        img[:] = COLORS["bg"]
        cv2.rectangle(img, px(VIEW_X[0], 0.0), px(VIEW_X[1], VIEW_Y[0]), COLORS["ground"], -1)
        cv2.rectangle(img, px(VIEW_X[0], VIEW_Y[1]), px(0.0, 0.0), COLORS["wall"], -1)
        cv2.rectangle(img, px(WORLD_W, VIEW_Y[1]), px(VIEW_X[1], 0.0), COLORS["wall"], -1)

        # Basket: a U of thick segments, exactly the collision geometry.
        bx = self.basket.position.x
        thick = max(1, int(round(2 * SEG_R * sx)))
        pts = [(bx - BASKET_HALF, BASKET_WALL_H), (bx - BASKET_HALF, SEG_R),
               (bx + BASKET_HALF, SEG_R), (bx + BASKET_HALF, BASKET_WALL_H)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, px(*a), px(*b), COLORS["basket"], thick, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, px(*p), thick // 2, COLORS["basket"], -1, cv2.LINE_AA)

        ball = self.ball.position
        cv2.circle(img, px(ball.x, ball.y), int(round(BALL_R * sx)), COLORS["ball"], -1, cv2.LINE_AA)

        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
#  Behaviour policies for data collection
# --------------------------------------------------------------------------- #
class OUPolicy:
    """Smoothed random basket velocities (Ornstein-Uhlenbeck around zero)."""

    def __init__(self, rng: np.random.Generator, sigma: float | None = None, theta: float = 0.15):
        self.rng = rng
        self.sigma = rng.uniform(0.15, 0.5) if sigma is None else sigma
        self.theta = theta
        self.a = rng.uniform(-1.0, 1.0)

    def __call__(self, obs: np.ndarray, sim: BallCatchSim) -> float:
        self.a += -self.theta * self.a + self.sigma * self.rng.standard_normal()
        # Occasional jumps so the data holds sharp velocity changes too.
        if self.rng.random() < 0.05:
            self.a = self.rng.uniform(-1.0, 1.0)
        self.a = float(np.clip(self.a, -1.0, 1.0))
        return self.a


class NoisyExpert:
    """Chase the predicted landing point with per-episode gain, delay, aim error and noise.

    After the ball lands the basket carries it gently (slow OU) if it caught
    it, and wanders (OU) if it missed, so the tail of an episode still moves.
    The spread of parameters puts the catch rate somewhat above half.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.gain = rng.uniform(1.0, 5.0)  # 1/s
        self.delay = int(rng.integers(0, 12))  # control steps before reacting
        self.aim = rng.normal(0.0, 0.25)  # metres of aiming error
        self.noise = rng.uniform(0.0, 0.3)
        self.wander = OUPolicy(rng)
        self.carry = OUPolicy(rng, sigma=rng.uniform(0.05, 0.2))
        self.carry.a = 0.0
        self.t = 0

    def __call__(self, obs: np.ndarray, sim: BallCatchSim) -> float:
        self.t += 1
        if sim.landed:
            return (self.carry if sim.caught else self.wander)(obs, sim)
        if self.t <= self.delay:
            return 0.0
        target = predicted_landing_x(obs) + self.aim
        a = self.gain * (target - float(obs[4])) / MAX_SPEED + self.noise * self.rng.standard_normal()
        return float(np.clip(a, -1.0, 1.0))


def collect_episode(seed: int, steps: int, policy: str, img_size: int = IMG_SIZE) -> dict:
    """Roll out one episode and return its per-row arrays (steps + 1 rows).

    The last row's action is NaN: no action follows the final frame.
    """
    rng = np.random.default_rng(seed)
    sim = BallCatchSim(img_size)
    obs = sim.reset_launch(rng)
    actor = NoisyExpert(rng) if policy == "expert" else OUPolicy(rng)

    pixels = np.empty((steps + 1, img_size, img_size, 3), np.uint8)
    observation = np.empty((steps + 1, OBS_DIM), np.float32)
    state = np.empty((steps + 1, STATE_DIM), np.float64)
    action = np.full((steps + 1, ACTION_DIM), np.nan, np.float32)
    for t in range(steps + 1):
        pixels[t] = sim.render()
        observation[t] = obs
        state[t] = sim.get_state()
        if t == steps:
            break
        action[t, 0] = actor(obs, sim)
        obs = sim.step(action[t])

    landed, caught = state[:, 10] > 0.5, state[:, 11] > 0.5
    first = lambda m: int(np.argmax(m)) if m.any() else -1  # noqa: E731
    launch = state[0]
    return {
        "pixels": pixels,
        "observation": observation,
        "sim_state": state,
        "action": action,
        "landed": landed,
        "caught": caught,
        "success": bool(caught[-1]),
        "catch_step": first(caught),
        "land_step": first(landed),
        "launch": np.array([launch[8], math.hypot(launch[2], launch[3]), launch[9]], np.float32),
    }


# --------------------------------------------------------------------------- #
#  Lockstep vector env for evaluation
# --------------------------------------------------------------------------- #
class BallVecEnv:
    """`num_envs` BallCatch simulators driven together.

    Args:
        num_envs: simulators.
        img_size: render resolution, matched to the model's input.
        max_episode_steps: truncation cap per env.
        terminate_at_goal: stop an env once the ball has landed in the basket.
    """

    def __init__(self, num_envs: int, img_size: int = IMG_SIZE, max_episode_steps: int = 100,
                 terminate_at_goal: bool = True):
        self.num_envs = num_envs
        self.sims = [BallCatchSim(img_size) for _ in range(num_envs)]
        self.max_episode_steps = max_episode_steps
        self.terminate_at_goal = terminate_at_goal
        self.action_dim = ACTION_DIM
        self.bounds = (np.full(ACTION_DIM, -1.0, np.float32), np.full(ACTION_DIM, 1.0, np.float32))
        self.t = 0
        self.success = np.zeros(num_envs, dtype=bool)
        self.landed = np.zeros(num_envs, dtype=bool)
        self.terminated = np.zeros(num_envs, dtype=bool)
        self.truncated = np.zeros(num_envs, dtype=bool)

    def reset_to(self, states: np.ndarray) -> dict:
        """Reset every env to a recorded `SimState` row."""
        for sim, s in zip(self.sims, states):
            sim.set_state(s)
        self.t = 0
        self.terminated[:] = self.truncated[:] = False
        self._read_flags()
        return self._obs()

    def step(self, actions: np.ndarray) -> dict:
        """Advance every running env one control step; finished envs hold their last frame."""
        actions = np.asarray(actions, np.float32).reshape(self.num_envs, ACTION_DIM)
        for i, sim in enumerate(self.sims):
            if not self.done[i]:
                sim.step(np.clip(actions[i], *self.bounds))
        self.t += 1
        self._read_flags()
        running = ~self.done
        if self.terminate_at_goal:
            self.terminated |= running & self.success
        self.truncated |= running & (self.t >= self.max_episode_steps)
        return {**self._obs(), "success": self.success.copy(), "landed": self.landed.copy()}

    def _read_flags(self) -> None:
        self.success = np.array([sim.caught for sim in self.sims])
        self.landed = np.array([sim.landed for sim in self.sims])

    def _obs(self) -> dict:
        return {
            "pixels": np.stack([sim.render() for sim in self.sims]),
            "observation": np.stack([sim.observation() for sim in self.sims]),
        }

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated

    def close(self) -> None:
        pass
