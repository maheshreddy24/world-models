"""The seven v1 (Direct) tasks on OGBench Scene, and how start/goal pairs are mined.

A pair is two frames of the *same* episode: a start the task can begin from and
a goal a few steps later where the task's target object has changed and nothing
else has.  "Nothing else" is checked on the simulator state (`qpos`,
`button_states`), never on pixels, so the pair is exact rather than approximate.

    open_drawer, close_drawer, open_window, close_window,
    move_cube, cube_into_drawer, toggle_lock

The two CLIs on top of this module write the files the training code reads:

    datasets/make_v1_pairs.py -> v1_pairs.npz     (evaluation: capped per task, deduped,
                                            carries the simulator state to reset to)
    datasets/scene_v1.py -> train_pairs.npz  (training: every valid start, indices only)

This module is import-only — nothing here touches a file or an environment
except `reset_to`, which is handed one.
"""

from __future__ import annotations

import numpy as np

# qpos / qvel layout of scene-v0 (verified against the MuJoCo model:
# qpos[0:6] UR5 joints, [6:14] gripper, [14:21] cube pose, [21:23] buttons).
CUBE = slice(14, 17)  # cube x, y, z
CUBE_VEL = slice(14, 17)  # cube linear velocity (qvel)
DRAWER = 23  # drawer_slide: 0.0 closed, -0.16 open
WINDOW = 24  # window_slide: 0.0 closed, 0.2 open

TASKS = ["open_drawer", "close_drawer", "open_window", "close_window",
         "move_cube", "cube_into_drawer", "toggle_lock"]

# Defaults shared by both mining CLIs, so eval pairs and training pairs are the
# same kind of thing.
MIN_GAP, MAX_GAP = 20, 150
SETTLE = 10  # the goal must still hold this many steps later
STRIDE = 5  # check every Nth frame as a start
HORIZON = MAX_GAP + SETTLE  # longest possible start -> goal gap


def compute_features(d) -> dict:
    """Derived scene state per row, from a mapping with `qpos`/`qvel`/`button_states`.

    Accepts the raw npz dict or anything else that indexes by those keys, so the
    same code mines from a recording and from a prepared store.
    """
    q, v, btn = np.asarray(d["qpos"]), np.asarray(d["qvel"]), np.asarray(d["button_states"])
    f = {}
    f["drawer"] = q[:, DRAWER]
    f["window"] = q[:, WINDOW]
    f["drawer_closed"] = f["drawer"] > -0.01
    f["drawer_open"] = f["drawer"] < -0.15
    f["window_closed"] = f["window"] < 0.01
    f["window_open"] = f["window"] > 0.19
    f["cube"] = q[:, CUBE]
    f["cube_still"] = np.linalg.norm(v[:, CUBE_VEL], axis=1) < 0.01
    # In-drawer box, same as SceneEnv._is_in_drawer (drawer site y = -0.24 - drawer_slide).
    site_y = -0.24 - f["drawer"]
    c = f["cube"]
    f["cube_in_drawer"] = ((c[:, 0] >= 0.21) & (c[:, 0] <= 0.45) & (c[:, 1] >= site_y - 0.27)
                           & (c[:, 1] <= site_y - 0.07) & (c[:, 2] >= 0.0) & (c[:, 2] <= 0.15))
    f["cube_on_table"] = f["cube_still"] & (c[:, 2] < 0.025) & ~f["cube_in_drawer"]
    f["btn"] = btn
    return f


def start_mask(f: dict, task: str) -> np.ndarray:
    """Which frames can be a start frame for this task."""
    b = f["btn"]
    table = f["cube_on_table"]
    return {
        "open_drawer": f["drawer_closed"] & (b[:, 0] == 1) & table,
        "close_drawer": f["drawer_open"] & (b[:, 0] == 1) & table,
        "open_window": f["window_closed"] & (b[:, 1] == 1) & table,
        "close_window": f["window_open"] & (b[:, 1] == 1) & table,
        "move_cube": table,
        "cube_into_drawer": f["drawer_open"] & (b[:, 0] == 1) & table,
        "toggle_lock": table,
    }[task]


def goal_mask(f: dict, task: str, t: int, g: np.ndarray) -> np.ndarray:
    """For start `t`, which candidate goal frames `g` (an array) are valid."""
    b = f["btn"]
    same_drawer = np.abs(f["drawer"][g] - f["drawer"][t]) < 0.02
    same_window = np.abs(f["window"][g] - f["window"][t]) < 0.02
    same_btn0 = b[g, 0] == b[t, 0]
    same_btn1 = b[g, 1] == b[t, 1]
    same_cube = (np.linalg.norm(f["cube"][g] - f["cube"][t], axis=1) < 0.02) & f["cube_still"][g]

    if task == "open_drawer":
        return f["drawer_open"][g] & same_window & same_btn0 & same_btn1 & same_cube
    if task == "close_drawer":
        return f["drawer_closed"][g] & same_window & same_btn0 & same_btn1 & same_cube
    if task == "open_window":
        return f["window_open"][g] & same_drawer & same_btn0 & same_btn1 & same_cube
    if task == "close_window":
        return f["window_closed"][g] & same_drawer & same_btn0 & same_btn1 & same_cube
    if task == "move_cube":
        moved = np.linalg.norm(f["cube"][g] - f["cube"][t], axis=1) > 0.05
        return f["cube_on_table"][g] & moved & same_drawer & same_window & same_btn0 & same_btn1
    if task == "cube_into_drawer":
        return (f["cube_in_drawer"][g] & f["cube_still"][g] & f["drawer_open"][g]
                & same_window & same_btn0 & same_btn1)
    if task == "toggle_lock":
        flipped_one = (b[g, 0] != b[t, 0]) ^ (b[g, 1] != b[t, 1])  # exactly one button changed
        return flipped_one & same_drawer & same_window & same_cube
    raise ValueError(task)


def mine(d, f, task, min_gap, max_gap, settle, stride, rng, dedupe=True) -> np.ndarray:
    """All `(start, goal)` row pairs of one task: (n, 2) int64.

    `dedupe=True` keeps roughly one pair per event (evaluation); `False` keeps
    every valid start, which is what the training set wants.
    """
    ends = np.flatnonzero(np.asarray(d["terminals"]))
    starts_ep = np.r_[0, ends[:-1] + 1]
    smask = start_mask(f, task)
    found = {}  # (episode, goal bucket) -> (t, g); dedupes starts leading to one event
    for ep, (s, e) in enumerate(zip(starts_ep, ends)):
        for t in range(s, e + 1, stride):
            if not smask[t]:
                continue
            lo, hi = t + min_gap, min(t + max_gap, e - settle)
            if lo > hi:
                continue
            g = np.arange(lo, hi + 1)
            ok = goal_mask(f, task, t, g) & goal_mask(f, task, t, g + settle)  # still valid after settling
            if ok.any():
                goal = int(g[np.argmax(ok)] + settle)
                key = (ep, goal // 25) if dedupe else (ep, t)
                if key not in found or rng.random() < 0.5:
                    found[key] = (t, goal)
    return np.array(list(found.values()), dtype=np.int64).reshape(-1, 2)


def is_success(final_qpos, final_btn, goal_qpos, goal_btn, cube_tol=0.04, slide_tol=0.03) -> bool:
    """Task-agnostic success: the final scene state matches the goal state.

    Use it after running a planner. The arm pose is ignored — only the objects
    the tasks are about have to line up.
    """
    return bool(
        np.linalg.norm(np.asarray(final_qpos)[CUBE] - np.asarray(goal_qpos)[CUBE]) < cube_tol
        and abs(final_qpos[DRAWER] - goal_qpos[DRAWER]) < slide_tol
        and abs(final_qpos[WINDOW] - goal_qpos[WINDOW]) < slide_tol
        and np.array_equal(final_btn, goal_btn)
    )


def reset_to(env, qpos, qvel, button_states) -> np.ndarray:
    """Put `env` in a saved start state and return the observation.

    `env` must be the scene env itself (`gymnasium.make('visual-scene-v0')` or
    `ogbench.make_env_and_datasets('visual-scene-play-v0', env_only=True)`).
    """
    env.reset()
    u = env.unwrapped
    u.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64),
                np.asarray(button_states).astype(int))
    return u.compute_observation()


def load_pairs(path) -> dict:
    """Read a `*_pairs.npz` written by either CLI into a plain dict of arrays."""
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    out["task_names"] = [str(t) for t in out["task_names"]]
    return out
