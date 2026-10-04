"""OGBench's scene state vector, rebuilt from what the recording stores.

The scene recordings hold frames plus `qpos`, `qvel` and `button_states`, but
not the 40-d state observation OGBench's `scene-v0` env returns.  Every entry of
that vector is a function of those three arrays except one: `gripper_contact`,
the external contact force on the gripper pad.  MuJoCo only computes it inside a
physics step, from actuator forces the recording does not keep, so rebuilt from
a stored state it reads 0 while the live env reports up to 1.  It is dropped.
The other 39 entries match the live env exactly.

    0-5    arm joint positions          18-20  cube position
    6-11   arm joint velocities         21-24  cube quaternion
    12-14  effector position            25-26  cube yaw cos/sin
    15-16  effector yaw cos/sin         27-34  buttons x2: one-hot (2), position, velocity
    17     gripper opening              35-38  drawer position/velocity, window position/velocity

Training and evaluation both call `SceneState`, never the env's own
`compute_observation`, so the oracle the model trains on and the one it is
evaluated on come from one function.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.lib.format as npy_format

STATE = "observation"  # same column name as the cube backend's oracle state
STATE_ENV_ID = "scene-v0"  # the state-observation twin of visual-scene-v0
GRIPPER_CONTACT = 18  # index in OGBench's 40-d vector; see module docstring
STATE_DIM = 39
# Entries that are velocities. Evaluation goals come with positions only (the
# pairs file stores no goal qvel), so training goals have these zeroed as well.
VELOCITY_DIMS = np.r_[6:12, 30, 34, 36, 38]


class SceneState:
    """`(qpos, qvel, button_states) -> (39,)`: the oracle state of one scene configuration.

    Holds one headless `scene-v0` env and uses it as a pure function: set the
    state, read the observation.  No rendering, so no GL context.
    """

    def __init__(self, env_id: str = STATE_ENV_ID):
        import gymnasium

        import ogbench  # noqa: F401  — registers the OGBench env ids

        self._env = gymnasium.make(env_id)
        self._env.reset(seed=0)
        self._u = self._env.unwrapped
        full = self._u.compute_observation().shape[0]
        self._keep = np.delete(np.arange(full), GRIPPER_CONTACT)
        if len(self._keep) != STATE_DIM:
            raise ValueError(f"{env_id} returns a {full}-d state; expected {STATE_DIM + 1}")

    def __call__(self, qpos, qvel, button_states) -> np.ndarray:
        self._u.set_state(
            np.asarray(qpos, np.float64), np.asarray(qvel, np.float64), np.asarray(button_states).astype(int)
        )
        return self._u.compute_observation()[self._keep].astype(np.float32)

    def batch(self, qpos, qvel, button_states) -> np.ndarray:
        """Row-wise over `(n, 25)`, `(n, 24)`, `(n, 2)` -> `(n, 39)`."""
        return np.stack([self(q, v, b) for q, v, b in zip(qpos, qvel, button_states)])

    def close(self) -> None:
        self._env.close()


def zero_velocities(state: np.ndarray) -> np.ndarray:
    """A copy of `(..., 39)` states with every velocity entry set to 0."""
    out = np.array(state, dtype=np.float32, copy=True)
    out[..., VELOCITY_DIMS] = 0.0
    return out


# --------------------------------------------------------------------------- #
#  Store column
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _init_worker(store_dir: str, out_path: str) -> None:
    d = Path(store_dir)
    _WORKER["fn"] = SceneState()
    _WORKER["q"] = np.load(d / "qpos.npy", mmap_mode="r")
    _WORKER["v"] = np.load(d / "qvel.npy", mmap_mode="r")
    _WORKER["b"] = np.load(d / "button_states.npy", mmap_mode="r")
    _WORKER["out"] = np.load(out_path, mmap_mode="r+")


def _fill(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    fn, out = _WORKER["fn"], _WORKER["out"]
    q, v, b = _WORKER["q"], _WORKER["v"], _WORKER["b"]
    for i in range(lo, hi):
        out[i] = fn(q[i], v[i], b[i])
    out.flush()
    return hi - lo


def add_state_column(
    store_dir: str | Path,
    workers: int = 16,
    chunk_rows: int = 20_000,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Write the 39-d oracle state of every row as the store's `observation` column.

    About 230 us a row, so the 2M-row training store takes ~8 minutes on one
    core; `workers` processes split it into chunks. The column is registered in
    `meta.json` only once it is complete, so an interrupted run leaves nothing
    half-written behind.
    """
    import multiprocessing as mp

    store_dir = Path(store_dir)
    out_path = store_dir / f"{STATE}.npy"
    meta = json.loads((store_dir / "meta.json").read_text())
    if STATE in meta["columns"] and out_path.exists() and not overwrite:
        if verbose:
            print(f"store  | {out_path.name} already present")
        return out_path

    rows = int(meta["rows"])
    npy_format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(rows, STATE_DIM)).flush()
    chunks = [(lo, min(lo + chunk_rows, rows)) for lo in range(0, rows, chunk_rows)]
    done = 0
    # fork, not spawn: spawned workers re-import the caller's main module, so a
    # script without an `if __name__ == "__main__"` guard would re-run itself in
    # every worker. Nothing here holds a GL context or CUDA when the pool forks.
    with mp.get_context("fork").Pool(
        max(1, workers), initializer=_init_worker, initargs=(str(store_dir), str(out_path))
    ) as pool:
        for n in pool.imap_unordered(_fill, chunks):
            done += n
            if verbose:
                print(f"\rstore  | {STATE}: {done:,}/{rows:,} rows", end="", flush=True)
    if verbose:
        print()

    meta["columns"][STATE] = {"shape": [rows, STATE_DIM], "dtype": "<f4"}
    (store_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"store  | {STATE} (39-d oracle state) -> {out_path}")
    return out_path
