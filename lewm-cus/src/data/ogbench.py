"""OGBench trajectory dataset.

The recorded file is one flat table of 2.01M rows (10k episodes x 201 steps),
with `ep_offset` / `ep_len` marking the episode boundaries.  Columns used here:

    pixels      (N, 224, 224, 3) uint8   front camera
    observation (N, 28)  float64         oracle state
    action      (N, 5)   float32         raw control; NaN on the last row of an episode
    qpos, qvel                           simulator state, for seeding eval episodes
    privileged_block_0_pos / _quat       cube pose, for setting the eval target

A training sample is `seq_len` latent steps taken `frameskip` raw steps apart.
The action of a latent step is the flattened block of the `frameskip` raw
actions it spans, so the model sees `action_dim * frameskip` numbers per step
and one latent step of planning commits to `frameskip` env steps.

Reads go through a contiguous slice that is then strided in numpy: the pixel
column is stored in 100-frame compressed chunks, so asking HDF5 for a strided
hyperslab costs about twice as much as reading the span and throwing rows away.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  — registers the codec `pixels` is compressed with
import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import Normalizer
from .transforms import ImageTransform

# Columns that are never z-scored.
RAW_COLUMNS = ("pixels",)


class H5Reader:
    """Lazily-opened HDF5 handle, safe to share with dataloader workers.

    An open h5py file cannot be pickled, so the handle is created on first use
    inside whichever process ends up touching it.
    """

    def __init__(self, path: str | Path, rdcc_mb: int = 256):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Download the OGBench data and point "
                "data.h5_path at it (see README)."
            )
        self.rdcc_mb = rdcc_mb
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.columns = sorted(f.keys())
            self.dims = {
                k: (int(f[k].shape[1]) if f[k].ndim > 1 else 1)
                for k in f
                if f[k].ndim <= 2 and f[k].dtype != object
            }

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            # A generous chunk cache pays for itself: one pixel chunk is 100
            # frames, and a sequence read touches one or two of them.
            self._file = h5py.File(
                self.path, "r", rdcc_nbytes=self.rdcc_mb * 1024 * 1024, rdcc_nslots=10_007
            )
        return self._file

    @property
    def num_episodes(self) -> int:
        return len(self.ep_len)

    def span(self, column: str, start: int, stop: int) -> np.ndarray:
        """Contiguous row slice of one column."""
        return self.file[column][start:stop]

    def rows(self, column: str, indices) -> np.ndarray:
        """Gather arbitrary rows, in any order and with repeats.

        HDF5 fancy indexing only accepts strictly increasing indices, so the
        read is done on the sorted unique set and expanded back afterwards.
        """
        indices = np.asarray(indices)
        unique, inverse = np.unique(indices, return_inverse=True)
        return self.file[column][unique][inverse]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_file"] = None  # never pickle the handle
        return state


class SequenceDataset(Dataset):
    """Fixed-length strided sequences that never cross an episode boundary.

    Args:
        reader: the shared `H5Reader`.
        episodes: episode ids this split owns.
        seq_len: latent steps per sample.
        frameskip: raw env steps per latent step.
        keys: columns to return besides `action` (e.g. `("pixels",)`).
        normalizer: z-score stats for non-pixel columns.
        image_transform: applied to pixel columns.
        stride: spacing between candidate start rows; >1 shrinks the epoch.
    """

    def __init__(
        self,
        reader: H5Reader,
        episodes: np.ndarray,
        seq_len: int,
        frameskip: int,
        keys: tuple[str, ...] = ("pixels",),
        normalizer: Normalizer | None = None,
        image_transform: ImageTransform | None = None,
        stride: int = 1,
    ):
        self.reader = reader
        self.seq_len = seq_len
        self.frameskip = frameskip
        self.keys = tuple(dict.fromkeys(keys + ("action",)))
        self.normalizer = normalizer
        self.image_transform = image_transform or ImageTransform()
        self.span = seq_len * frameskip
        self.starts = self._build_index(np.asarray(episodes), stride)

    def _build_index(self, episodes: np.ndarray, stride: int) -> np.ndarray:
        """Global row indices where a full sequence fits inside one episode.

        The final row of every episode carries a NaN action (an artefact of how
        the trajectories were recorded), so windows stop one row short of it.
        """
        starts = []
        for ep in episodes:
            usable = int(self.reader.ep_len[ep]) - 1  # drop the NaN-action row
            last = usable - self.span
            if last < 0:
                continue
            starts.append(int(self.reader.ep_offset[ep]) + np.arange(0, last + 1, stride))
        if not starts:
            raise ValueError(
                f"no episode is long enough for {self.seq_len} steps at "
                f"frameskip {self.frameskip} ({self.span} raw steps)"
            )
        return np.concatenate(starts)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[i])
        stop = start + self.span
        sample: dict[str, torch.Tensor] = {}

        for key in self.keys:
            block = self.reader.span(key, start, stop)
            if key == "action":
                # (span, A) -> (seq_len, frameskip * A): consecutive raw actions
                # are grouped into the block that drives one latent step.
                block = np.nan_to_num(block, nan=0.0)
                if self.normalizer is not None:
                    block = self.normalizer.normalize("action", block)
                sample[key] = torch.from_numpy(
                    np.ascontiguousarray(block, dtype=np.float32).reshape(self.seq_len, -1)
                )
            elif key in RAW_COLUMNS:
                sample[key] = self.image_transform(block[:: self.frameskip])
            else:
                block = block[:: self.frameskip]
                if self.normalizer is not None:
                    block = self.normalizer.normalize(key, block)
                sample[key] = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))

        return sample


# --------------------------------------------------------------------------- #
#  Construction helpers
# --------------------------------------------------------------------------- #
def compute_normalizer(reader: H5Reader, keys, max_rows: int, seed: int = 0) -> Normalizer:
    """Estimate z-score stats from a random subsample of rows."""
    total = int(reader.ep_offset[-1] + reader.ep_len[-1])
    rng = np.random.default_rng(seed)
    n = min(max_rows, total)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    columns = {k: reader.file[k][idx] for k in keys if k not in RAW_COLUMNS}
    return Normalizer.from_columns(columns)


def get_normalizer(cfg, reader: H5Reader, keys) -> Normalizer:
    """Load cached statistics, computing and caching them on first use."""
    path = Path(cfg.data.stats_path or reader.path.with_suffix(".stats.npz"))
    keys = tuple(k for k in keys if k not in RAW_COLUMNS)
    if path.exists():
        norm = Normalizer.load(path)
        if all(k in norm for k in keys):
            return norm
    norm = compute_normalizer(reader, keys, cfg.data.stats_max_rows, cfg.seed)
    try:
        norm.save(path)
    except OSError:  # read-only dataset directory is not fatal
        pass
    return norm


def split_episodes(num_episodes: int, val_episodes: int, seed: int, max_episodes: int | None = None):
    """Hold out whole episodes, never rows.

    Splitting by row would put frames from the same trajectory on both sides of
    the split and make validation loss meaningless.
    """
    rng = np.random.default_rng(seed)
    episodes = rng.permutation(num_episodes)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    n_val = min(val_episodes, max(1, len(episodes) // 10))
    return episodes[n_val:], episodes[:n_val]


def build_datasets(cfg, train_stride: int = 1, val_stride: int = 4):
    """Everything train.py needs from the data side.

    Returns:
        (train_set, val_set, reader, normalizer)
    """
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    keys = (cfg.data.obs_key, "action")
    normalizer = get_normalizer(cfg, reader, keys)
    transform = ImageTransform(cfg.data.img_size)

    train_eps, val_eps = split_episodes(
        reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes
    )

    def make(episodes, stride):
        return SequenceDataset(
            reader=reader,
            episodes=episodes,
            seq_len=cfg.data.seq_len,
            frameskip=cfg.data.frameskip,
            keys=(cfg.data.obs_key,),
            normalizer=normalizer,
            image_transform=transform,
            stride=stride,
        )

    return make(train_eps, train_stride), make(val_eps, val_stride), reader, normalizer


# --------------------------------------------------------------------------- #
#  Evaluation episodes
# --------------------------------------------------------------------------- #
#  Columns pulled for each evaluation episode: the simulator state to start
#  from, and the cube pose that defines the target.
EVAL_START_COLUMNS = ("qpos", "qvel")
EVAL_GOAL_COLUMNS = ("privileged_block_0_pos", "privileged_block_0_quat")


def sample_eval_episodes(
    reader: H5Reader,
    num: int,
    goal_offset: int,
    seed: int = 0,
    episodes: np.ndarray | None = None,
    obs_key: str = "pixels",
    min_goal_distance: float = 0.10,
    oversample: int = 40,
) -> dict:
    """Draw evaluation tasks of the form "get from here to there in N steps".

    A task is a recorded start state plus the state the *expert* reached
    `goal_offset` raw steps later, which guarantees every goal is reachable
    inside the budget and makes the success rate comparable across models.

    Tasks where the cube barely moves are dropped. The expert spends much of
    each episode reaching for the cube before touching it, so over a 25-step
    window the cube sits inside the 4cm success radius of its own goal about
    40% of the time — those tasks are already solved at t=0 and a random policy
    "succeeds" at them, which quietly inflates every number reported.
    Requiring real displacement makes the success rate mean what it says.

    Args:
        reader: the dataset.
        num: number of tasks.
        goal_offset: raw steps between the start and the goal.
        seed: sampling seed.
        episodes: episode pool to draw from (defaults to all).
        obs_key: which observation the planner consumes at the goal.
        min_goal_distance: metres the cube must travel for a task to count.
            The simulator's success threshold is 0.04, so tasks near that are
            won before the planner acts. Set 0 to disable filtering.
        oversample: candidates drawn per requested task before filtering.

    Returns:
        dict with `qpos`/`qvel` (start state), `goal_obs` (what the planner
        sees), `goal_pixels` (for video), the privileged goal pose, the
        `episode`/`start` indices, and `goal_distance` per task.
    """
    pool = np.arange(reader.num_episodes) if episodes is None else np.asarray(episodes)
    lengths = reader.ep_len[pool]
    # Need room for start + goal_offset inside the episode, minus the NaN row.
    usable = lengths - 1 - goal_offset
    pool, usable = pool[usable > 0], usable[usable > 0]
    if len(pool) == 0:
        raise ValueError(f"no episode is longer than goal_offset={goal_offset}")

    rng = np.random.default_rng(seed)
    n_draw = num if min_goal_distance <= 0 else num * oversample
    picks = rng.choice(len(pool), size=n_draw, replace=n_draw > len(pool))
    eps = pool[picks]
    starts = (rng.random(n_draw) * usable[picks]).astype(np.int64)
    start_rows = reader.ep_offset[eps] + starts
    goal_rows = start_rows + goal_offset

    cube_start = reader.rows("privileged_block_0_pos", start_rows)
    cube_goal = reader.rows("privileged_block_0_pos", goal_rows)
    distance = np.linalg.norm(cube_goal - cube_start, axis=1)

    if min_goal_distance > 0:
        keep = np.nonzero(distance > min_goal_distance)[0]
        # Oversampling can draw the same (episode, start) twice; evaluating one
        # task twice would just weight it double.
        _, first = np.unique(np.stack([eps[keep], starts[keep]], 1), axis=0, return_index=True)
        keep = keep[np.sort(first)][:num]
        if len(keep) < num:
            raise ValueError(
                f"only {len(keep)} of {n_draw} candidates move the cube more than "
                f"{min_goal_distance}m; raise eval.goal_offset, lower "
                "eval.min_goal_distance, or widen the episode pool"
            )
        eps, starts = eps[keep], starts[keep]
        start_rows, goal_rows, distance = start_rows[keep], goal_rows[keep], distance[keep]

    out = {"episode": eps, "start": starts, "goal_distance": distance}
    for col in EVAL_START_COLUMNS:
        out[col] = reader.rows(col, start_rows)
    for col in EVAL_GOAL_COLUMNS:
        out[col] = reader.rows(col, goal_rows)
    out["goal_pixels"] = reader.rows("pixels", goal_rows)
    out["goal_obs"] = out["goal_pixels"] if obs_key == "pixels" else reader.rows(obs_key, goal_rows)
    # The expert's own path to the goal, shown next to the agent in videos.
    out["reference"] = np.stack(
        [reader.span("pixels", int(r), int(r) + goal_offset + 1) for r in start_rows]
    )
    return out
