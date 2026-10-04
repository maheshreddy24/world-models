"""The HDF5 recording and the training windows cut from it.

`datasets/prepare_mmbench.py` writes one flat table, episodes back to back:

    pixels       (N, 224, 224, 3) uint8   lz4-compressed in 100-frame chunks
    observation  (N, 6)  float32          oracle state (cos/sin of both links, 2 joint velocities)
    action       (N, 1)  float32          action *taken at* row t; NaN on each episode's last row
    reward       (N,)    float32
    ep_offset, ep_len                     where each episode starts and how long it is

A training window is `seq_len` latent steps `frameskip` rows apart. Its
action for a latent step is the flattened block of the `frameskip` raw actions
in between, so the model sees `action_dim * frameskip` numbers per step.

Reads take a contiguous slice and stride it in numpy: with compressed chunks,
a strided HDF5 read costs about twice as much as reading the whole span.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the codec `pixels` is compressed with)
import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import Normalizer
from .transforms import ImageTransform

RAW_COLUMNS = ("pixels",)  # never z-scored


class H5Reader:
    """Lazily-opened HDF5 handle, safe to share with dataloader workers.

    An open h5py file cannot be pickled, so each process opens its own handle
    on first use.
    """

    def __init__(self, path: str | Path, rdcc_mb: int = 256):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found; build it with datasets/prepare_mmbench.py")
        self.rdcc_mb = rdcc_mb
        self._file: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            self.ep_len = f["ep_len"][:].astype(np.int64)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r", rdcc_nbytes=self.rdcc_mb * 1024 * 1024, rdcc_nslots=10_007)
        return self._file

    @property
    def num_episodes(self) -> int:
        return len(self.ep_len)

    @property
    def stats_path(self) -> Path:
        """Where `get_normalizer` caches this recording's z-score statistics."""
        return self.path.with_suffix(".stats.npz")

    def span(self, column: str, start: int, stop: int) -> np.ndarray:
        """Contiguous rows [start, stop) of one column."""
        return self.file[column][start:stop]

    def rows(self, column: str, indices) -> np.ndarray:
        """Arbitrary rows, in any order and with repeats (HDF5 wants sorted unique indices)."""
        unique, inverse = np.unique(np.asarray(indices), return_inverse=True)
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
    """Strided windows of `seq_len` latent steps that never cross an episode boundary.

    Items: `{obs_key: (seq_len, ...), "action": (seq_len, frameskip * A)}`.
    Pixels go through `image_transform`; vectors and actions are z-scored.
    """

    def __init__(
        self,
        reader: H5Reader,
        episodes: np.ndarray,
        seq_len: int,
        frameskip: int,
        obs_key: str = "pixels",
        normalizer: Normalizer | None = None,
        image_transform: ImageTransform | None = None,
        stride: int = 1,
    ):
        self.reader = reader
        self.seq_len = seq_len
        self.frameskip = frameskip
        self.obs_key = obs_key
        self.normalizer = normalizer
        self.image_transform = image_transform or ImageTransform()
        self.span = seq_len * frameskip  # rows per window
        self.starts = self._window_starts(np.asarray(episodes), stride)

    def _window_starts(self, episodes: np.ndarray, stride: int) -> np.ndarray:
        """Global start rows of every window that fits inside its episode.

        The last row of an episode has no action (NaN), so windows stop one row short.
        """
        starts = []
        for ep in episodes:
            last = int(self.reader.ep_len[ep]) - 1 - self.span
            if last >= 0:
                starts.append(int(self.reader.ep_offset[ep]) + np.arange(0, last + 1, stride))
        if not starts:
            raise ValueError(f"no episode fits {self.seq_len} steps at frameskip {self.frameskip}")
        return np.concatenate(starts)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[i])
        stop = start + self.span

        obs = self.reader.span(self.obs_key, start, stop)[:: self.frameskip]
        if self.obs_key in RAW_COLUMNS:
            obs = self.image_transform(obs)
        else:
            obs = torch.from_numpy(self._normalize(self.obs_key, obs))

        # (span, A) -> (seq_len, frameskip * A): the block that drives each latent step
        action = self._normalize("action", np.nan_to_num(self.reader.span("action", start, stop), nan=0.0))
        action = torch.from_numpy(action.reshape(self.seq_len, -1))
        return {self.obs_key: obs, "action": action}

    def _normalize(self, key: str, x: np.ndarray) -> np.ndarray:
        if self.normalizer is not None:
            x = self.normalizer.normalize(key, x)
        return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Construction helpers
# --------------------------------------------------------------------------- #
def split_episodes(num_episodes: int, val_episodes: int, seed: int, max_episodes: int | None = None):
    """Hold out whole episodes (at most 10% of them), never rows.

    Returns:
        (train_episodes, val_episodes) as index arrays.
    """
    episodes = np.random.default_rng(seed).permutation(num_episodes)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    n_val = min(val_episodes, max(1, len(episodes) // 10))
    return episodes[n_val:], episodes[:n_val]


def compute_normalizer(reader: H5Reader, keys, max_rows: int, seed: int = 0) -> Normalizer:
    """z-score statistics from a random subsample of rows."""
    total = int(reader.ep_offset[-1] + reader.ep_len[-1])
    n = min(max_rows, total)
    idx = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    return Normalizer.from_columns({k: reader.rows(k, idx) for k in keys if k not in RAW_COLUMNS})


def get_normalizer(cfg, reader: H5Reader, keys) -> Normalizer:
    """Cached statistics for `keys`, computed and cached on first use."""
    path = Path(cfg.data.stats_path or reader.stats_path)
    keys = tuple(k for k in keys if k not in RAW_COLUMNS)
    if path.exists():
        norm = Normalizer.load(path)
        if all(k in norm for k in keys):
            return norm
    norm = compute_normalizer(reader, keys, cfg.data.stats_max_rows, cfg.seed)
    try:
        norm.save(path)
    except OSError:  # a read-only dataset directory is not fatal
        pass
    return norm


def build_datasets(cfg, train_stride: int | None = None, val_stride: int | None = None):
    """The training and validation windows of `cfg.data.h5_path`.

    Returns:
        (train_set, val_set, reader, normalizer). The caller closes the reader.
    """
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, (cfg.data.obs_key, "action"))
    transform = ImageTransform(cfg.data.img_size)
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)

    def make(episodes, stride):
        return SequenceDataset(
            reader, episodes, cfg.data.seq_len, cfg.data.frameskip,
            obs_key=cfg.data.obs_key, normalizer=normalizer, image_transform=transform, stride=stride,
        )

    train_set = make(train_eps, cfg.data.train_stride if train_stride is None else train_stride)
    val_set = make(val_eps, cfg.data.val_stride if val_stride is None else val_stride)
    return train_set, val_set, reader, normalizer
