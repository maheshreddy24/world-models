"""Random-access store over an OGBench scene-play `.npz` recording.

The recording ships as one compressed `.npz`: 2,002,000 rows of 64x64 frames
(24.6 GB raw, 3.4 GB deflated) plus the simulator state behind them.  Deflate
cannot seek, so a dataloader cannot sample a random window out of it — reaching
row 1,500,000 means inflating the 1.5M rows in front of it.

`prepare_store` therefore inflates each column once into a plain `.npy` file,
and every read afterwards is a memory map.  The decompression streams in
chunks, so it costs a few hundred MB of RAM rather than the full 24.6 GB, and
the resulting directory is page-cached after the first epoch.

Columns are renamed on the way in so that `data.obs_key="pixels"` selects the
same thing here as on the HDF5 cube backend, and `SequenceDataset` reads either:

    npz key          store column     shape             dtype
    observations  -> pixels           (N, 64, 64, 3)    uint8
    actions       -> action           (N, 5)            float32
    qpos          -> qpos             (N, 25)           float32
    qvel          -> qvel             (N, 24)           float32
    button_states -> button_states    (N, 2)            int64
    terminals     -> episodes.npz     episode boundaries

`effector_pos` (N, 3) is not in the recording: `add_effector_column` derives it
from `qpos` by forward kinematics, and the diffusion policy's proprio needs it.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as npy_format

# npz key -> the name the rest of the codebase uses.
COLUMN_ALIASES = {"observations": "pixels", "actions": "action"}
# Columns copied by default; `terminals` becomes the episode index instead.
DEFAULT_COLUMNS = ("observations", "actions", "qpos", "qvel", "button_states")
# Copied when present, skipped when not: cube-double-play has no buttons.
OPTIONAL_COLUMNS = ("button_states",)

META_FILE = "meta.json"
EPISODES_FILE = "episodes.npz"
EFFECTOR = "effector_pos"

# MuJoCo site whose world position OGBench reports as `proprio/effector_pos`.
PINCH_SITE = "ur5e/robotiq/pinch"
SCENE_ENV_ID = "visual-scene-v0"


def default_store_dir(npz_path: str | Path) -> Path:
    """`foo.npz` -> `foo_store/`, next to the recording."""
    npz_path = Path(npz_path)
    return npz_path.with_name(npz_path.name.replace(".npz", "") + "_store")


# --------------------------------------------------------------------------- #
#  Conversion
# --------------------------------------------------------------------------- #
def _stream_member(zf: zipfile.ZipFile, member: str, out_path: Path, chunk_bytes: int) -> dict:
    """Inflate one `.npy` member of the zip straight into a memory-mapped file.

    Reading the member whole would materialise the full uncompressed array (24.6
    GB for the frames); this copies it a chunk at a time instead.
    """
    with zf.open(member) as fh:
        version = npy_format.read_magic(fh)
        shape, fortran_order, dtype = npy_format._read_array_header(fh, version)
        if fortran_order:
            raise ValueError(f"{member} is Fortran-ordered; the store only handles C order")

        out = npy_format.open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
        flat = out.reshape(-1)
        per_chunk = max(1, chunk_bytes // dtype.itemsize)
        filled = 0
        while filled < flat.size:
            want = min(per_chunk, flat.size - filled) * dtype.itemsize
            buf = fh.read(want)
            if not buf:
                raise ValueError(f"{member} ended after {filled} of {flat.size} items")
            values = np.frombuffer(buf, dtype=dtype)
            flat[filled : filled + len(values)] = values
            filled += len(values)
        out.flush()
        del out, flat
    return {"shape": list(shape), "dtype": dtype.str}


def episode_index(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Episode boundaries from the terminal flags: `(ep_offset, ep_len)`.

    A row is terminal when it is an episode's last row, so the offsets are the
    rows just after them.  Any tail after the final terminal is dropped: a
    truncated episode has no recorded end and would silently run into nothing.
    """
    ends = np.flatnonzero(np.asarray(terminals).ravel())
    if len(ends) == 0:
        raise ValueError("no terminal flags — cannot tell where episodes end")
    ep_offset = np.r_[0, ends[:-1] + 1].astype(np.int64)
    ep_len = (ends - ep_offset + 1).astype(np.int64)
    return ep_offset, ep_len


def prepare_store(
    npz_path: str | Path,
    store_dir: str | Path | None = None,
    columns: tuple[str, ...] = DEFAULT_COLUMNS,
    overwrite: bool = False,
    chunk_bytes: int = 256 << 20,
    verbose: bool = True,
) -> Path:
    """Inflate `npz_path` into a memory-mappable store directory.

    Args:
        npz_path: the recorded `visual-scene-play-v0*.npz`.
        store_dir: output directory (defaults to `<name>_store` beside the npz).
        columns: npz keys to copy, named as they appear in the recording.
        overwrite: rebuild even if a complete store is already there.
        chunk_bytes: bytes inflated per copy step, i.e. the peak buffer size.

    Returns:
        the store directory.
    """
    npz_path = Path(npz_path)
    store_dir = Path(store_dir) if store_dir else default_store_dir(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found")

    if store_dir.exists() and not overwrite:
        try:
            SceneStore(store_dir).close()
            if verbose:
                print(f"store  | {store_dir} already prepared (pass overwrite=True to rebuild)")
            return store_dir
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass  # incomplete or stale: fall through and rebuild

    store_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source": str(npz_path.resolve()), "columns": {}}
    with zipfile.ZipFile(npz_path) as zf:
        available = {name[:-4] for name in zf.namelist() if name.endswith(".npy")}
        columns = tuple(c for c in columns if c in available or c not in OPTIONAL_COLUMNS)
        missing = [c for c in (*columns, "terminals") if c not in available]
        if missing:
            raise KeyError(f"{npz_path.name} has no {missing}; it holds {sorted(available)}")

        for key in columns:
            name = COLUMN_ALIASES.get(key, key)
            if verbose:
                size = zf.getinfo(f"{key}.npy").file_size / 1e9
                print(f"store  | {key} -> {name}.npy ({size:.2f} GB raw)", flush=True)
            meta["columns"][name] = _stream_member(zf, f"{key}.npy", store_dir / f"{name}.npy", chunk_bytes)

        with zf.open("terminals.npy") as fh:
            terminals = np.lib.format.read_array(fh)

    ep_offset, ep_len = episode_index(terminals)
    np.savez(store_dir / EPISODES_FILE, ep_offset=ep_offset, ep_len=ep_len)
    meta["rows"] = int(ep_offset[-1] + ep_len[-1])
    meta["episodes"] = int(len(ep_len))
    (store_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"store  | {meta['episodes']:,} episodes x {ep_len[0]} steps -> {store_dir}")
    return store_dir


def add_effector_column(
    store_dir: str | Path,
    env_id: str = SCENE_ENV_ID,
    site: str = PINCH_SITE,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Derive `effector_pos` from `qpos` by forward kinematics.

    The recording stores joint angles but not the gripper's world position,
    which is what the policy's proprio is made of.  `mj_kinematics` recovers it
    exactly — the values match the env's own `proprio/effector_pos` to float
    precision — and the whole dataset takes a few seconds.
    """
    import mujoco  # imported lazily: the store itself needs no simulator

    store_dir = Path(store_dir)
    out_path = store_dir / f"{EFFECTOR}.npy"
    meta = json.loads((store_dir / META_FILE).read_text())
    if EFFECTOR in meta["columns"] and out_path.exists() and not overwrite:
        if verbose:
            print(f"store  | {out_path.name} already present")
        return out_path

    import gymnasium

    import ogbench  # noqa: F401  — registers the OGBench env ids

    env = gymnasium.make(env_id)
    env.reset()
    model = env.unwrapped._model
    data = mujoco.MjData(model)
    site_id = model.site(site).id

    qpos = np.load(store_dir / "qpos.npy", mmap_mode="r")
    out = npy_format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(qpos), 3))
    for i in range(len(qpos)):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        out[i] = data.site_xpos[site_id]
    out.flush()
    del out
    env.close()

    meta["columns"][EFFECTOR] = {"shape": [int(len(qpos)), 3], "dtype": "<f4"}
    (store_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"store  | {EFFECTOR} from {site} -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
#  Reading
# --------------------------------------------------------------------------- #
class SceneStore:
    """Memory-mapped columns plus the episode index, read-only.

    The interface deliberately matches `H5Reader` (`ep_offset`, `ep_len`,
    `span`, `rows`, `dims`), so `SequenceDataset` and the normaliser helpers do
    not care which backend they are handed.

    Memory maps are opened on first use inside whichever process touches them,
    because a `np.memmap` pickled to a dataloader worker would copy its bytes.
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        meta_path = self.store_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{self.store_dir} is not a prepared store. Run "
                f"`python datasets/prepare_scene.py` first (see README)."
            )
        meta = json.loads(meta_path.read_text())
        self.meta = meta
        self.columns = sorted(meta["columns"])
        self.num_rows = int(meta["rows"])
        self.dims = {
            name: (int(np.prod(spec["shape"][1:])) if len(spec["shape"]) > 1 else 1)
            for name, spec in meta["columns"].items()
        }
        for name in self.columns:
            if not (self.store_dir / f"{name}.npy").exists():
                raise FileNotFoundError(f"{self.store_dir}/{name}.npy is missing; rebuild the store")

        episodes = np.load(self.store_dir / EPISODES_FILE)
        self.ep_offset = episodes["ep_offset"].astype(np.int64)
        self.ep_len = episodes["ep_len"].astype(np.int64)
        self._maps: dict[str, np.memmap] = {}

    # ---- identity -------------------------------------------------------- #
    @property
    def path(self) -> Path:
        return self.store_dir

    @property
    def stats_path(self) -> Path:
        """Where `get_normalizer` caches this dataset's z-score statistics."""
        return self.store_dir / "stats.npz"

    @property
    def num_episodes(self) -> int:
        return len(self.ep_len)

    def has(self, column: str) -> bool:
        return column in self.dims

    def require(self, column: str, hint: str = "") -> None:
        if not self.has(column):
            raise KeyError(f"{self.store_dir} has no {column!r} column. {hint}".strip())

    # ---- access ---------------------------------------------------------- #
    def column(self, name: str) -> np.memmap:
        if name not in self._maps:
            if name not in self.dims:
                raise KeyError(f"unknown column {name!r}; the store holds {self.columns}")
            self._maps[name] = np.load(self.store_dir / f"{name}.npy", mmap_mode="r")
        return self._maps[name]

    def span(self, column: str, start: int, stop: int) -> np.ndarray:
        """Contiguous row slice of one column, as an ordinary array.

        The copy is deliberate: a slice of the map is a read-only view into a
        24.6 GB file, and torch warns every time one is turned into a tensor.
        Reads here are a handful of frames, so the copy costs nothing.
        """
        return np.array(self.column(column)[start:stop])

    def rows(self, column: str, indices) -> np.ndarray:
        """Gather arbitrary rows, in any order and with repeats."""
        return np.array(self.column(column)[np.asarray(indices)])

    def episode_bounds(self, episode: int) -> tuple[int, int]:
        """`(first_row, last_row)` of an episode, both inclusive."""
        offset = int(self.ep_offset[episode])
        return offset, offset + int(self.ep_len[episode]) - 1

    def close(self) -> None:
        self._maps.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_maps"] = {}  # never pickle an open map
        return state

    def __enter__(self) -> "SceneStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
