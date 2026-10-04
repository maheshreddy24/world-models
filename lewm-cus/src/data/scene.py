"""Datasets over a prepared OGBench scene-play store.

Two stages, two shapes of sample:

    stage 1  dynamics predictor  `SequenceDataset` (reused unchanged)
             strided latent-step windows inside one episode, exactly as on the
             cube backend — the store exposes `pixels` / `action` under the same
             names, so nothing about `SequenceDataset` had to be scene-specific.

    stage 2  diffusion planner   `ScenePairDataset`
             a frame, a goal frame it can reach, the expert's next `chunk` raw
             actions from that frame, and the proprio at it.

Where stage 2's goals come from is the whole design of this file.  A mined pair
`(t, g)` says "from row t, the expert reaches the v1 goal g".  Training the
policy only at `t` would teach it the first move of each task and nothing about
recovering half-way through, so a start row `c` is redrawn uniformly inside
`[t, g - chunk]`: the same goal, seen from every point on the way to it.  That
is the state distribution closed-loop replanning actually visits.

`scene.p_task` mixes those mined pairs with hindsight goals — any frame and a
later frame of the same episode — which cover the regions between tasks and are
effectively unlimited.  Tasks are sampled with equal probability by default:
`toggle_lock` has 80,385 mined pairs and `cube_into_drawer` 3,244, so uniform
sampling over pairs would almost never show the rare one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import Normalizer
from .ogbench import RAW_COLUMNS, SequenceDataset, get_normalizer
from .scene_state import STATE, VELOCITY_DIMS
from .scene_store import EFFECTOR, SceneStore, default_store_dir
from .scene_tasks import TASKS, load_pairs
from .transforms import ImageTransform

PROPRIO_HINT = (
    "Run `python datasets/prepare_scene.py --effector` to derive it from qpos, "
    "or train with policy.use_proprio=false."
)


# --------------------------------------------------------------------------- #
#  Proprio
# --------------------------------------------------------------------------- #
def scene_proprio(store: SceneStore, rows, first_rows) -> np.ndarray:
    """End-effector position and per-step velocity: `(n,) -> (n, 6)`.

    The same definition as `src.data.ogbench.ee_proprio`: position, then the
    displacement over the last env step.  `first_rows` is each row's episode
    start, so the first frame of an episode gets zero velocity instead of
    borrowing the previous episode's last frame.
    """
    store.require(EFFECTOR, PROPRIO_HINT)
    rows = np.asarray(rows, dtype=np.int64)
    pos = store.rows(EFFECTOR, rows)
    prev = store.rows(EFFECTOR, np.maximum(rows - 1, np.asarray(first_rows, np.int64)))
    return np.concatenate([pos, pos - prev], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Stage 2: goal-conditioned action chunks
# --------------------------------------------------------------------------- #
class ScenePairDataset(Dataset):
    """Start frame, goal frame, the next `chunk` expert actions, and proprio.

    Items are *sampled*, not enumerated: `__len__` is `length`, the number of
    draws that make up one epoch, and every draw picks a fresh goal.  A start
    row therefore pairs with many goals over training instead of being frozen to
    one, which is the same trick `GoalChunkDataset` uses on the cube backend.

    Args:
        store: the prepared `SceneStore`.
        pairs_path: a `*_pairs.npz` of mined v1 pairs; `None` means hindsight only.
        chunk: raw env actions per sample.
        p_task: probability of drawing a mined pair rather than a hindsight one.
        balance_tasks: draw the 7 tasks with equal probability.
        hindsight_min_gap, hindsight_max_gap: goal distance for hindsight draws.
        resample_start: redraw the start row inside `[t, g - chunk]` (see module
            docstring).  `False` always starts at the mined `t`.
        episodes: episode ids hindsight goals may come from (default: all).
        length: draws per epoch.
        use_proprio: include the `proprio` key (needs the `effector_pos` column).
        zero_goal_velocity: zero the velocity entries of the goal state. Only
            meaningful for the oracle state: evaluation goals come from the pairs
            file, which stores goal positions but not goal velocities, so they
            are zero there, and training goals have to look the same.
        seed: base seed; dataloader workers offset it so they do not agree.

    Returns per item:
        pixels   (2, 3, H, W)  current and goal frame, preprocessed as in training
        action   (chunk, A)    raw env actions (the policy normalises them itself)
        proprio  (6,)          raw end-effector pos + velocity, if enabled
        task_id  ()            mined task index, or -1 for a hindsight goal
    """

    def __init__(
        self,
        store: SceneStore,
        pairs_path: str | Path | None,
        chunk: int = 16,
        p_task: float = 0.5,
        balance_tasks: bool = True,
        hindsight_min_gap: int = 30,
        hindsight_max_gap: int = 160,
        resample_start: bool = True,
        episodes: np.ndarray | None = None,
        length: int = 200_000,
        obs_key: str = "pixels",
        normalizer: Normalizer | None = None,
        image_transform: ImageTransform | None = None,
        use_proprio: bool = True,
        zero_goal_velocity: bool = False,
        seed: int = 0,
    ):
        if hindsight_min_gap < chunk:
            raise ValueError(
                f"scene.hindsight_min_gap ({hindsight_min_gap}) must be >= policy.chunk "
                f"({chunk}); a goal closer than one chunk leaves no actions to supervise"
            )
        if not 0.0 <= p_task <= 1.0:
            raise ValueError(f"scene.p_task must be in [0, 1], got {p_task}")

        self.store = store
        self.chunk = chunk
        self.p_task = p_task if pairs_path else 0.0
        self.balance = balance_tasks
        self.min_gap, self.max_gap = hindsight_min_gap, hindsight_max_gap
        self.resample_start = resample_start
        self.length = length
        self.obs_key = obs_key
        self.normalizer = normalizer
        self.image_transform = image_transform or ImageTransform()
        self.use_proprio = use_proprio
        self.zero_goal_velocity = zero_goal_velocity
        self.seed = seed
        self._rng: np.random.Generator | None = None

        if use_proprio:
            store.require(EFFECTOR, PROPRIO_HINT)

        # Hindsight goals need room for the longest gap inside an episode.
        self.episodes = np.arange(store.num_episodes) if episodes is None else np.asarray(episodes)
        room = store.ep_len[self.episodes] - 1 - hindsight_max_gap
        self.episodes = self.episodes[room > 0]
        if len(self.episodes) == 0:
            raise ValueError(
                f"no episode is longer than scene.hindsight_max_gap ({hindsight_max_gap})"
            )

        self.nonempty: list[int] = []
        self.task_pairs, self.task_names = self._load_pairs(pairs_path)

    # ---- construction ---------------------------------------------------- #
    def _load_pairs(self, pairs_path):
        """Mined pairs grouped by task, dropping any the chunk does not fit in."""
        if not pairs_path:
            return [], list(TASKS)
        data = load_pairs(pairs_path)
        start, goal, task_id = data["start_idx"], data["goal_idx"], data["task_id"]

        # A pair is usable when the whole chunk fits before the goal and both rows
        # are the same episode of *this* store. A pair file indexes the recording
        # it was mined from, which is not necessarily the one being trained on —
        # v1_pairs.npz, for instance, comes from a recording of its own — and a
        # mismatch would otherwise show up as actions that cross an episode.
        def episode(rows):
            return np.searchsorted(self.store.ep_offset, np.clip(rows, 0, self.store.num_rows - 1), side="right")

        in_store = (start >= 0) & (goal < self.store.num_rows)
        keep = (goal - start >= self.chunk) & in_store & (episode(start) == episode(goal))
        dropped = int((~keep).sum())
        if dropped:
            print(
                f"data  | dropped {dropped:,}/{len(keep):,} mined pairs (shorter than "
                f"policy.chunk={self.chunk}, or not one episode of this store)"
            )
        if not keep.any():
            raise ValueError(
                f"{pairs_path} has no pair this store can use. Mined pairs index into the "
                f"recording they came from; this store has {self.store.num_rows:,} rows."
            )
        start, goal, task_id = start[keep], goal[keep], task_id[keep]

        names = data.get("task_names", list(TASKS))
        by_task = [np.stack([start[task_id == k], goal[task_id == k]], axis=1) for k in range(len(names))]
        self.nonempty = [k for k, pairs in enumerate(by_task) if len(pairs)]
        return by_task, list(names)

    # ---- sampling -------------------------------------------------------- #
    @property
    def rng(self) -> np.random.Generator:
        """Per-worker generator, created on first use inside the worker process."""
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng(self.seed + (info.id + 1 if info else 0))
        return self._rng

    def _draw_pair(self) -> tuple[int, int, int]:
        """A `(start, goal, task_id)` triple; `task_id` is -1 for hindsight."""
        if self.task_pairs and self.rng.random() < self.p_task:
            if self.balance:
                k = int(self.rng.choice(self.nonempty))
            else:
                sizes = np.array([len(p) for p in self.task_pairs], dtype=np.float64)
                k = int(self.rng.choice(len(sizes), p=sizes / sizes.sum()))
            t, g = self.task_pairs[k][self.rng.integers(len(self.task_pairs[k]))]
            return int(t), int(g), k

        ep = int(self.rng.choice(self.episodes))
        first = int(self.store.ep_offset[ep])
        last = first + int(self.store.ep_len[ep]) - 1
        gap = int(self.rng.integers(self.min_gap, self.max_gap + 1))
        t = int(self.rng.integers(first, last - gap + 1))
        return t, t + gap, -1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, _) -> dict[str, torch.Tensor]:
        t, goal, task_id = self._draw_pair()
        start = int(self.rng.integers(t, goal - self.chunk + 1)) if self.resample_start else t

        obs = self.store.rows(self.obs_key, [start, goal])
        if self.zero_goal_velocity:
            obs[1, VELOCITY_DIMS] = 0.0  # before z-scoring, exactly as at evaluation
        if self.obs_key in RAW_COLUMNS:
            obs = self.image_transform(obs)
        else:
            if self.normalizer is not None:
                obs = self.normalizer.normalize(self.obs_key, obs)
            obs = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))

        sample = {
            self.obs_key: obs,
            "action": torch.from_numpy(
                np.ascontiguousarray(self.store.span("action", start, start + self.chunk), dtype=np.float32)
            ),
            "task_id": torch.tensor(task_id, dtype=torch.long),
        }
        if self.use_proprio:
            episode = int(np.searchsorted(self.store.ep_offset, start, side="right") - 1)
            first = int(self.store.ep_offset[episode])
            sample["proprio"] = torch.from_numpy(scene_proprio(self.store, [start], [first])[0])
        return sample


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def compute_scene_policy_stats(store: SceneStore, max_rows: int = 500_000, seed: int = 0) -> dict:
    """Action min/max and proprio mean/std, the normalisation `DiffusionPolicy` holds.

    Actions are read whole (2M x 5 is small); proprio is estimated from a
    subsample, which is plenty for a mean and a standard deviation.
    """
    actions = np.asarray(store.column("action"))
    stats = {"action_low": actions.min(axis=0), "action_high": actions.max(axis=0)}

    if store.has(EFFECTOR):
        rng = np.random.default_rng(seed)
        n = min(max_rows, store.num_rows)
        rows = np.sort(rng.choice(store.num_rows, size=n, replace=False))
        firsts = store.ep_offset[np.searchsorted(store.ep_offset, rows, side="right") - 1]
        proprio = scene_proprio(store, rows, firsts)
        stats["proprio_mean"] = proprio.mean(axis=0)
        stats["proprio_std"] = np.maximum(proprio.std(axis=0), 1e-6)
    else:
        stats["proprio_mean"] = np.zeros(6, np.float32)
        stats["proprio_std"] = np.ones(6, np.float32)
    return stats


# --------------------------------------------------------------------------- #
#  Construction helpers
# --------------------------------------------------------------------------- #
def open_scene_stores(cfg) -> tuple[SceneStore, SceneStore]:
    """The train and validation stores named by `cfg.scene`."""
    scene = cfg.scene
    return SceneStore(scene.train_store_dir), SceneStore(scene.val_store_dir)


def build_scene_datasets(cfg, train_stride: int = 5, val_stride: int = 25):
    """Stage 1: strided sequence windows over the scene recording.

    The recording ships its own validation file, so the split is a whole
    separate set of episodes rather than a slice of the training ones.

    Returns:
        `(train_set, val_set, train_store, normalizer)`
    """
    train_store, val_store = open_scene_stores(cfg)
    normalizer = get_normalizer(cfg, train_store, (cfg.data.obs_key, "action"))
    transform = ImageTransform(cfg.data.img_size)

    def make(store, stride):
        return SequenceDataset(
            reader=store,
            episodes=np.arange(store.num_episodes)[: cfg.data.max_episodes],
            seq_len=cfg.data.seq_len,
            frameskip=cfg.data.frameskip,
            keys=(cfg.data.obs_key,),
            normalizer=normalizer,
            image_transform=transform,
            stride=stride,
        )

    return make(train_store, train_stride), make(val_store, val_stride), train_store, normalizer


def build_scene_policy_datasets(cfg):
    """Stage 2: goal-conditioned action chunks for the diffusion planner.

    Validation runs on the held-out recording.  It has no mined pairs of its own
    unless `scene.val_pairs` points at some (see `datasets/scene_v1.py`), so by default
    its goals are hindsight goals, which need no task labels.

    Returns:
        `(train_set, val_set, train_store, stats, action_dim)`
    """
    train_store, val_store = open_scene_stores(cfg)
    normalizer = get_normalizer(cfg, train_store, (cfg.data.obs_key, "action"))
    transform = ImageTransform(cfg.data.img_size)
    scene = cfg.scene

    def make(store, pairs_path, length, seed):
        return ScenePairDataset(
            store=store,
            pairs_path=pairs_path,
            chunk=cfg.policy.chunk,
            p_task=scene.p_task,
            balance_tasks=scene.balance_tasks,
            hindsight_min_gap=scene.hindsight_min_gap,
            hindsight_max_gap=scene.hindsight_max_gap,
            resample_start=scene.resample_start,
            episodes=np.arange(store.num_episodes)[: cfg.data.max_episodes],
            length=length,
            obs_key=cfg.data.obs_key,
            normalizer=normalizer,
            image_transform=transform,
            use_proprio=cfg.policy.use_proprio,
            zero_goal_velocity=cfg.data.obs_key == STATE,
            seed=seed,
        )

    train_set = make(train_store, scene.train_pairs_path, scene.samples_per_epoch, cfg.seed)
    val_set = make(val_store, scene.val_pairs_path, scene.val_samples, cfg.seed + 1000)
    stats = compute_scene_policy_stats(train_store, seed=cfg.seed)
    return train_set, val_set, train_store, stats, train_store.dims["action"]


__all__ = [
    "ScenePairDataset",
    "build_scene_datasets",
    "build_scene_policy_datasets",
    "compute_scene_policy_stats",
    "open_scene_stores",
    "scene_proprio",
    "default_store_dir",
]
