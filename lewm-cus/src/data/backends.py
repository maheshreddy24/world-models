"""Which dataset the scripts are pointed at.

Two recordings live behind the same training code:

    "h5"     OGBench cube-single, one HDF5 table (`src.data.ogbench`)
    "scene"  OGBench scene-play, a memory-mapped npz store (`src.data.scene`)

`data.backend` picks one, and every script goes through the three functions
here rather than naming a reader.  Adding a third recording means writing its
store plus a `build_*` pair and adding one line to each table below — nothing in
`train.py`, `train_policy.py` or `rollout.py` has to know about it.

Each `build_*` returns the store it opened, and the caller closes it.
"""

from __future__ import annotations

from .ogbench import H5Reader, build_datasets as _build_h5_datasets, get_normalizer
from .scene import build_scene_datasets, build_scene_policy_datasets, open_scene_stores

BACKENDS = ("h5", "scene")


def _check(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unknown data.backend {backend!r}; expected one of {BACKENDS}")
    return backend


def open_store(cfg):
    """The training store for `cfg.data.backend`, without building any dataset."""
    if _check(cfg.data.backend) == "scene":
        return open_scene_stores(cfg)[0]
    return H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)


def build_datasets(cfg, **kwargs):
    """Stage 1: sequence windows for the dynamics predictor.

    Returns:
        `(train_set, val_set, store, normalizer)`
    """
    if _check(cfg.data.backend) == "scene":
        return build_scene_datasets(cfg, **kwargs)
    return _build_h5_datasets(cfg, **kwargs)


def build_policy_datasets(cfg):
    """Stage 2: goal-conditioned action chunks for the diffusion planner.

    Returns:
        `(train_set, val_set, store, policy_stats, action_dim)` where
        `policy_stats` is the kwargs of `DiffusionPolicy.set_stats`.
    """
    pcfg = cfg.policy
    if _check(cfg.data.backend) == "scene":
        if (pcfg.goal, pcfg.proprio, pcfg.use_contact) != ("latent", "ee", False):
            raise ValueError(
                "the scene policy data carries goal frames and end-effector proprio only: set "
                "policy.goal=latent policy.proprio=ee policy.use_contact=false (--preset scene_policy does)"
            )
        return build_scene_policy_datasets(cfg)

    # The cube backend keeps its own episode split and reads its statistics
    # straight off the HDF5 table.
    import numpy as np

    from .ogbench import GoalChunkDataset, compute_policy_stats, split_episodes
    from .transforms import ImageTransform

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, (cfg.data.obs_key, "action"))
    train_eps, val_eps = split_episodes(
        reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes
    )

    conditions = dict(goal=pcfg.goal, proprio=pcfg.proprio, use_contact=pcfg.use_contact)

    def make(episodes, stride):
        return GoalChunkDataset(
            reader=reader,
            episodes=np.asarray(episodes),
            chunk=pcfg.chunk,
            goal_offset_min=pcfg.goal_offset_min,
            goal_offset_max=pcfg.goal_offset_max,
            obs_key=cfg.data.obs_key,
            normalizer=normalizer,
            image_transform=ImageTransform(cfg.data.img_size),
            stride=stride,
            **conditions,
        )

    stats = compute_policy_stats(reader, train_eps, **conditions)
    return make(train_eps, 1), make(val_eps, 10), reader, stats, reader.dims["action"]


__all__ = ["BACKENDS", "build_datasets", "build_policy_datasets", "open_store"]
