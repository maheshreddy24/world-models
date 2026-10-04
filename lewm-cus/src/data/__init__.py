"""Datasets, normalisation and pixel transforms for the OGBench recordings.

Two backends sit behind one interface, chosen by `data.backend`:

    "h5"     cube-single, an HDF5 table          (`ogbench.py`)
    "scene"  scene-play, a memory-mapped store   (`scene_store.py`, `scene.py`)

Import `build_datasets` / `build_policy_datasets` / `open_store` from here and
the choice stays in the config.
"""

from .backends import BACKENDS, build_datasets, build_policy_datasets, open_store
from .normalize import Normalizer
from .ogbench import (
    GoalChunkDataset,
    H5Reader,
    SequenceDataset,
    compute_normalizer,
    compute_policy_stats,
    cube_proprio,
    ee_proprio,
    get_normalizer,
    sample_eval_episodes,
    split_episodes,
)
from .scene import (
    ScenePairDataset,
    build_scene_datasets,
    build_scene_policy_datasets,
    compute_scene_policy_stats,
    open_scene_stores,
    scene_proprio,
)
from .scene_store import (
    EFFECTOR,
    SceneStore,
    add_effector_column,
    default_store_dir,
    prepare_store,
)
from .scene_state import STATE, STATE_DIM, VELOCITY_DIMS, SceneState, add_state_column, zero_velocities
from .scene_tasks import TASKS, is_success, load_pairs, reset_to
from .transforms import ImageTransform

__all__ = [
    # backend-agnostic entry points
    "BACKENDS", "build_datasets", "build_policy_datasets", "open_store",
    # shared pieces
    "Normalizer", "ImageTransform", "SequenceDataset",
    "compute_normalizer", "get_normalizer",
    # cube / h5 backend
    "H5Reader", "GoalChunkDataset", "cube_proprio", "ee_proprio", "compute_policy_stats",
    "sample_eval_episodes", "split_episodes",
    # scene backend
    "SceneStore", "prepare_store", "add_effector_column", "default_store_dir", "EFFECTOR",
    "ScenePairDataset", "scene_proprio", "compute_scene_policy_stats",
    "build_scene_datasets", "build_scene_policy_datasets", "open_scene_stores",
    "TASKS", "is_success", "load_pairs", "reset_to",
    "STATE", "STATE_DIM", "VELOCITY_DIMS", "SceneState", "add_state_column", "zero_velocities",
]
