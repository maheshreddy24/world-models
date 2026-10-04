"""Turn the OGBench scene-play `.npz` recordings into memory-mappable stores.

    python datasets/prepare_scene.py                      # train + val recordings
    python datasets/prepare_scene.py --effector           # and the proprio column
    python datasets/prepare_scene.py --state              # and the 39-d oracle state (for scene_oracle)
    python datasets/prepare_scene.py --split val          # just the small one, to try it
    python datasets/prepare_scene.py --overwrite          # rebuild from scratch

Run this once before `train.py --preset scene_pixels`.  The frames are deflated
inside the npz, and deflate cannot seek, so sampling a random window out of the
file would mean inflating every row in front of it.  This inflates each column
once into a plain `.npy` that the dataloader memory-maps.

Cost: about 25 GB on disk and a few minutes for the training recording, a tenth
of that for the validation one.  Peak memory stays a few hundred MB — the copy
streams in chunks rather than materialising the 24.6 GB array.

`--effector` adds `effector_pos`, the gripper's world position, which the
recording does not store.  It is recovered from `qpos` by forward kinematics
(seconds for the whole dataset, and it matches the env's own
`proprio/effector_pos` to float precision) and is what the diffusion policy's
proprio is built from.

`--state` adds `observation`, OGBench's own scene state vector rebuilt from
`qpos`/`qvel`/`button_states` (39 of its 40 entries; see src/data/scene_state.py
for the one that cannot be rebuilt). It is what the oracle world model trains
on. About 230 us a row, split over `--workers` processes: a minute or so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from src.data.scene_state import add_state_column
from src.data.scene_store import add_effector_column, prepare_store


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="both", choices=("train", "val", "both"))
    parser.add_argument("--effector", action="store_true", help="also derive the proprio column")
    parser.add_argument("--state", action="store_true", help="also derive the 39-d oracle state column")
    parser.add_argument("--workers", type=int, default=16, help="processes for --state")
    parser.add_argument("--overwrite", action="store_true", help="rebuild even if a store exists")
    parser.add_argument("--chunk-mb", type=int, default=256, help="inflate buffer size, i.e. peak RAM")
    args, overrides = parser.parse_known_args(argv)

    # Reuse the config so the paths are the exact ones training will read.
    cfg, _ = get_config(overrides, default_preset="scene_pixels")
    splits = {
        "train": [(cfg.scene.train_npz_path, cfg.scene.train_store_dir)],
        "val": [(cfg.scene.val_npz_path, cfg.scene.val_store_dir)],
    }
    todo = splits["train"] + splits["val"] if args.split == "both" else splits[args.split]

    for npz_path, store_dir in todo:
        print(f"\n=== {npz_path.name} ===")
        prepare_store(npz_path, store_dir, overwrite=args.overwrite, chunk_bytes=args.chunk_mb << 20)
        if args.effector:
            add_effector_column(store_dir, env_id=cfg.scene.env_id, overwrite=args.overwrite)
        if args.state:
            add_state_column(store_dir, workers=args.workers, overwrite=args.overwrite)

    print("\nready | train.py --preset scene_pixels")


if __name__ == "__main__":
    main()
