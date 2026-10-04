"""Mine the training pairs: every valid v1 start/goal in a recording.

    python datasets/scene_v1.py --data ogbench_scene_single/visual-scene-play-v0.npz \
                       --out  ogbench_scene_single/train_pairs.npz

Same seven tasks and the same mining code as `make_v1_pairs.py`, run with
different settings: no per-task cap and no dedupe, so every valid start is kept,
and only the row indices are written (187k pairs, a few MB).  The frames stay in
the recording, which `src.data.SceneStore` memory-maps.

The pairs index the recording they were mined from, so mine the *training*
recording here and keep `v1_pairs.npz` — the evaluation set — from a different
one.

The dataset that reads this file is `src.data.ScenePairDataset`, built for you
by `train_policy.py --preset scene_policy`.  It mixes these mined pairs with
hindsight goals and redraws the start row on the way to each goal; see
`src/data/scene.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.scene_tasks import (  # noqa: E402
    MAX_GAP,
    MIN_GAP,
    SETTLE,
    STRIDE,
    TASKS,
    compute_features,
    mine,
)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="the TRAINING recording (never the eval one)")
    p.add_argument("--out", default="ogbench_scene_single/train_pairs.npz")
    p.add_argument("--min_gap", type=int, default=MIN_GAP)
    p.add_argument("--max_gap", type=int, default=MAX_GAP)
    p.add_argument("--settle", type=int, default=SETTLE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d = dict(np.load(args.data))
    f = compute_features(d)

    task_id, start_idx, goal_idx = [], [], []
    for k, task in enumerate(TASKS):
        pairs = mine(d, f, task, args.min_gap, args.max_gap, args.settle, args.stride, rng, dedupe=False)
        print(f"{task:18s} {len(pairs):6d} pairs")
        task_id += [k] * len(pairs)
        start_idx += list(pairs[:, 0]) if len(pairs) else []
        goal_idx += list(pairs[:, 1]) if len(pairs) else []

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, task_names=np.array(TASKS), task_id=np.array(task_id),
             start_idx=np.array(start_idx), goal_idx=np.array(goal_idx))
    print("Saved", args.out, "| total task pairs:", len(task_id))


if __name__ == "__main__":
    main()
