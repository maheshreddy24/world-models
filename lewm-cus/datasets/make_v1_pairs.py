"""Mine the held-out evaluation set: v1 (Direct) start/goal pairs for OGBench Scene.

    python datasets/make_v1_pairs.py --data <recording>.npz --out ogbench_scene_single/v1_pairs.npz --per_task 50

Every pair comes from the same episode, so the goal is reachable and in
distribution, and everything except the task's target object is identical in the
two frames.  The output is *self-contained*: it carries the simulator state to
reset to (`start_qpos`/`start_qvel`/`start_btn`), the state success is measured
against (`goal_qpos`/`goal_btn`) and both frames, so `eval_scene.py` never has
to open the recording it was mined from.

Mine this from a recording the world model and the policy do not train on.

The task definitions live in `src/data/scene_tasks.py`; this file is the CLI
around them.  `scene_v1.py` is the same mining run with different settings, for
the training pairs.
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
    is_success,  # noqa: F401  — re-exported for anything importing it from here
    mine,
    reset_to,  # noqa: F401
)


def save_preview(out, pairs_by_task, obs, n=4):
    """A start/goal contact sheet, one row per task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in TASKS if len(pairs_by_task[t])]
    fig, axes = plt.subplots(len(tasks), 2 * n, figsize=(2 * n * 1.6, len(tasks) * 1.8))
    axes = np.atleast_2d(axes)
    for r, task in enumerate(tasks):
        for i in range(n):
            for j, name in enumerate(["start", "goal"]):
                ax = axes[r, 2 * i + j]
                ax.axis("off")
                if i < len(pairs_by_task[task]):
                    ax.imshow(obs[pairs_by_task[task][i][j]])
                    ax.set_title(f"{name}", fontsize=7)
        axes[r, 0].text(-10, 32, task, fontsize=8, ha="right", va="center")
    plt.tight_layout()
    path = str(out).replace(".npz", "_preview.png")
    plt.savefig(path, dpi=120)
    print("Saved preview:", path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="recording to mine; keep it out of training")
    p.add_argument("--out", default="ogbench_scene_single/v1_pairs.npz")
    p.add_argument("--per_task", type=int, default=50)
    p.add_argument("--min_gap", type=int, default=MIN_GAP)
    p.add_argument("--max_gap", type=int, default=MAX_GAP)
    p.add_argument("--settle", type=int, default=SETTLE, help="goal must stay valid this many steps later")
    p.add_argument("--stride", type=int, default=STRIDE, help="check every Nth frame as a start")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    d = dict(np.load(args.data))
    f = compute_features(d)
    has_images = d["observations"].ndim == 4

    out = {k: [] for k in ["task_id", "start_idx", "goal_idx", "start_qpos", "start_qvel",
                           "start_btn", "goal_qpos", "goal_btn", "start_obs", "goal_obs"]}
    pairs_by_task = {}
    for k, task in enumerate(TASKS):
        pairs = mine(d, f, task, args.min_gap, args.max_gap, args.settle, args.stride, rng)
        if len(pairs) > args.per_task:
            pairs = pairs[rng.choice(len(pairs), args.per_task, replace=False)]
        pairs_by_task[task] = pairs
        print(f"{task:18s} {len(pairs):4d} pairs"
              + ("" if len(pairs) >= args.per_task else "  (fewer than requested; use more data)"))
        for t, g in pairs:
            out["task_id"].append(k)
            out["start_idx"].append(t); out["goal_idx"].append(g)
            out["start_qpos"].append(d["qpos"][t]); out["start_qvel"].append(d["qvel"][t])
            out["start_btn"].append(d["button_states"][t])
            out["goal_qpos"].append(d["qpos"][g]); out["goal_btn"].append(d["button_states"][g])
            out["start_obs"].append(d["observations"][t]); out["goal_obs"].append(d["observations"][g])

    arrays = {k: np.array(v) for k, v in out.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, task_names=np.array(TASKS), **arrays)
    print("Saved", args.out, "| total pairs:", len(arrays["task_id"]))
    if has_images:
        save_preview(args.out, pairs_by_task, d["observations"])


if __name__ == "__main__":
    main()
