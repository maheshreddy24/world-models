"""One figure from several rollout.py results: a row per run, the same conditions everywhere.

    python ablations/compare_rollouts.py checkpoints/double_pend_lwm checkpoints/cartpole-swingup/vit \\
        checkpoints/pendulum-swingup/vit --out ../rollout_comparison.png

Columns: latent MSE (copy / teacher-forced / autoregressive / shuffled), then
the probe error of each of the task's quantities with the probe on the real
frame as a floor. Reads <run>/rollout/rollout.json; the task comes from the
json or the run's config.json. Older rollout.json files (`latent_mse`,
`<q>_deg`) are read too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks import get_task  # noqa: E402

# categorical slots 1-3 of the reference palette, plus recessive references
COLORS = {"ar": "#2a78d6", "tf": "#eb6834", "shuf": "#1baf7a"}
MARKERS = {"ar": "o", "tf": "s", "shuf": "^"}
LABELS = {"ar": "autoregressive", "tf": "teacher-forced", "shuf": "shuffled actions"}
REFERENCE = "#898781"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def load(run: Path) -> tuple[dict, object]:
    res = json.loads((run / "rollout" / "rollout.json").read_text())
    name = res.get("task")
    if name is None:
        name = json.loads((run / "config.json").read_text())["data"].get("task", "acrobot-swingup")
    task = get_task(name)
    # older files: latent_mse -> ar_mse, <q>_deg -> <q>_ar
    if "ar_mse" not in res:
        res["ar_mse"] = res["latent_mse"]
        for q in task.quantities:
            res[f"{q.name}_ar"] = res[f"{q.name}_deg"]
    return res, task


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run directories holding rollout/rollout.json")
    p.add_argument("--out", type=Path, default=Path("rollout_comparison.png"))
    args = p.parse_args()

    loaded = [load(run) for run in args.runs]
    cols = 1 + max(len(task.quantities) for _, task in loaded)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(len(loaded), cols, figsize=(4.6 * cols, 3.2 * len(loaded)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    def line(ax, h, y, cond):
        ax.plot(h, y, color=COLORS[cond], lw=2, marker=MARKERS[cond], ms=4, markevery=5, label=LABELS[cond])

    for row, (res, task) in zip(axes, loaded):
        h = res["horizon"]
        ax = row[0]
        ax.plot(h, res["copy_mse"], color=REFERENCE, lw=1.5, ls="--", label="copy last context")
        for c in ("ar", "tf", "shuf"):
            line(ax, h, res[f"{c}_mse"], c)
        ax.set_ylabel(f"{task.name}\n\nlatent MSE", color=INK)
        ax.set_title("latent error", loc="left", color=INK)

        for ax, q in zip(row[1:], task.quantities):
            ax.plot(h, res[f"{q.name}_floor"], color=REFERENCE, lw=1.5, ls=":", label="probe on the real frame")
            for c in ("ar", "tf", "shuf"):
                line(ax, h, res[f"{q.name}_{c}"], c)
            ax.set_ylabel(f"abs error ({q.unit})")
            ax.set_title(f"{q.name} (probe)", loc="left", color=INK)
        for ax in row[1 + len(task.quantities):]:
            ax.axis("off")

    for ax in axes.flat:
        if not ax.axison:
            continue
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("horizon (latent steps of 5 rows)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    floor = axes[0][1].get_legend_handles_labels()
    handles.append(floor[0][0])
    labels.append(floor[1][0])
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Rollouts with the recorded actions, 66 unseen episodes per task (ViT LeWM, epoch 6)",
                 y=1.035, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
