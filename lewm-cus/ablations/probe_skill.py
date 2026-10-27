"""Skill of rollout.py's predictions against copying the last context frame.

    python ablations/probe_skill.py              # cartpole vit, pendulum vit, double pendulum (acrobot)
    python ablations/probe_skill.py --runs checkpoints/cartpole-swingup/vit

For a run's rollout.json, per horizon step h, for teacher-forced and autoregressive predictions:

    skill(h) = 1 - err_pred(h) / err_copy(h)

    latent panel     err = latent MSE; err_copy = the last context latent repeated (rollout.py's copy_mse)
    probe panels     err = probe error (deg, cm); err_copy = |state(t + h) - state(t)|, the true state
                     of the last context frame held, on the same windows as rollout.py (every
                     full-horizon window, stride 1, of the model's 26 held-out episodes and every
                     val/test episode), from the recorded state alone

Copy is the reference: its skill is 0 by definition. 1 = perfect, < 0 = worse than copying.

Writes <run>/rollout/skill.{png,json} and a copy of every figure in checkpoints/skill_plots/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import heldout_h5, task_h5  # noqa: E402
from src.data import H5Reader, split_episodes  # noqa: E402
from src.tasks import get_task  # noqa: E402
from src.utils import save_json  # noqa: E402

PREDICTIONS = {"tf": ("teacher-forced", "#2a78d6", "s-"), "ar": ("autoregressive", "#eb6834", "o-")}
SEED, HELDOUT = 3072, 26  # the world models' held-out split (config: seed, data.val_episodes)
RUNS = ["checkpoints/double_pend_lwm", "checkpoints/pendulum-swingup/vit", "checkpoints/cartpole-swingup/vit"]
NAMES = {"acrobot-swingup": "double pendulum (acrobot-swingup)", "pendulum-swingup": "single pendulum (pendulum-swingup)",
         "cartpole-swingup": "cartpole-swingup"}


def load_rollout(run: Path) -> tuple[dict, str]:
    """rollout.json with the current key names, and its task (older runs store neither the same way)."""
    res = json.loads((run / "rollout" / "rollout.json").read_text())
    task = res.get("task")
    if task is None:
        cfg = json.loads((run / "config.json").read_text())
        task = Path(cfg["data"]["h5_path"]).stem
    if "ar_mse" not in res and "latent_mse" in res:
        res["ar_mse"] = res["latent_mse"]
    for q in get_task(task).quantities:
        if f"{q.name}_ar" not in res and f"{q.name}_deg" in res:
            res[f"{q.name}_ar"] = res[f"{q.name}_deg"]
    return res, task


def copy_errors(task, frameskip: int, history: int, horizon: int) -> tuple[np.ndarray, int]:
    """Mean |state(t + h) - state(t)| per quantity over rollout.py's windows: (horizon, Q), windows."""
    total, windows = np.zeros((horizon, len(task.quantities))), 0
    train = H5Reader(task_h5(task.name))
    _, val_eps = split_episodes(train.num_episodes, HELDOUT, SEED)
    valtest = H5Reader(heldout_h5(task.name))
    for reader, eps in ((train, np.sort(val_eps)), (valtest, np.arange(valtest.num_episodes))):
        for ep in eps:
            off, rows = int(reader.ep_offset[ep]), int(reader.ep_len[ep])
            n = (rows - 1) // frameskip
            obs = torch.from_numpy(reader.span(task.state_key, off, off + n * frameskip + 1)[::frameskip].copy())
            t = task.targets(obs.float())  # (n + 1, K)
            starts = np.arange(0, n + 1 - history - horizon + 1)
            if not len(starts):
                continue
            t0 = torch.as_tensor(starts + history - 1)
            future = t0[:, None] + torch.arange(1, horizon + 1)
            total += task.errors(t[t0][:, None].expand(-1, horizon, -1), t[future]).sum(0).numpy()
            windows += len(starts)
    train.close()
    valtest.close()
    return total / windows, windows


def skills(res: dict, task, copy: np.ndarray) -> dict:
    out = {"horizon": res["horizon"]}
    for c in PREDICTIONS:
        out[f"latent_{c}_skill"] = (1 - np.asarray(res[f"{c}_mse"]) / np.asarray(res["copy_mse"])).tolist()
    for k, q in enumerate(task.quantities):
        out[f"{q.name}_copy_err"] = copy[:, k].tolist()
        for c in PREDICTIONS:
            out[f"{q.name}_{c}_skill"] = (1 - np.asarray(res[f"{q.name}_{c}"]) / copy[:, k]).tolist()
    return out


def plot(skill: dict, task, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(skill["horizon"])
    panels = [("latent", "latent MSE", "copy: last context latent repeated")] + [
        (q.name, f"{q.name} probe error ({q.unit})", "copy: true state held") for q in task.quantities]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), squeeze=False)
    for ax, (key, what, copy_label) in zip(axes[0], panels):
        for c, (lab, color, style) in PREDICTIONS.items():
            ax.plot(h, skill[f"{key}_{c}_skill"], style, color=color, ms=3, label=lab)
        ax.axhline(0, color="gray", lw=1.5, ls="--", label=copy_label)
        lo = min(-0.25, np.floor(4 * min(min(skill[f"{key}_{c}_skill"]) for c in PREDICTIONS)) / 4)
        ax.set(title=f"{what}", ylabel="skill = 1 - err / err_copy", ylim=(lo, 1.0),
               xlabel="horizon (latent steps of 5 rows)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, default=[ROOT / r for r in RUNS])
    args = p.parse_args()

    gallery = ROOT / "checkpoints" / "skill_plots"
    gallery.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        res, name = load_rollout(run)
        task = get_task(name)
        copy, windows = copy_errors(task, res.get("frameskip", 5), res.get("history", 3), len(res["horizon"]))
        if res.get("windows") not in (None, windows):
            print(f"warning: {run.name}: {res['windows']} windows in rollout.json, {windows} for the copy error")
        skill = skills(res, task, copy)
        out = run / "rollout"
        save_json(out / "skill.json", {"task": name, "copy_windows": windows, **skill})
        plot(skill, task, f"{NAMES.get(name, name)}: skill vs copy ({run.name})", out / "skill.png")
        shutil.copy(out / "skill.png", gallery / f"skill_{name}_{run.name}.png")
        show = [x for x in (1, 5, 10, 20, 50) if x <= len(skill["horizon"])]
        print(f"{NAMES.get(name, name)} [{run.name}], skill at h={'/'.join(map(str, show))}")
        for key in ["latent", *(q.name for q in task.quantities)]:
            print(f"  {key:9s} " + " | ".join(f"{c} " + "/".join(f"{skill[f'{key}_{c}_skill'][x - 1]:+.2f}" for x in show)
                                          for c in PREDICTIONS))
    print(f"saved <run>/rollout/skill.{{png,json}}, copies in {gallery}")


if __name__ == "__main__":
    main()
