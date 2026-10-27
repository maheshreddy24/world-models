"""Rollout errors of the dynamics experiments on a log scale: teacher-forced, autoregressive, copy.

    python ablations/plot_errors_log.py           # double pendulum, cartpole, single pendulum

One row per experiment; columns: latent MSE, then the probe error of each quantity (deg, cm),
per horizon step, with the clean probe (<run dir>/probe/probe.pt). Copy repeats the last
context latent: its latent MSE is rollout's copy_mse, its probe error |probe(z_last) - true(t + h)|.
Teacher-forced and autoregressive come from probe_mixed.py's rollout.json (clean probe); copy's
probe error is computed here on the same windows (every full-horizon window, stride 1, of the
model's held-out episodes and every val/test episode) and cached in <run dir>/rollout/copy_probe.json.

Writes checkpoints/skill_plots/errors_log.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ablations"))

from common import LinearProbe, analysis_dir, encode_frames, load_world_model  # noqa: E402
from config import heldout_h5  # noqa: E402
from probe_mixed import NAMES, Episodes  # noqa: E402
from src.data import H5Reader, ImageTransform, split_episodes  # noqa: E402
from src.utils import save_json  # noqa: E402

RUNS = ["checkpoints/ckpts_dynamics/double_pend_lwm/epoch_006.pt",  # row order: double pendulum, cartpole, single
        "checkpoints/cartpole-swingup/vit/epoch_006.pt",
        "checkpoints/ckpts_dynamics/pendulum-swingup/vit/epoch_006.pt"]
STYLES = {"tf": ("teacher-forced", "#2a78d6", "s-"), "ar": ("autoregressive", "#eb6834", "o-"),
          "copy": ("copy last context latent", "gray", "--")}


@torch.no_grad()
def copy_probe_errors(ckpt: Path, horizon: int, device) -> dict:
    """|probe(z_last) - true(t + h)| per quantity and horizon step, on rollout.py's windows."""
    model, cfg, task = load_world_model(ckpt, device)
    probe = LinearProbe.load(analysis_dir(ckpt, "probe") / "probe.pt", device)
    transform = ImageTransform(cfg.data.img_size)
    fs, hist = cfg.data.frameskip, cfg.data.history
    total, windows = torch.zeros(horizon, len(task.quantities), device=device, dtype=torch.float64), 0
    train = H5Reader(cfg.data.h5_path)
    _, val_eps = split_episodes(train.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    extra = H5Reader(heldout_h5(task.name))
    for reader, eps in ((train, np.sort(val_eps)), (extra, np.arange(extra.num_episodes))):
        for _, frames, obs, _ in DataLoader(Episodes(reader, eps, task.state_key), batch_size=None, num_workers=4):
            n = (len(frames) - 1) // fs
            last = n + 1 - hist - horizon
            if last < 0:
                continue
            emb = encode_frames(model, frames.numpy()[: n * fs + 1 : fs], transform, device)
            truth = task.targets(obs[: n * fs + 1 : fs]).to(device)
            read = probe(emb)  # (n+1, K)
            s = torch.arange(0, last + 1)
            t0 = s + hist - 1
            tgt = t0[:, None] + torch.arange(1, horizon + 1)
            total += task.errors(read[t0][:, None].expand(-1, horizon, -1), truth[tgt]).sum(0)
            windows += len(s)
    train.close()
    extra.close()
    return {"task": task.name, "windows": windows,
            **{f"{q.name}_copy_probe": (total[:, k] / windows).tolist() for k, q in enumerate(task.quantities)}}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ckpt in [ROOT / r for r in RUNS]:
        res = json.loads((analysis_dir(ckpt, "probe_mixed") / "rollout.json").read_text())
        cache = analysis_dir(ckpt, "rollout") / "copy_probe.json"
        copy = json.loads(cache.read_text()) if cache.exists() else None
        if copy is None or copy["windows"] != res["windows"]:
            copy = copy_probe_errors(ckpt, len(res["horizon"]), device)
            save_json(cache, copy)
        if copy["windows"] != res["windows"]:
            print(f"warning: {ckpt}: {copy['windows']} copy windows, {res['windows']} rollout windows")
        rows.append((res, copy))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.tasks import get_task

    cols = 1 + max(len(get_task(r["task"]).quantities) for r, _ in rows)
    fig, axes = plt.subplots(len(rows), cols, figsize=(4.8 * cols, 3.8 * len(rows)), squeeze=False)
    for row, (res, copy) in zip(axes, rows):
        task = get_task(res["task"])
        h = np.array(res["horizon"])
        series = [("latent MSE", {"tf": res["tf_mse"], "ar": res["ar_mse"], "copy": res["copy_mse"]})]
        series += [(f"{q.name} probe error ({q.unit})",
                    {"tf": res[f"{q.name}_clean_tf"], "ar": res[f"{q.name}_clean_ar"],
                     "copy": copy[f"{q.name}_copy_probe"]}) for q in task.quantities]
        for ax, (title, curves) in zip(row, series):
            for c, (lab, color, style) in STYLES.items():
                ax.plot(h, curves[c], style, color=color, ms=3 if c != "copy" else 0, lw=1.6, label=lab)
            ax.set_yscale("log")
            ax.set(title=title, xlabel="horizon (latent steps of 5 rows)")
            ax.grid(True, which="both", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].set_ylabel(f"{NAMES.get(task.name, task.name)}\n\nerror (log)")
        for ax in row[len(series):]:
            ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / "checkpoints" / "skill_plots" / "errors_log.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
