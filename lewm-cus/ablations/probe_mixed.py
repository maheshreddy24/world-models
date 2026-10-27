"""A probe that has also seen predicted latents, against the clean one, on the dynamics experiments.

    python ablations/probe_mixed.py                  # double pendulum, single pendulum, cartpole
    python ablations/probe_mixed.py --runs checkpoints/cartpole-swingup/vit --pred-frac 0.5

probe.py fits a ridge probe on clean latents enc(x) only. Predicted latents z_hat drift off
that distribution, so part of the rollout error may be the probe misreading them. The mixed
probe is the same ridge (same features, strength, training episodes and clean rows: every
--row-stride-th row) plus autoregressive predictions, so that --pred-frac of all its samples
are z_hat. A predicted sample is a random training window (random frameskip phase, random
start) rolled out with the recorded actions to a horizon drawn uniformly from 1..--max-h,
labelled with the true state of the frame it predicts.

Then both probes score the same rollouts, the windows of rollout.py: every full-horizon
window (stride 1, frameskip phase 0) of the model's held-out episodes and every val/test
episode, teacher-forced and autoregressive. Skill per horizon step h:

    latent    1 - MSE_pred / MSE_copy,  copy = the last context latent repeated
    probe     1 - err_pred / err_copy,  copy = |state(t + h) - state(t)|, the true state held

Writes, per run, <run dir>/probe_mixed/:
    probe.pt, probe.json      the mixed probe; held-out error of both probes on clean and on
                              predicted latents
    rollout.json              per horizon: latent MSE, probe errors of both probes, copy errors
    skill.png                 skill with the mixed probe (as probe_skill.py's figure)
    skill_compare.png         clean probe vs mixed probe
and the stacked figures checkpoints/skill_plots/skill_all_mixed.png, skill_compare_all.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ablations"))

from common import LinearProbe, analysis_dir, encode_frames, load_world_model, probe_features  # noqa: E402
from config import heldout_h5  # noqa: E402
from rollout import teacher_forced  # noqa: E402
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes  # noqa: E402
from src.models import latent_sq_dist  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402

RUNS = ["checkpoints/ckpts_dynamics/double_pend_lwm/epoch_006.pt",
        "checkpoints/ckpts_dynamics/pendulum-swingup/vit/epoch_006.pt",
        "checkpoints/cartpole-swingup/vit/epoch_006.pt"]
NAMES = {"acrobot-swingup": "double pendulum (acrobot-swingup)", "pendulum-swingup": "single pendulum (pendulum-swingup)",
         "cartpole-swingup": "cartpole-swingup"}
PROBES = ("clean", "mixed")
PREDS = ("tf", "ar")


class Episodes(Dataset):
    """Whole episodes: frames (rows, H, W, 3) uint8, state, raw actions."""

    def __init__(self, reader: H5Reader, episodes, state_key: str):
        self.reader, self.episodes, self.state_key = reader, np.asarray(episodes), state_key

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        a, n = int(self.reader.ep_offset[ep]), int(self.reader.ep_len[ep])
        return (ep, self.reader.span("pixels", a, a + n), self.reader.span(self.state_key, a, a + n).astype(np.float32),
                self.reader.span("action", a, a + n))


def blocks_from(actions: np.ndarray, normalizer, phase: int, fs: int) -> torch.Tensor:
    """(n, fs * A) normalised action blocks starting at row `phase`."""
    n = (len(actions) - 1 - phase) // fs
    act = normalizer.normalize("action", np.nan_to_num(actions[phase : phase + n * fs], nan=0.0)).astype(np.float32)
    return torch.from_numpy(act.reshape(n, -1))


@torch.no_grad()
def collect_train(model, task, reader, episodes, normalizer, transform, cfg, args, device, rng):
    """Clean features/targets (every row_stride-th row) and predicted ones (to make pred_frac of the total)."""
    fs, hist, H = cfg.data.frameskip, cfg.data.history, args.max_h
    loader = DataLoader(Episodes(reader, episodes, task.state_key), batch_size=None, num_workers=args.workers)
    rows = [int(reader.ep_len[int(e)]) for e in episodes]
    n_clean = sum(len(range(0, r, args.row_stride)) for r in rows)
    n_pred = int(round(args.pred_frac / (1 - args.pred_frac) * n_clean))
    per_ep = np.full(len(episodes), n_pred // len(episodes))
    per_ep[: n_pred % len(episodes)] += 1
    Xc, Yc, Xp, Yp, hs = [], [], [], [], []
    for k, (ep, frames, obs, actions) in enumerate(loader):
        frames, obs, actions = frames.numpy(), obs.numpy(), actions.numpy()
        emb = encode_frames(model, frames, transform, device)  # (rows, P, D), every row
        targets = task.targets(torch.from_numpy(obs)).to(device)
        Xc.append(probe_features(emb[:: args.row_stride], args.grid).cpu())
        Yc.append(targets[:: args.row_stride].cpu())
        # predicted samples, grouped by frameskip phase
        phases = rng.integers(0, fs, size=per_ep[k])
        for ph in np.unique(phases):
            m = int((phases == ph).sum())
            blocks = blocks_from(actions, normalizer, int(ph), fs).to(device)
            lat = emb[int(ph) :: fs][: len(blocks) + 1]  # latent frames of this phase
            last = len(blocks) + 1 - hist - H  # last start with a full max_h rollout
            if last < 0:
                continue
            s = torch.as_tensor(rng.integers(0, last + 1, size=m))
            h = torch.as_tensor(rng.integers(1, H + 1, size=m))
            ctx = s[:, None] + torch.arange(hist)
            acts = s[:, None] + torch.arange(hist + H - 1)
            z = model.rollout(lat[ctx], blocks[acts])  # (m, H, P, D)
            z = z[torch.arange(m), h - 1]  # (m, P, D)
            frame = int(ph) + (s + hist - 1 + h) * fs  # row each prediction is of
            Xp.append(probe_features(z, args.grid).cpu())
            Yp.append(targets[frame.to(device)].cpu())
            hs.append(h)
        if (k + 1) % 25 == 0:
            print(f"  train episodes {k + 1}/{len(episodes)}", flush=True)
    return torch.cat(Xc), torch.cat(Yc), torch.cat(Xp), torch.cat(Yp), torch.cat(hs)


@torch.no_grad()
def evaluate(model, task, probes, reader, episodes, normalizer, transform, cfg, args, device, sums, windows):
    """Add every full-horizon window of `episodes` (phase 0) to `sums`; returns the window count added."""
    fs, hist, H = cfg.data.frameskip, cfg.data.history, args.horizon
    loader = DataLoader(Episodes(reader, episodes, task.state_key), batch_size=None, num_workers=args.workers)
    for ep, frames, obs, actions in loader:
        frames, obs, actions = frames.numpy(), obs.numpy(), actions.numpy()
        blocks = blocks_from(actions, normalizer, 0, fs).to(device)
        n = len(blocks)
        emb = encode_frames(model, frames[: n * fs + 1 : fs], transform, device)  # (n+1, P, D)
        truth = task.targets(torch.from_numpy(obs[: n * fs + 1 : fs])).to(device)
        last = n + 1 - hist - H
        if last < 0:
            continue
        tf_all = teacher_forced(model, emb, blocks, hist, args.window_batch)
        for s in torch.arange(0, last + 1).split(args.window_batch):
            ctx = s[:, None] + torch.arange(hist)
            acts = s[:, None] + torch.arange(hist + H - 1)
            tgt = s[:, None] + hist + torch.arange(H)
            pred = {"tf": tf_all[tgt], "ar": model.rollout(emb[ctx], blocks[acts])}
            target = emb[tgt]
            sums["copy_mse"] += latent_sq_dist(emb[ctx[:, -1:]].expand_as(target), target).sum(0)
            for c, z in pred.items():
                sums[f"{c}_mse"] += latent_sq_dist(z, target).sum(0)
                for name, probe in probes.items():
                    sums[f"{name}_{c}"] += task.errors(probe(z), truth[tgt]).sum(0)
            for name, probe in probes.items():
                sums[f"{name}_floor"] += task.errors(probe(target), truth[tgt]).sum(0)
            held = truth[ctx[:, -1]][:, None].expand(-1, H, -1)
            sums["copy_state"] += task.errors(held, truth[tgt]).sum(0)
            windows += len(s)
    return windows


def score(probe, task, X, Y, device) -> dict:
    err = task.errors(probe.from_features(X.to(device)).cpu(), Y)
    return {f"{q.name}_mae_{q.unit}": float(err[:, k].mean()) for k, q in enumerate(task.quantities)}


def run(ckpt: Path, args, device) -> dict:
    model, cfg, task = load_world_model(ckpt, device)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    transform = ImageTransform(cfg.data.img_size)
    out = analysis_dir(ckpt, "probe_mixed")
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action",))
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    clean = LinearProbe.load(analysis_dir(ckpt, "probe") / "probe.pt", device)

    print(f"\n{task.name} | {ckpt}\n  fitting the mixed probe ({args.pred_frac:.0%} predicted)", flush=True)
    Xc, Yc, Xp, Yp, hs = collect_train(model, task, reader, np.sort(train_eps), normalizer, transform, cfg, args,
                                       device, rng)
    mixed = LinearProbe.fit(torch.cat([Xc, Xp]), torch.cat([Yc, Yp]), args.ridge, args.grid, device)
    mixed.save(out / "probe.pt", ckpt=str(ckpt), task=task.name, pred_frac=args.pred_frac, max_h=args.max_h)

    probes = {"clean": clean, "mixed": mixed}
    H, Q = args.horizon, len(task.quantities)
    sums = {k: torch.zeros(H, device=device, dtype=torch.float64) for k in ("copy_mse", "tf_mse", "ar_mse")}
    sums |= {f"{p}_{c}": torch.zeros(H, Q, device=device, dtype=torch.float64) for p in PROBES for c in (*PREDS, "floor")}
    sums["copy_state"] = torch.zeros(H, Q, device=device, dtype=torch.float64)
    print("  rollouts on the held-out episodes", flush=True)
    windows = evaluate(model, task, probes, reader, np.sort(val_eps), normalizer, transform, cfg, args, device, sums, 0)
    reader.close()
    extra = H5Reader(heldout_h5(task.name), rdcc_mb=cfg.data.rdcc_mb)
    print("  rollouts on the val/test episodes", flush=True)
    windows = evaluate(model, task, probes, extra, np.arange(extra.num_episodes), normalizer, transform, cfg, args,
                       device, sums, windows)
    extra.close()

    res = {"ckpt": str(ckpt), "task": task.name, "windows": windows, "horizon": list(range(1, H + 1)),
           "units": {q.name: q.unit for q in task.quantities}}
    for k in ("copy_mse", "tf_mse", "ar_mse"):
        res[k] = (sums[k] / windows).tolist()
    for c in PREDS:
        res[f"latent_{c}_skill"] = (1 - sums[f"{c}_mse"] / sums["copy_mse"]).tolist()
    for i, q in enumerate(task.quantities):
        copy = sums["copy_state"][:, i]
        res[f"{q.name}_copy"] = (copy / windows).tolist()
        for p in PROBES:
            res[f"{q.name}_{p}_floor"] = (sums[f"{p}_floor"][:, i] / windows).tolist()
            for c in PREDS:
                res[f"{q.name}_{p}_{c}"] = (sums[f"{p}_{c}"][:, i] / windows).tolist()
                res[f"{q.name}_{p}_{c}_skill"] = (1 - sums[f"{p}_{c}"][:, i] / copy).tolist()
    save_json(out / "rollout.json", res)

    info = {"ckpt": str(ckpt), "task": task.name, "pred_frac": args.pred_frac, "max_h": args.max_h,
            "ridge": args.ridge, "row_stride": args.row_stride, "train_clean": len(Xc), "train_pred": len(Xp),
            "train_pred_horizon_mean": float(hs.float().mean()),
            "train_fit": {p: {"clean": score(pr, task, Xc, Yc, device), "pred": score(pr, task, Xp, Yp, device)}
                          for p, pr in probes.items()},
            "heldout_floor_mean": {p: {q.name: float(np.mean(res[f"{q.name}_{p}_floor"])) for q in task.quantities}
                                   for p in PROBES}}
    save_json(out / "probe.json", info)
    plot_skill(res, task, f"{NAMES.get(task.name, task.name)}: skill vs copy, mixed probe "
                          f"({args.pred_frac:.0%} predicted)", out / "skill.png")
    plot_compare(res, task, f"{NAMES.get(task.name, task.name)}: clean probe vs mixed probe "
                            f"({args.pred_frac:.0%} predicted)", out / "skill_compare.png")

    show = [x for x in (1, 5, 10, 20, 50) if x <= H]
    print(f"  {len(Xc)} clean + {len(Xp)} predicted training samples | {windows} rollout windows")
    for q in task.quantities:
        for p in PROBES:
            print(f"  {q.name:9s} {p:5s} probe: floor {np.mean(res[f'{q.name}_{p}_floor']):6.2f} {q.unit} | ar skill "
                  f"h={'/'.join(map(str, show))}: " + "/".join(f"{res[f'{q.name}_{p}_ar_skill'][x - 1]:+.2f}" for x in show)
                  + " | tf: " + "/".join(f"{res[f'{q.name}_{p}_tf_skill'][x - 1]:+.2f}" for x in show))
    return {"out": out, "task": task.name}


def _style(ax, title):
    ax.axhline(0, color="gray", lw=1.5, ls="--", label="copy" if "latent" in title else "copy: true state held")
    lo = min(-0.25, np.floor(4 * min(min(line.get_ydata()) for line in ax.get_lines())) / 4)
    ax.set(title=title, ylabel="skill = 1 - err / err_copy", ylim=(lo, 1.0), xlabel="horizon (latent steps of 5 rows)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def plot_skill(res: dict, task, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(res["horizon"])
    fig, axes = plt.subplots(1, 1 + len(task.quantities), figsize=(max(5 * (1 + len(task.quantities)), 10), 4.2), squeeze=False)
    ax = axes[0, 0]
    ax.plot(h, res["latent_tf_skill"], "s-", color="#2a78d6", ms=3, label="teacher-forced")
    ax.plot(h, res["latent_ar_skill"], "o-", color="#eb6834", ms=3, label="autoregressive")
    _style(ax, "latent MSE")
    for ax, q in zip(axes[0, 1:], task.quantities):
        ax.plot(h, res[f"{q.name}_mixed_tf_skill"], "s-", color="#2a78d6", ms=3, label="teacher-forced")
        ax.plot(h, res[f"{q.name}_mixed_ar_skill"], "o-", color="#eb6834", ms=3, label="autoregressive")
        _style(ax, f"{q.name} probe error ({q.unit}), mixed probe")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_compare(res: dict, task, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(res["horizon"])
    nq = len(task.quantities)
    fig, axes = plt.subplots(1, nq, figsize=(max(5.5 * nq, 9), 4.2), squeeze=False)
    for ax, q in zip(axes[0], task.quantities):
        for c, color, mk, lab in (("tf", "#2a78d6", "s", "teacher-forced"), ("ar", "#eb6834", "o", "autoregressive")):
            ax.plot(h, res[f"{q.name}_clean_{c}_skill"], f"{mk}-", color=color, ms=3, label=f"{lab}, clean probe")
            ax.plot(h, res[f"{q.name}_mixed_{c}_skill"], f"{mk}--", color=color, ms=3, mfc="white",
                    label=f"{lab}, mixed probe")
        _style(ax, f"{q.name} probe error ({q.unit})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stack(paths: list[Path], out: Path) -> None:
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in ims)
    canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        canvas.paste(i, ((w - i.width) // 2, y))
        y += i.height
    canvas.save(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, default=[ROOT / r for r in RUNS], help="checkpoints")
    p.add_argument("--pred-frac", type=float, default=0.3, help="share of the mixed probe's samples that are z_hat")
    p.add_argument("--max-h", type=int, default=50, help="predicted samples' horizon is uniform in 1..max-h")
    p.add_argument("--horizon", type=int, default=50, help="evaluation rollout horizon (latent steps)")
    p.add_argument("--row-stride", type=int, default=2, help="clean rows: every k-th training row (as probe.py)")
    p.add_argument("--ridge", type=float, default=1e-3, help="as probe.py")
    p.add_argument("--grid", type=int, default=4, help="as probe.py")
    p.add_argument("--window-batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = [run(ckpt, args, device) for ckpt in args.runs]
    gallery = ROOT / "checkpoints" / "skill_plots"
    gallery.mkdir(parents=True, exist_ok=True)
    stack([d["out"] / "skill.png" for d in done], gallery / "skill_all_mixed.png")
    stack([d["out"] / "skill_compare.png" for d in done], gallery / "skill_compare_all.png")
    print(f"\nsaved <run dir>/probe_mixed/ and {gallery}/skill_all_mixed.png, skill_compare_all.png")


if __name__ == "__main__":
    main()
