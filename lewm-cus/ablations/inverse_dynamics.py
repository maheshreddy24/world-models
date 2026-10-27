"""Inverse dynamics probes on the dynamics experiments: is the action recoverable from (predicted) latents?

    python ablations/inverse_dynamics.py                   # double pendulum, cartpole, single pendulum
    python ablations/inverse_dynamics.py --plot-only       # redraw from the saved results

Target: the action block between latent frames t and t+1 (frameskip raw actions, 5 torques),
standardised with training statistics. Loss MSE. Metric: R^2 on held-out episodes (the model's
held-out episodes + every val/test episode), overall (mean over the 5 sub-steps) and per sub-step.
Every probe is fitted at depth 0 (linear) to 5 hidden layers (512 wide, ReLU), Adam, early stopping
on 10% of the training episodes.

    A        policy only       z_{t-1}, z_t                     true latents
    B        inverse model     z_{t-1}, z_t, z_{t+1}            true latents
    C        B frozen          z_{t-1}, z_t, zh_{t+1}           zh_{t+1}: one-step prediction (= F at k = 1)
    D        retrained         z_{t-1}, z_t, zh_{t+1}           (= G at k = 1)
    E        upper bound       state_{t-1}, state_t, state_{t+1}  recorded simulator state
    sanity   as B, training targets shuffled                    should give R^2 ~ 0
    F(k)     B frozen          zh_{t-1}, zh_t, zh_{t+1}         autoregressive rollout, zh_{t+1} at horizon k
    G(k)     retrained         same
    A_roll   A frozen / retrained on zh_{t-1}, zh_t

Rollouts: `history` real context frames, then model.rollout with the recorded actions (the action
block t is an input of zh_{t+1}, as in C). Frames at horizon <= 0 are the real context. Train
windows every 2nd latent step, held-out windows every step, all frameskip phases.

Writes checkpoints/inverse_dynamics/<task>/results.json and checkpoints/inverse_dynamics/
{depth.png, rollout.png, substep.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ablations"))

from common import encode_frames, load_world_model  # noqa: E402
from config import heldout_h5  # noqa: E402
from probe_mixed import Episodes, blocks_from  # noqa: E402
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402

RUNS = {"acrobot-swingup": "checkpoints/ckpts_dynamics/double_pend_lwm/epoch_006.pt",
        "cartpole-swingup": "checkpoints/cartpole-swingup/vit/epoch_006.pt",
        "pendulum-swingup": "checkpoints/ckpts_dynamics/pendulum-swingup/vit/epoch_006.pt"}
NAMES = {"acrobot-swingup": "double pendulum", "cartpole-swingup": "cartpole", "pendulum-swingup": "single pendulum"}
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]
DEPTHS = range(6)
OUT = ROOT / "checkpoints" / "inverse_dynamics"


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(model, cfg, normalizer, reader, episodes, stride: int, horizon: int, device, workers: int) -> dict:
    """True triples (latent and state) and rollout sequences of `episodes`, every frameskip phase."""
    fs, hist = cfg.data.frameskip, cfg.data.history
    transform = ImageTransform(cfg.data.img_size)
    true = {"z": [], "s": [], "y": [], "ep": []}
    roll = {"seq": [], "y": [], "ep": []}
    for ep, frames, obs, actions in DataLoader(Episodes(reader, episodes, "observation"), batch_size=None,
                                               num_workers=workers):
        frames, obs, actions = frames.numpy(), obs.numpy(), actions.numpy()
        emb = encode_frames(model, frames, transform, device)[:, 0]  # (rows, D): one token
        for o in range(fs):
            blocks = blocks_from(actions, normalizer, o, fs).to(device)  # (n, fs * A) normalised, predictor input
            n = len(blocks)
            raw = np.nan_to_num(actions[o : o + n * fs], nan=0.0).reshape(n, -1)  # (n, fs * A) targets
            lat = emb[o :: fs][: n + 1]  # (n + 1, D)
            st = torch.from_numpy(obs[o :: fs][: n + 1])
            t = torch.arange(1, n)  # triples t-1, t, t+1 and block t
            true["z"].append(torch.stack([lat[t - 1], lat[t], lat[t + 1]], 1).half().cpu())
            true["s"].append(torch.stack([st[t - 1], st[t], st[t + 1]], 1))
            true["y"].append(torch.from_numpy(raw[1:n]).float())
            true["ep"].append(torch.full((n - 1,), int(ep)))
            starts = torch.arange(0, n - hist + 2 - horizon, stride)
            if not len(starts):
                continue
            c = starts + hist - 1  # last context frame
            for b in starts.split(256):
                cb = b + hist - 1
                ctx = b[:, None] + torch.arange(hist)
                acts = b[:, None] + torch.arange(hist + horizon - 1)
                z = model.rollout(lat[ctx][:, :, None], blocks[acts])[:, :, 0]  # (W, horizon, D)
                seq = torch.cat([lat[cb - 1][:, None], lat[cb][:, None], z], 1)  # frames c-1, c, c+1 .. c+horizon
                roll["seq"].append(seq.half().cpu())
            yidx = c[:, None] + torch.arange(horizon)  # block c + j - 1 for horizon j = 1..horizon
            roll["y"].append(torch.from_numpy(raw[yidx.numpy()]).float())
            roll["ep"].append(torch.full((len(starts),), int(ep)))
    return {"true": {k: torch.cat(v) for k, v in true.items()}, "roll": {k: torch.cat(v) for k, v in roll.items()}}


# --------------------------------------------------------------------------- #
#  Probes
# --------------------------------------------------------------------------- #
class Probe(nn.Module):
    def __init__(self, d_in: int, d_out: int, depth: int, width: int = 512):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, d_out))
        for name, n in (("x_mean", d_in), ("x_std", d_in), ("y_mean", d_out), ("y_std", d_out)):
            self.register_buffer(name, torch.zeros(n) if "mean" in name else torch.ones(n))

    def forward(self, x):  # raw input -> standardised target
        return self.net((x - self.x_mean) / self.x_std)


def fit(X, Y, Xv, Yv, depth: int, device, seed: int, max_epochs: int = 60, patience: int = 6, batch: int = 1024):
    """Adam on MSE of standardised targets, best validation epoch kept. Returns (probe, val R^2)."""
    torch.manual_seed(seed)
    X, Y, Xv, Yv = (t.to(device).float() for t in (X, Y, Xv, Yv))
    p = Probe(X.size(1), Y.size(1), depth).to(device)
    p.x_mean.copy_(X.mean(0)); p.x_std.copy_(X.std(0) + 1e-6)
    p.y_mean.copy_(Y.mean(0)); p.y_std.copy_(Y.std(0) + 1e-6)
    Ys, Yvs = (Y - p.y_mean) / p.y_std, (Yv - p.y_mean) / p.y_std
    opt = torch.optim.AdamW(p.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        p.train()
        for idx in torch.randperm(len(X), device=device).split(batch):
            loss = ((p(X[idx]) - Ys[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        p.eval()
        with torch.no_grad():
            v = ((p(Xv) - Yvs) ** 2).mean().item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in p.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    p.load_state_dict(best_state)
    return p.eval(), r2(p, Xv, Yv)["overall"]


@torch.no_grad()
def r2(p: Probe, X, Y) -> dict:
    """R^2 per output (sub-step) and their mean, in standardised target space (= raw, per output)."""
    X, Y = X.to(p.x_mean.device).float(), Y.to(p.x_mean.device).float()
    Ys = (Y - p.y_mean) / p.y_std
    P = p(X)
    per = 1 - ((P - Ys) ** 2).sum(0) / ((Ys - Ys.mean(0)) ** 2).sum(0)
    return {"overall": float(per.mean()), "substep": per.tolist()}


# --------------------------------------------------------------------------- #
#  One task
# --------------------------------------------------------------------------- #
def run_task(task: str, args, device) -> dict:
    t0 = time.time()
    ckpt = ROOT / RUNS[task]
    model, cfg, _ = load_world_model(ckpt, device)
    set_seed(cfg.seed)
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action",))
    train_eps, held_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    train_eps = np.sort(train_eps)
    val_eps = set(np.random.default_rng(cfg.seed).choice(train_eps, size=len(train_eps) // 10, replace=False).tolist())

    print(f"\n{NAMES[task]} ({task}) | {ckpt}", flush=True)
    tr = collect(model, cfg, normalizer, reader, train_eps, 2, args.horizon, device, args.workers)
    print(f"  train collected ({time.time() - t0:.0f}s)", flush=True)
    te_a = collect(model, cfg, normalizer, reader, np.sort(held_eps), 1, args.horizon, device, args.workers)
    reader.close()
    extra = H5Reader(heldout_h5(task), rdcc_mb=cfg.data.rdcc_mb)
    te_b = collect(model, cfg, normalizer, extra, np.arange(extra.num_episodes), 1, args.horizon, device, args.workers)
    extra.close()
    te = {part: {k: torch.cat([te_a[part][k], te_b[part][k]]) for k in te_a[part]} for part in ("true", "roll")}
    print(f"  held-out collected ({time.time() - t0:.0f}s): train {len(tr['true']['y'])} true / {len(tr['roll']['y'])} "
          f"rollout windows | held-out {len(te['true']['y'])} / {len(te['roll']['y'])}", flush=True)
    del model
    torch.cuda.empty_cache()

    def split(part, x, y):
        m = torch.tensor([int(e) in val_eps for e in tr[part]["ep"]])
        return x[~m], y[~m], x[m], y[m]

    def true_x(d, kind):
        z, s = d["true"]["z"].float(), d["true"]["s"].float()
        return {"A": z[:, :2].flatten(1), "B": z.flatten(1), "E": s.flatten(1)}[kind]

    def roll_x(d, k, n_frames):
        seq = d["roll"]["seq"]
        return seq[:, k + 1 - n_frames + 1 : k + 2].float().flatten(1) if n_frames == 3 else \
            seq[:, k - 1 : k + 1].float().flatten(1)

    res = {"task": task, "ckpt": str(ckpt), "ks": KS, "depths": list(DEPTHS),
           "n_train_true": len(tr["true"]["y"]), "n_train_roll": len(tr["roll"]["y"]),
           "n_test_true": len(te["true"]["y"]), "n_test_roll": len(te["roll"]["y"])}
    yt_tr, yt_te = tr["true"]["y"], te["true"]["y"]
    frozen = {}
    for depth in DEPTHS:
        d = res.setdefault(f"depth{depth}", {})
        for name in ("A", "B", "E"):
            X, Y, Xv, Yv = split("true", true_x(tr, name), yt_tr)
            p, v = fit(X, Y, Xv, Yv, depth, device, args.seed)
            d[name] = {**r2(p, true_x(te, name), yt_te), "val": v}
            if name in ("A", "B"):
                frozen[(name, depth)] = p
        X, Y, Xv, Yv = split("true", true_x(tr, "B"), yt_tr)
        g = torch.Generator().manual_seed(args.seed)
        p, v = fit(X, Y[torch.randperm(len(Y), generator=g)], Xv, Yv, depth, device, args.seed)
        d["sanity"] = {**r2(p, true_x(te, "B"), yt_te), "val": v}
        for k in KS:
            yr_tr, yr_te = tr["roll"]["y"][:, k - 1], te["roll"]["y"][:, k - 1]
            x3, x3t, x2, x2t = roll_x(tr, k, 3), roll_x(te, k, 3), roll_x(tr, k, 2), roll_x(te, k, 2)
            d[f"F_k{k}"] = r2(frozen[("B", depth)], x3t, yr_te)
            d[f"A_roll_frozen_k{k}"] = r2(frozen[("A", depth)], x2t, yr_te)
            X, Y, Xv, Yv = split("roll", x3, yr_tr)
            p, v = fit(X, Y, Xv, Yv, depth, device, args.seed)
            d[f"G_k{k}"] = {**r2(p, x3t, yr_te), "val": v}
            X, Y, Xv, Yv = split("roll", x2, yr_tr)
            p, v = fit(X, Y, Xv, Yv, depth, device, args.seed)
            d[f"A_roll_retrained_k{k}"] = {**r2(p, x2t, yr_te), "val": v}
        d["C"], d["D"] = d["F_k1"], d["G_k1"]
        print(f"  depth {depth}: A {d['A']['overall']:+.3f} | B {d['B']['overall']:+.3f} | C {d['C']['overall']:+.3f} "
              f"| D {d['D']['overall']:+.3f} | E {d['E']['overall']:+.3f} | sanity {d['sanity']['overall']:+.3f} | "
              f"F k=10/50 {d['F_k10']['overall']:+.3f}/{d['F_k50']['overall']:+.3f} | G k=10/50 "
              f"{d['G_k10']['overall']:+.3f}/{d['G_k50']['overall']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    save_json(OUT / task / "results.json", res)
    return res


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
COLORS = {"A": "#7b4fd6", "B": "#2a78d6", "C": "#eb6834", "D": "#d4a017", "E": "black", "sanity": "gray",
          "F": "#eb6834", "G": "#d4a017", "A_roll_frozen": "#7b4fd6", "A_roll_retrained": "#7b4fd6"}
LABELS = {"A": "A  policy only (true z)", "B": "B  inverse model (true z)", "C": "C  B frozen, predicted z_t+1",
          "D": "D  retrained, predicted z_t+1", "E": "E  simulator states", "sanity": "sanity (shuffled targets)",
          "F": "F  B frozen", "G": "G  retrained", "A_roll_frozen": "A frozen (policy floor)",
          "A_roll_retrained": "A retrained (policy floor)"}


def best_depth(res: dict, key: str) -> int:
    """Depth with the best validation R^2 of the trained probe `key` (A, B, E, G_k<k>, A_roll_retrained_k<k>)."""
    return max(DEPTHS, key=lambda d: res[f"depth{d}"][key]["val"])


def plots(results: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    # 1. R^2 vs probe depth
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.4 * n), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        for key in ("A", "B", "C", "D", "E", "sanity"):
            ax.plot(list(DEPTHS), [res[f"depth{d}"][key]["overall"] for d in DEPTHS], "o-", color=COLORS[key],
                    ls=":" if key == "sanity" else "-", lw=1.8, ms=4, label=LABELS[key])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set(title=NAMES[res["task"]], ylabel="held-out R²", ylim=(-0.1, 1.0), xticks=list(DEPTHS))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("hidden layers (0 = linear)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if n > 1 else 0.85))
    fig.savefig(OUT / "depth.png", dpi=140)
    plt.close(fig)

    # 2. R^2 vs rollout horizon, depth selected on validation
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.4 * n), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        dB = best_depth(res, "B")
        dA = best_depth(res, "A")
        for key, ls in (("F", "-"), ("G", "-"), ("A_roll_frozen", "--"), ("A_roll_retrained", ":")):
            vals = []
            for k in KS:
                d = {"F": dB, "A_roll_frozen": dA}.get(key) if key in ("F", "A_roll_frozen") else \
                    best_depth(res, f"{key}_k{k}")
                vals.append(res[f"depth{d}"][f"{key}_k{k}"]["overall"])
            ax.plot(KS, vals, "o", ls=ls, color=COLORS[key], lw=1.8, ms=4, label=LABELS[key])
        ax.axhline(res[f"depth{dB}"]["B"]["overall"], color=COLORS["B"], lw=1.2, ls="--", label="B  true latents")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xscale("log")
        ax.set(title=f"{NAMES[res['task']]}", ylabel="held-out R²", ylim=(-0.1, 1.0), xticks=KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("rollout horizon k of ẑ_t+1 (latent steps of 5 rows)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if n > 1 else 0.85))
    fig.savefig(OUT / "rollout.png", dpi=140)
    plt.close(fig)

    # 3. R^2 per sub-step, depth selected on validation
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.4 * n), squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        for key in ("A", "B", "C", "D", "E"):
            d = best_depth(res, {"C": "B", "D": "G_k1"}.get(key, key))
            sub = res[f"depth{d}"][key]["substep"]
            ax.plot(range(1, len(sub) + 1), sub, "o-", color=COLORS[key], lw=1.8, ms=4, label=LABELS[key])
        ax.axhline(0, color="gray", lw=0.6)
        ax.set(title=NAMES[res["task"]], ylabel="held-out R²", ylim=(-0.1, 1.0), xticks=range(1, len(sub) + 1))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("sub-step of the action block (row t+0 .. t+4)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93 if n > 1 else 0.85))
    fig.savefig(OUT / "substep.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(RUNS))
    p.add_argument("--horizon", type=int, default=max(KS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for task in args.tasks:
        path = OUT / task / "results.json"
        results.append(json.loads(path.read_text()) if args.plot_only else run_task(task, args, device))
    plots(results)
    print(f"\nsaved {OUT}/{{depth,rollout,substep}}.png")


if __name__ == "__main__":
    main()
