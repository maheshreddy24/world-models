"""BallCatch probe: can the ball and basket state be read off the frozen latent?

    python ablations/probe_ball.py --ckpt checkpoints/exp_1790681067/epoch_005.pt
    python ablations/probe_ball.py --ckpt checkpoints/exp_1790681067/epoch_005.pt --probe mlp
    python ablations/probe_ball.py --ckpt checkpoints/exp_1790681067/epoch_005.pt --random-init   # untrained-encoder floor

The ball counterpart of `probe_cube.py`. The world model is frozen; every frame
of a sample of episodes is encoded once, and a probe is fitted from the latent
to the 6-d oracle state (`observation`: ball x, y, vx, vy, basket x, basket vx)
on training episodes and scored on the world model's own held-out episodes.

Two inputs are probed:

    single   e_t                 positions should be readable from one frame;
                                 velocities mostly are not (one frame shows no motion)
    pair     [e_t, e_t+k]        two frames `frameskip` steps apart, as the predictor
                                 sees them: this is where velocity has to show up

Scores are also reported on in-flight frames only (before the ball first
lands), since that is the part of the episode a catch depends on and where the
ball moves fastest. Compare against `--random-init` (same architecture,
untrained weights): a random ViT already keeps some position information, so
the trained encoder has to beat that, not just the mean guess.

`--probe linear` is closed-form ridge regression — the question the planner's
L2 cost cares about, whether the state is stored linearly. `--probe mlp` asks
whether it is there at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
from src.utils import load_model, save_json, set_seed

NAMES = ("ball_x", "ball_y", "ball_vx", "ball_vy", "basket_x", "basket_vx")
UNITS = ("m", "m", "m/s", "m/s", "m", "m/s")


class EpisodeFrames(Dataset):
    """One item per episode: all its observations, oracle states and the landed flag."""

    def __init__(self, reader, episodes, obs_key, transform, normalizer, state_key=None):
        self.reader, self.episodes = reader, np.asarray(episodes)
        self.obs_key, self.transform, self.normalizer, self.state_key = obs_key, transform, normalizer, state_key

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        start = int(self.reader.ep_offset[ep])
        stop = start + int(self.reader.ep_len[ep])
        obs = self.reader.span(self.obs_key, start, stop)
        if self.obs_key == "pixels":
            obs = self.transform(obs)
        else:
            obs = torch.from_numpy(self.normalizer.normalize(self.obs_key, obs).astype(np.float32))
        target = torch.from_numpy(self.reader.span("observation", start, stop).astype(np.float32))
        landed = torch.from_numpy(self.reader.span("landed", start, stop).astype(bool))
        state = torch.zeros(0)
        if self.state_key is not None:
            raw = self.reader.span(self.state_key, start, stop)
            state = torch.from_numpy(self.normalizer.normalize(self.state_key, raw).astype(np.float32))
        return obs, state, target, landed


@torch.no_grad()
def encode_split(model, dataset, device, workers: int, gap: int, desc: str) -> dict:
    """Encode every frame; build single-frame and frame-pair features.

    Returns:
        {"single": (X, Y, in_flight), "pair": (X, Y, in_flight)}. A pair's
        target is the state at its second frame, the "current" one.
    """
    loader = DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda")
    out = {"single": ([], [], []), "pair": ([], [], [])}
    for obs, state, target, landed in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        state = state.to(device, non_blocking=True)[:, None] if state.numel() else None
        emb = model.encode(obs.to(device, non_blocking=True)[:, None], state)[:, 0]  # (T, P, D)
        if model.oracle_encoder is not None:
            emb = emb[:, :-1]  # the oracle token *is* the answer; probe the pixels
        feat = emb.mean(1).float().cpu()  # (T, D); pooled encoders have P = 1
        flying = ~landed
        for key, x, y, f in (
            ("single", feat, target, flying),
            ("pair", torch.cat([feat[:-gap], feat[gap:]], -1), target[gap:], flying[gap:]),
        ):
            out[key][0].append(x)
            out[key][1].append(y)
            out[key][2].append(f)
    return {k: tuple(torch.cat(v) for v in vals) for k, vals in out.items()}


class Probe(nn.Module):
    """Features -> 6-d state, with input/output standardisation built in."""

    def __init__(self, dim: int, out: int, kind: str = "linear", hidden: int = 512):
        super().__init__()
        self.kind = kind
        self.net = nn.Linear(dim, out) if kind == "linear" else nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out))
        for name, n in (("x_mean", dim), ("x_std", dim), ("y_mean", out), ("y_std", out)):
            self.register_buffer(name, torch.zeros(n) if "mean" in name else torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean


def fit(X: torch.Tensor, Y: torch.Tensor, kind: str, args, device) -> Probe:
    probe = Probe(X.size(1), Y.size(1), kind).to(device)
    X, Y = X.to(device), Y.to(device)
    probe.x_mean.copy_(X.mean(0)); probe.x_std.copy_(X.std(0) + 1e-6)
    probe.y_mean.copy_(Y.mean(0)); probe.y_std.copy_(Y.std(0) + 1e-6)
    Xn, Yn = (X - probe.x_mean) / probe.x_std, (Y - probe.y_mean) / probe.y_std

    if kind == "linear":
        # Closed-form ridge: exact, no optimiser settings to get wrong.
        A = Xn.T @ Xn + args.ridge * len(Xn) * torch.eye(Xn.size(1), device=device)
        W = torch.linalg.solve(A, Xn.T @ Yn)
        probe.net.weight.data.copy_(W.T)
        probe.net.bias.data.zero_()
        return probe.eval()

    opt = torch.optim.AdamW(probe.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    probe.train()
    for _ in tqdm(range(args.steps), desc="fit mlp", dynamic_ncols=True, disable=not sys.stderr.isatty()):
        idx = torch.randint(len(Xn), (args.batch_size,), device=device)
        loss = F.mse_loss(probe.net(Xn[idx]), Yn[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return probe.eval()


def metrics(pred: torch.Tensor, true: torch.Tensor, train_mean: torch.Tensor) -> dict:
    """Per dimension: R2 (1 = perfect, 0 = no better than the mean), MAE, and the mean-guess MAE."""
    r2 = 1 - (pred - true).pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0).clamp_min(1e-12)
    mae = (pred - true).abs().mean(0)
    guess = (train_mean - true).abs().mean(0)
    out = {"num_frames": len(true)}
    for j, name in enumerate(NAMES):
        out[name] = {"r2": r2[j].item(), "mae": mae[j].item(), "mean_guess_mae": guess[j].item()}
    ball = (pred[:, :2] - true[:, :2]).norm(dim=-1) * 100
    out["ball_pos_err_cm"] = {"mean": ball.mean().item(), "median": ball.median().item(),
                              "within_ball_radius": (ball < 12).float().mean().item()}
    return out


def plot(pred: torch.Tensor, true: torch.Tensor, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
    idx = torch.randperm(len(true))[:5000]
    for j, ax in enumerate(axes):
        t, p = true[idx, j].numpy(), pred[idx, j].numpy()
        ax.scatter(t, p, s=2, alpha=0.25)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(title=NAMES[j], xlabel=f"true ({UNITS[j]})", ylabel="probe")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Probe the frozen latent for the BallCatch state.")
    parser.add_argument("--ckpt", required=True, help="world-model checkpoint (epoch_XXX.pt)")
    parser.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--random-init", action="store_true", help="same architecture, untrained weights")
    parser.add_argument("--train-episodes", type=int, default=2000, help="training episodes to fit on (61 frames each)")
    parser.add_argument("--val-episodes", type=int, default=500, help="held-out episodes to score on")
    parser.add_argument("--ridge", type=float, default=1e-3, help="linear probe L2, relative to the frame count")
    parser.add_argument("--steps", type=int, default=5000, help="mlp probe optimiser steps")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="default: <ckpt dir>/ball_probe[_random]")
    args = parser.parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device), random_init=args.random_init)
    if cfg.data.backend != "h5":
        raise SystemExit("this probe reads the BallCatch HDF5 recording")

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    if "landed" not in reader.file or reader.file["observation"].shape[1] != len(NAMES):
        raise SystemExit(f"{cfg.data.h5_path} is not a BallCatch recording; use probe_cube.py for the cube")
    state_key = model.oracle_key if model.oracle_encoder is not None else None
    normalizer = get_normalizer(cfg, reader, (cfg.data.obs_key, "action") + ((state_key,) if state_key else ()))
    transform = ImageTransform(cfg.data.img_size)

    # The world model's own split: scored on episodes the encoder never trained on.
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    rng = np.random.default_rng(args.seed)
    train_eps = rng.choice(train_eps, min(args.train_episodes, len(train_eps)), replace=False)
    val_eps = rng.choice(val_eps, min(args.val_episodes, len(val_eps)), replace=False)

    def split(episodes):
        return EpisodeFrames(reader, episodes, cfg.data.obs_key, transform, normalizer, state_key)

    gap = cfg.data.frameskip
    train = encode_split(model, split(train_eps), device, args.workers, gap, "encode train")
    val = encode_split(model, split(val_eps), device, args.workers, gap, "encode val")

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / ("ball_probe_random" if args.random_init else "ball_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"ckpt": str(args.ckpt), "random_init": args.random_init, "probe": args.probe,
               "encoder": cfg.model.encoder, "pair_gap_steps": gap, "args": vars(args)}

    label = f"{args.probe} probe on {'RANDOM-INIT ' if args.random_init else ''}{cfg.model.encoder} latents"
    print(f"\n{label}  |  {Path(args.ckpt).parent.name}/{Path(args.ckpt).name}")
    for key in ("single", "pair"):
        Xtr, Ytr, _ = train[key]
        Xva, Yva, fly = val[key]
        probe = fit(Xtr, Ytr, args.probe, args, device)
        with torch.no_grad():
            pred_va = probe(Xva.to(device)).cpu()
            pred_tr = probe(Xtr.to(device)).cpu()
        mean = Ytr.mean(0)
        results[key] = {
            "val": metrics(pred_va, Yva, mean),
            "val_in_flight": metrics(pred_va[fly], Yva[fly], mean),
            "train": metrics(pred_tr, Ytr, mean),
            "num_train_frames": len(Xtr),
        }
        torch.save({"probe": probe.state_dict(), "kind": args.probe, "dim": Xtr.size(1), "input": key,
                    "ckpt": str(args.ckpt)}, out_dir / f"probe_{args.probe}_{key}.pt")
        plot(pred_va, Yva, out_dir / f"scatter_{args.probe}_{key}.png", f"{label} — {key}, held-out")

        r = results[key]
        desc = "e_t" if key == "single" else f"[e_t-{gap}, e_t]"
        print(f"\n  {key:6} ({desc})  {len(Xtr):,} train / {len(Xva):,} held-out frames ({int(fly.sum()):,} in flight)")
        print(f"  {'':12}{'R2 all':>9}{'R2 flight':>11}{'MAE':>9}{'mean-guess':>12}{'train R2':>10}")
        for name, unit in zip(NAMES, UNITS):
            a, f_, t = r["val"][name], r["val_in_flight"][name], r["train"][name]
            print(f"  {name:12}{a['r2']:9.3f}{f_['r2']:11.3f}{a['mae']:7.3f} {unit:<4}{a['mean_guess_mae']:8.3f}"
                  f"{t['r2']:10.3f}")
        e = r["val_in_flight"]["ball_pos_err_cm"]
        print(f"  ball position error in flight: {e['mean']:.1f} cm mean, {e['median']:.1f} cm median, "
              f"{e['within_ball_radius']:.0%} within one ball radius (12 cm)")

    save_json(out_dir / f"metrics_{args.probe}.json", results)
    print(f"\n  R2: 1 = read perfectly, 0 = no better than always guessing the mean")
    print(f"  saved {out_dir}")
    reader.close()


if __name__ == "__main__":
    main()
