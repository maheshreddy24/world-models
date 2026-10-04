"""Ball-centre probe in pixel space: can (u, v) of the ball be read off the frozen latent?

    python ablations/probe_ball_px.py --ckpt checkpoints/exp_1790681067/epoch_005.pt
    python ablations/probe_ball_px.py --ckpt checkpoints/exp_1790681067/epoch_005.pt --probe mlp
    python ablations/probe_ball_px.py --ckpt checkpoints/exp_1790681067/epoch_005.pt --random-init   # untrained floor

One frame in, one target out: the ball's centre in pixels of the 128x128 frame
the model sees (u = column from the left, v = row from the top). The centre is
the simulator position projected with the renderer's own camera, so it is
exact, not detected. The probe is fitted on training episodes and scored on the
world model's held-out episodes; every frame of each episode is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_ball import EpisodeFrames, fit
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
from src.envs.ball import BALL_R, SUPERSAMPLE, VIEW_X, VIEW_Y
from src.utils import load_model, save_json, set_seed


def world_to_px(xy: torch.Tensor, size: int) -> torch.Tensor:
    """Ball centre in metres (x right, y up) -> (u, v) pixels, as `BallCatchSim.render` draws it.

    Pixel j's centre is at j. The renderer draws at SUPERSAMPLE x and area-
    downsamples, which shifts every point by (SUPERSAMPLE - 1) / (2 SUPERSAMPLE) px.
    """
    shift = (SUPERSAMPLE - 1) / (2 * SUPERSAMPLE)
    u = (xy[:, 0] - VIEW_X[0]) * size / (VIEW_X[1] - VIEW_X[0]) - shift
    v = (VIEW_Y[1] - xy[:, 1]) * size / (VIEW_Y[1] - VIEW_Y[0]) - shift
    return torch.stack([u, v], -1)


@torch.no_grad()
def encode(model, dataset, device, workers: int, size: int, desc: str):
    """(features, ball centre in px) for every frame of every episode."""
    loader = DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda")
    feats, targets = [], []
    for obs, state, obs6, _ in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        state = state.to(device, non_blocking=True)[:, None] if state.numel() else None
        emb = model.encode(obs.to(device, non_blocking=True)[:, None], state)[:, 0]
        if model.oracle_encoder is not None:
            emb = emb[:, :-1]
        feats.append(emb.mean(1).float().cpu())
        targets.append(world_to_px(obs6[:, :2], size))
    return torch.cat(feats), torch.cat(targets)


def metrics(pred: torch.Tensor, true: torch.Tensor, train_mean: torch.Tensor, radius_px: float) -> dict:
    err = (pred - true).norm(dim=-1)
    guess = (train_mean - true).norm(dim=-1)
    r2 = 1 - (pred - true).pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)
    return {
        "mean_err_px": err.mean().item(),
        "median_err_px": err.median().item(),
        "within_2px": (err < 2).float().mean().item(),
        "within_ball_radius": (err < radius_px).float().mean().item(),
        "mean_guess_err_px": guess.mean().item(),
        "u_r2": r2[0].item(),
        "v_r2": r2[1].item(),
        "num_frames": len(true),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Probe the frozen latent for the ball centre in pixels.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--random-init", action="store_true", help="same architecture, untrained weights")
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--val-episodes", type=int, default=500)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="default: <ckpt dir>/ball_px_probe[_random]")
    args = parser.parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device), random_init=args.random_init)
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    if "landed" not in reader.file:
        raise SystemExit(f"{cfg.data.h5_path} is not a BallCatch recording")
    state_key = model.oracle_key if model.oracle_encoder is not None else None
    normalizer = get_normalizer(cfg, reader, (cfg.data.obs_key, "action") + ((state_key,) if state_key else ()))
    size = cfg.data.img_size
    radius_px = BALL_R * size / (VIEW_X[1] - VIEW_X[0])

    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    rng = np.random.default_rng(args.seed)
    train_eps = rng.choice(train_eps, min(args.train_episodes, len(train_eps)), replace=False)
    val_eps = rng.choice(val_eps, min(args.val_episodes, len(val_eps)), replace=False)

    def split(eps):
        return EpisodeFrames(reader, eps, cfg.data.obs_key, ImageTransform(size), normalizer, state_key)

    Xtr, Ytr = encode(model, split(train_eps), device, args.workers, size, "encode train")
    Xva, Yva = encode(model, split(val_eps), device, args.workers, size, "encode val")

    probe = fit(Xtr, Ytr, args.probe, args, device)
    with torch.no_grad():
        pred_va, pred_tr = probe(Xva.to(device)).cpu(), probe(Xtr.to(device)).cpu()
    val_m = metrics(pred_va, Yva, Ytr.mean(0), radius_px)
    train_m = metrics(pred_tr, Ytr, Ytr.mean(0), radius_px)

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / (
        "ball_px_probe_random" if args.random_init else "ball_px_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"probe": probe.state_dict(), "kind": args.probe, "dim": Xtr.size(1), "ckpt": str(args.ckpt)},
               out_dir / f"probe_{args.probe}.pt")
    save_json(out_dir / f"metrics_{args.probe}.json", {"val": val_m, "train": train_m, "img_size": size, "args": vars(args)})

    tag = f"{'RANDOM-INIT ' if args.random_init else ''}{cfg.model.encoder}"
    print(f"\nball centre (u, v) in px of the {size}x{size} frame, {args.probe} probe on {tag} latents")
    print(f"  {len(Xtr):,} train / {len(Xva):,} held-out frames")
    print(f"  held-out error   {val_m['mean_err_px']:.1f} px mean, {val_m['median_err_px']:.1f} px median"
          f"   | always-the-mean guess {val_m['mean_guess_err_px']:.1f} px")
    print(f"  within 2 px / one ball radius ({radius_px:.1f} px)   "
          f"{val_m['within_2px']:.0%} / {val_m['within_ball_radius']:.0%}")
    print(f"  R2  u {val_m['u_r2']:.3f}   v {val_m['v_r2']:.3f}   (train: u {train_m['u_r2']:.3f}, v {train_m['v_r2']:.3f})")
    print(f"  saved {out_dir}")
    reader.close()


if __name__ == "__main__":
    main()
