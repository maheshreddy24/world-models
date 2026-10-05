"""Linear probe: can the task's state be read off the frozen latent?

    python ablations/probe.py --ckpt checkpoints/<task>/vit/epoch_006.pt
    python ablations/probe.py --ckpt <...> --random-init     # same architecture, untrained weights

The world model is frozen. Every `--row-stride`-th frame of its training
episodes is encoded, and a closed-form ridge probe is fitted from the latent to
the task's probe targets (src/tasks.py: angles as cos/sin, positions as is).
Errors per quantity (degrees or cm) are scored on the model's own held-out
episodes, next to a constant mean-state guess.

A patch-grid latent (DINO) is pooled to a 4x4 grid first; see
common.probe_features.

Writes <run dir>/probe[_random]/{probe.pt, probe.json}; rollout.py reads
probe/probe.pt to turn predicted latents into angles and positions.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import LinearProbe, analysis_dir, encode_frames, load_world_model, probe_features
from src.data import H5Reader, ImageTransform, split_episodes
from src.tasks import Task
from src.utils import save_json, set_seed


class EpisodeRows(Dataset):
    """Every `stride`-th frame of an episode (uint8) and its probe targets."""

    def __init__(self, reader: H5Reader, episodes, task: Task, stride: int):
        self.reader, self.episodes, self.task, self.stride = reader, np.asarray(episodes), task, stride

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        start = int(self.reader.ep_offset[ep])
        stop = start + int(self.reader.ep_len[ep])
        frames = self.reader.span("pixels", start, stop)[:: self.stride]
        obs = torch.from_numpy(self.reader.span(self.task.state_key, start, stop)[:: self.stride].astype(np.float32))
        return frames, self.task.targets(obs)


@torch.no_grad()
def collect(model, transform, reader, episodes, task, stride, grid, device, workers, desc):
    """Probe features (N, F) and targets (N, K) on the CPU."""
    loader = DataLoader(EpisodeRows(reader, episodes, task, stride), batch_size=None, num_workers=workers)
    xs, ys = [], []
    for frames, target in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        xs.append(probe_features(encode_frames(model, frames.numpy(), transform, device), grid).cpu())
        ys.append(target)
    return torch.cat(xs), torch.cat(ys)


def score(probe: LinearProbe, task: Task, X: torch.Tensor, Y: torch.Tensor, device) -> dict:
    """Mean / median error per quantity (in its unit) and R^2 per probe target."""
    with torch.no_grad():
        P = probe.from_features(X.to(device)).cpu()
    err = task.errors(P, Y)  # (N, Q)
    r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    out = {}
    for k, q in enumerate(task.quantities):
        out[f"{q.name}_mae_{q.unit}"] = float(err[:, k].mean())
        out[f"{q.name}_median_{q.unit}"] = float(err[:, k].median())
    out["r2_targets"] = r2.tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--random-init", action="store_true", help="probe the untrained architecture instead")
    p.add_argument("--row-stride", type=int, default=2, help="encode every k-th training row")
    p.add_argument("--grid", type=int, default=4, help="pooled grid side for patch latents")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge strength, relative to the sample count")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, task = load_world_model(args.ckpt, device, random_init=args.random_init)
    transform = ImageTransform(cfg.data.img_size)
    set_seed(cfg.seed)

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    # the world model's own split: the probe never sees its held-out episodes
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    X, Y = collect(model, transform, reader, train_eps, task, args.row_stride, args.grid, device, args.workers,
                   "encode train")
    Xv, Yv = collect(model, transform, reader, val_eps, task, 1, args.grid, device, args.workers, "encode held-out")
    reader.close()

    probe = LinearProbe.fit(X, Y, args.ridge, args.grid, device)
    mean_guess = LinearProbe(X.size(1), Y.size(1), args.grid).to(device)  # zero weights: always the mean state
    mean_guess.y_mean.copy_(Y.mean(0))
    mean_guess.net.weight.data.zero_()
    mean_guess.net.bias.data.zero_()

    results = {
        "ckpt": str(args.ckpt), "task": task.name, "random_init": args.random_init, "features": X.size(1),
        "train_frames": len(X), "heldout_frames": len(Xv),
        "train_episodes": len(train_eps), "heldout_episodes": len(val_eps),
        "train": score(probe, task, X, Y, device),
        "heldout": score(probe, task, Xv, Yv, device),
        "heldout_mean_guess": score(mean_guess, task, Xv, Yv, device),
    }
    out = analysis_dir(args.ckpt, "probe_random" if args.random_init else "probe")
    probe.save(out / "probe.pt", ckpt=str(args.ckpt), task=task.name)
    save_json(out / "probe.json", results)

    cols = [(f"{q.name}_mae_{q.unit}", f"{q.name} ({q.unit})") for q in task.quantities]
    print(f"\n{task.name}: mean absolute error")
    print(f"{'':20s}" + "".join(f"{label:>18s}" for _, label in cols))
    for name in ("train", "heldout", "heldout_mean_guess"):
        print(f"{name:20s}" + "".join(f"{results[name][key]:18.2f}" for key, _ in cols))
    print(f"R^2 per target (held-out)  {np.round(results['heldout']['r2_targets'], 3).tolist()}")
    print(f"saved {out / 'probe.pt'}")


if __name__ == "__main__":
    main()
