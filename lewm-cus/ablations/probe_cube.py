"""Cube-position probe: can the cube's xyz be read off the frozen latent?

    python ablations/probe_cube.py --ckpt checkpoints/observation_model/epoch_007.pt
    python ablations/probe_cube.py --ckpt checkpoints/observation_model/epoch_007.pt --probe mlp

The world model is frozen.  Every `stride`-th frame of a sample of episodes is
encoded once, one linear layer is fitted from the latent to the cube position
(`privileged_block_0_pos`, metres), and it is scored on held-out episodes.  The
episode split is the world model's own, so the probe is scored on trajectories
the encoder never trained on.

`--probe linear` asks whether the position is stored *explicitly*, which is what
the planner's L2 cost relies on.  `--probe mlp` asks the weaker question of
whether it is there at all; a big gap between the two means the information is
present but entangled.

Pooled encoders (ViT class token, state MLP) give one vector per frame and it
is used as is.  A DINO patch grid is average-pooled to 4x4 first, which keeps
the probe small while keeping coarse location.

The probe is saved with its standardisation built in, so in a notebook:

    ckpt = torch.load("<out>/probe_linear.pt")
    probe = CubeProbe(ckpt["dim"], ckpt["kind"]); probe.load_state_dict(ckpt["probe"])
    xyz = probe(features(model.encode(obs)[:, 0]))     # metres
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

TARGET = "privileged_block_0_pos"  # cube xyz, metres
AXES = ("x", "y", "z")


class EpisodeFrames(Dataset):
    """One item per episode: its every `stride`-th observation and the cube position there.

    Whole episodes are read as one contiguous slice: pixels are stored in
    100-frame compressed chunks, so this decompresses each chunk once.
    """

    def __init__(self, reader, episodes, obs_key, stride, transform, normalizer, state_key=None):
        self.reader = reader
        self.episodes = np.asarray(episodes)
        self.obs_key = obs_key
        self.stride = stride
        self.transform = transform
        self.normalizer = normalizer
        # Set for a model that reads the oracle state next to the pixels; its
        # encoder needs the state even though the probe never sees that token.
        self.state_key = state_key

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        start = int(self.reader.ep_offset[ep])
        stop = start + int(self.reader.ep_len[ep])
        obs = self.reader.span(self.obs_key, start, stop)[:: self.stride]
        if self.obs_key == "pixels":
            obs = self.transform(obs)
        else:
            obs = torch.from_numpy(self.normalizer.normalize(self.obs_key, obs).astype(np.float32))
        cube = self.reader.span(TARGET, start, stop)[:: self.stride]
        state = torch.zeros(0)
        if self.state_key is not None:
            raw = self.reader.span(self.state_key, start, stop)[:: self.stride]
            state = torch.from_numpy(self.normalizer.normalize(self.state_key, raw).astype(np.float32))
        return obs, state, torch.from_numpy(cube.astype(np.float32))


def features(emb: torch.Tensor) -> torch.Tensor:
    """(N, P, D) latents -> (N, F) probe inputs; a patch grid is pooled to 4x4."""
    if emb.size(1) == 1:
        return emb[:, 0]
    g = int(round(emb.size(1) ** 0.5))
    grid = emb.transpose(1, 2).reshape(emb.size(0), emb.size(2), g, g)
    return F.adaptive_avg_pool2d(grid, 4).flatten(1)


class CubeProbe(nn.Module):
    """Latent features -> cube xyz in metres, with input/output standardisation built in."""

    def __init__(self, dim: int, kind: str = "linear", hidden: int = 512):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, 3)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
            )
        self.register_buffer("x_mean", torch.zeros(dim))
        self.register_buffer("x_std", torch.ones(dim))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))

    def set_stats(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Standardisation from the training split only."""
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0) + 1e-6)
        self.y_mean.copy_(Y.mean(0))
        self.y_std.copy_(Y.std(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean


@torch.no_grad()
def encode_split(model, dataset, device, workers: int, desc: str):
    """Encode every frame of a split: returns (features, cube xyz) on the CPU."""
    loader = DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda")
    feats, cubes = [], []
    for obs, state, cube in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        # frames go through as a batch of length-1 sequences: (N, 1, ...) -> (N, 1, P, D)
        state = state.to(device, non_blocking=True)[:, None] if state.numel() else None
        emb = model.encode(obs.to(device, non_blocking=True)[:, None], state)[:, 0]
        if model.oracle_encoder is not None:
            # Drop the oracle token: it is the cube position, so probing it would
            # measure the MLP, not what the pixels encode.
            emb = emb[:, :-1]
        feats.append(features(emb).cpu())
        cubes.append(cube)
    return torch.cat(feats), torch.cat(cubes)


def fit(probe: CubeProbe, X, Y, args, device) -> CubeProbe:
    """AdamW on standardised features and targets."""
    probe.to(device).set_stats(X.to(device), Y.to(device))
    Xn = (X.to(device) - probe.x_mean) / probe.x_std
    Yn = (Y.to(device) - probe.y_mean) / probe.y_std

    probe.train()
    opt = torch.optim.AdamW(probe.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for _ in tqdm(range(args.steps), desc="fit probe", dynamic_ncols=True, disable=not sys.stderr.isatty()):
        idx = torch.randint(len(Xn), (args.batch_size,), device=device)
        loss = F.mse_loss(probe.net(Xn[idx]), Yn[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return probe.eval()


def metrics(pred: torch.Tensor, true: torch.Tensor, mean_guess: torch.Tensor) -> dict:
    err = (pred - true).norm(dim=-1) * 100  # cm
    r2 = 1 - (pred - true).pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)
    mae = (pred - true).abs().mean(0) * 100
    out = {
        "mean_err_cm": err.mean().item(),
        "median_err_cm": err.median().item(),
        "within_1cm": (err < 1).float().mean().item(),
        "within_2cm": (err < 2).float().mean().item(),
        "within_4cm": (err < 4).float().mean().item(),  # the simulator's success tolerance
        # the error of always answering the training-set mean position
        "mean_guess_err_cm": ((mean_guess - true).norm(dim=-1) * 100).mean().item(),
        "num_frames": len(true),
    }
    for axis, m, r in zip(AXES, mae, r2):
        out[f"{axis}_mae_cm"] = m.item()
        out[f"{axis}_r2"] = r.item()
    return out


def plot(pred: torch.Tensor, true: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, ax in enumerate(axes):
        t, p = true[:, j].numpy() * 100, pred[:, j].numpy() * 100
        ax.scatter(t, p, s=2, alpha=0.25)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(title=f"cube {AXES[j]}", xlabel="true (cm)", ylabel="probe (cm)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Probe the frozen latent for the cube's xyz.")
    parser.add_argument("--ckpt", required=True, help="world-model checkpoint (epoch_XXX.pt)")
    parser.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--train-episodes", type=int, default=500, help="training episodes to fit on")
    parser.add_argument("--val-episodes", type=int, default=200, help="held-out episodes to score on")
    parser.add_argument("--stride", type=int, default=5, help="raw env steps between probed frames")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="default: <ckpt dir>/cube_probe")
    args = parser.parse_args(argv)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device))
    if cfg.data.backend != "h5":
        raise SystemExit("this probe reads the OGBench cube recording; the checkpoint was trained on the scene one")

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    state_key = model.oracle_key if model.oracle_encoder is not None else None
    keys = (cfg.data.obs_key, "action") + ((state_key,) if state_key else ())
    normalizer = get_normalizer(cfg, reader, keys)
    transform = ImageTransform(cfg.data.img_size)

    # The world model's own split: the probe is scored on episodes the encoder never trained on.
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    rng = np.random.default_rng(args.seed)
    train_eps = rng.choice(train_eps, min(args.train_episodes, len(train_eps)), replace=False)
    val_eps = rng.choice(val_eps, min(args.val_episodes, len(val_eps)), replace=False)

    def split(episodes):
        return EpisodeFrames(reader, episodes, cfg.data.obs_key, args.stride, transform, normalizer, state_key)

    Xtr, Ytr = encode_split(model, split(train_eps), device, args.workers, "encode train")
    Xva, Yva = encode_split(model, split(val_eps), device, args.workers, "encode val")

    probe = fit(CubeProbe(Xtr.size(1), args.probe), Xtr, Ytr, args, device)
    with torch.no_grad():
        pred_tr = probe(Xtr.to(device)).cpu()
        pred_va = probe(Xva.to(device)).cpu()

    mean_guess = Ytr.mean(0)
    train_m = metrics(pred_tr, Ytr, mean_guess)
    val_m = metrics(pred_va, Yva, mean_guess)

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "cube_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"probe": probe.state_dict(), "kind": args.probe, "dim": Xtr.size(1), "ckpt": str(args.ckpt)},
        out_dir / f"probe_{args.probe}.pt",
    )
    save_json(out_dir / f"metrics_{args.probe}.json", {"val": val_m, "train": train_m, "args": vars(args)})
    plot(pred_va, Yva, out_dir / f"scatter_{args.probe}.png")

    tag = f"{cfg.model.encoder} latents" + (" (image token only)" if state_key else "")
    print(f"\ncube xyz, {args.probe} probe on {tag} "
          f"({len(Xtr)} train frames, {len(Xva)} held-out frames)")
    print(f"  held-out error   {val_m['mean_err_cm']:.2f} cm   (median {val_m['median_err_cm']:.2f})"
          f"   | always-the-mean guess {val_m['mean_guess_err_cm']:.2f} cm")
    print(f"  within 1/2/4 cm  {val_m['within_1cm']:.0%} / {val_m['within_2cm']:.0%} / {val_m['within_4cm']:.0%}")
    print("  per axis         " + "  |  ".join(
        f"{a}: {val_m[f'{a}_mae_cm']:.2f} cm, R2 {val_m[f'{a}_r2']:.3f}" for a in AXES))
    print(f"  train error      {train_m['mean_err_cm']:.2f} cm   (far below held-out = the probe overfits)")
    print(f"  saved            {out_dir}")
    reader.close()


if __name__ == "__main__":
    main()
