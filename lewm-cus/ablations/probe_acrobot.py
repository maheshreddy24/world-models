"""Acrobot probe: can the two joint angles be read linearly off the frozen encoder?

    python ablations/probe_acrobot.py --ckpt checkpoints/<acrobot run>/epoch_0NN.pt
    python ablations/probe_acrobot.py --ckpt <...> --random-init        # untrained-encoder floor

The world model is frozen. Every `--row-stride`-th frame of the training
episodes is encoded, and a closed-form ridge probe is fitted from the pooled
latent to the cos/sin of both link angles (4 numbers: an angle itself wraps at
±180°, its cos/sin does not). Angles come back via atan2 and are scored, in
degrees, on the world model's held-out episodes.

The MMBench `observation` stores the two links' orientations as
[upper cos, lower cos, upper sin, lower sin] (columns 0, 2 and 1, 3 pair up to
unit norm), so

    shoulder = angle of the upper link
    elbow    = angle of the lower link - shoulder   (the joint angle between the poles)

Writes <ckpt dir>/acrobot_probe[_random]/probe.pt, which
`ablations/rollout_acrobot.py` loads to read angles off predicted latents.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_ball import Probe, fit  # noqa: E402  (same folder)
from src.data import H5Reader, ImageTransform, build_datasets, split_episodes  # noqa: E402
from src.utils import load_model, recalibrate_bn, save_json, set_seed  # noqa: E402

TARGET_COLS = [0, 2, 1, 3]  # observation -> [cos upper, sin upper, cos lower, sin lower]
JOINTS = ("shoulder", "elbow")


def joint_angles(cs: torch.Tensor) -> torch.Tensor:
    """(..., 4) cos/sin of both links -> (..., 2) shoulder and elbow angle, radians."""
    upper = torch.atan2(cs[..., 1], cs[..., 0])
    lower = torch.atan2(cs[..., 3], cs[..., 2])
    return torch.stack([upper, wrap(lower - upper)], -1)


def wrap(a: torch.Tensor) -> torch.Tensor:
    """Angle into (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def angle_error_deg(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Absolute wrapped angle difference, degrees."""
    return wrap(pred - true).abs() * (180.0 / math.pi)


def features(emb: torch.Tensor) -> torch.Tensor:
    """(..., P, D) latent -> (..., D) probe input; pooled encoders have P = 1."""
    return emb.mean(-2).float()


class EpisodeRows(Dataset):
    """One item per episode: every `stride`-th frame (preprocessed) and its probe target."""

    def __init__(self, reader: H5Reader, episodes, transform: ImageTransform, stride: int):
        self.reader, self.episodes, self.transform, self.stride = reader, np.asarray(episodes), transform, stride

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        start = int(self.reader.ep_offset[ep])
        stop = start + int(self.reader.ep_len[ep])
        frames = self.reader.span("pixels", start, stop)[:: self.stride]
        target = self.reader.span("observation", start, stop)[:: self.stride][:, TARGET_COLS]
        return self.transform(frames), torch.from_numpy(target.astype(np.float32))


@torch.no_grad()
def encode_frames(model, frames: torch.Tensor, device, chunk: int = 256) -> torch.Tensor:
    """(N, 3, H, W) -> (N, P, D), in chunks so a whole episode fits on the GPU."""
    out = [model.encode(frames[i : i + chunk].to(device, non_blocking=True)[:, None])[:, 0]
           for i in range(0, len(frames), chunk)]
    return torch.cat(out)


@torch.no_grad()
def encode_split(model, reader, episodes, transform, stride, device, workers, desc):
    loader = DataLoader(EpisodeRows(reader, episodes, transform, stride), batch_size=None,
                        num_workers=workers, pin_memory=device.type == "cuda")
    xs, ys = [], []
    for frames, target in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        xs.append(features(encode_frames(model, frames, device)).cpu())
        ys.append(target)
    return torch.cat(xs), torch.cat(ys)


def load_probe(path: str | Path, device) -> Probe:
    blob = torch.load(path, map_location="cpu")
    probe = Probe(blob["in_dim"], blob["out_dim"], "linear")
    probe.load_state_dict(blob["state_dict"])
    return probe.to(device).eval()


def score(probe: Probe, X: torch.Tensor, Y: torch.Tensor, device) -> dict:
    with torch.no_grad():
        P = probe(X.to(device)).cpu()
    err = angle_error_deg(joint_angles(P), joint_angles(Y))  # (N, 2)
    r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    out = {f"{j}_mae_deg": float(err[:, k].mean()) for k, j in enumerate(JOINTS)}
    out |= {f"{j}_median_deg": float(err[:, k].median()) for k, j in enumerate(JOINTS)}
    out["r2_cos_sin"] = r2.tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--random-init", action="store_true", help="same architecture, untrained weights")
    p.add_argument("--row-stride", type=int, default=2, help="encode every k-th training row")
    p.add_argument("--ridge", type=float, default=1e-3, help="ridge strength, relative to the sample count")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="default: <ckpt dir>/acrobot_probe[_random]")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device), random_init=args.random_init)
    if cfg.data.obs_key != "pixels":
        sys.exit("this probes the vision encoder; pass a model trained with data.obs_key=pixels")
    set_seed(cfg.seed)
    # eval-mode BatchNorm statistics must match the model (see src/utils.recalibrate_bn)
    windows, _, store, _ = build_datasets(cfg)
    recalibrate_bn(model, windows, device, num_workers=min(4, args.workers))
    store.close()

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    # exactly the world model's own split, so the probe never sees its held-out episodes
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    transform = ImageTransform(cfg.data.img_size)

    X, Y = encode_split(model, reader, train_eps, transform, args.row_stride, device, args.workers, "encode train")
    Xv, Yv = encode_split(model, reader, val_eps, transform, 1, device, args.workers, "encode held-out")
    probe = fit(X, Y, "linear", args, device)

    res = {"ckpt": str(args.ckpt), "random_init": args.random_init, "train_rows": len(X), "heldout_rows": len(Xv),
           "train_episodes": len(train_eps), "heldout_episodes": len(val_eps),
           "train": score(probe, X, Y, device), "heldout": score(probe, Xv, Yv, device)}
    # what "knowing nothing" scores: the training-mean cos/sin for every frame
    res["heldout_mean_guess"] = score(_constant(Y.mean(0), X.size(1), device), Xv, Yv, device)

    out = Path(args.out) if args.out else Path(args.ckpt).parent / ("acrobot_probe" + ("_random" if args.random_init else ""))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.state_dict(), "in_dim": X.size(1), "out_dim": Y.size(1),
                "target_cols": TARGET_COLS, "ckpt": str(args.ckpt)}, out / "probe.pt")
    save_json(out / "probe.json", res)

    print(f"\n{'':22s}{'shoulder MAE':>14s}{'elbow MAE':>12s}   (degrees, held-out episodes)")
    for name in ("train", "heldout", "heldout_mean_guess"):
        r = res[name]
        print(f"{name:22s}{r['shoulder_mae_deg']:14.2f}{r['elbow_mae_deg']:12.2f}")
    print(f"R^2 cos/sin (held-out)  {np.round(res['heldout']['r2_cos_sin'], 3).tolist()}")
    print(f"saved {out / 'probe.pt'}")
    reader.close()


def _constant(y: torch.Tensor, in_dim: int, device) -> Probe:
    """A probe that ignores its input and outputs `y`."""
    probe = Probe(in_dim, y.numel(), "linear").to(device)
    probe.net.weight.data.zero_()
    probe.net.bias.data.zero_()
    probe.y_mean.copy_(y)
    return probe.eval()


if __name__ == "__main__":
    main()
