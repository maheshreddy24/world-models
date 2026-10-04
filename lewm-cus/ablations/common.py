"""Shared pieces of the analyses: probe.py, decoder.py and rollout.py.

Latents are (..., P, D): P = 1 for the ViT's class token, P = 196 (a 14x14
grid) for DINOv2 patches. Everything here works for both, and for every task:
what a task's state means lives in src/tasks.py.

Outputs of the analyses live next to the checkpoint they analyse:

    <run dir>/probe/      linear probe: latent -> joint angles
    <run dir>/decoder/    pixel decoder: latent -> frame
    <run dir>/rollout/    rollout metrics, plot and videos
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import H5Reader, ImageTransform, Normalizer, build_datasets  # noqa: E402
from src.tasks import Task, get_task  # noqa: E402
from src.utils import load_model, recalibrate_bn  # noqa: E402

# --------------------------------------------------------------------------- #
#  The world model
# --------------------------------------------------------------------------- #


def load_world_model(ckpt: str | Path, device, random_init: bool = False):
    """A frozen pixel world model, ready for analysis.

    BatchNorm statistics are recalibrated (src.utils.recalibrate_bn) with a
    fixed seed, so every analysis script sees exactly the same latents.

    Returns:
        (model, cfg, task): the model in eval mode on `device`, its training
        config, and the task spec of the recording it was trained on.
    """
    model, cfg = load_model(ckpt, device=str(device), random_init=random_init)
    if cfg.data.obs_key != "pixels":
        sys.exit("the analyses need a pixel model (presets vit, dino)")
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        train_set, _, reader, _ = build_datasets(cfg)
        recalibrate_bn(model, train_set, device)
        reader.close()
    return model, cfg, get_task(cfg.data.task)


def analysis_dir(ckpt: str | Path, name: str) -> Path:
    """`<run dir>/<name>`, where the analyses of a checkpoint are written."""
    return Path(ckpt).parent / name


@torch.no_grad()
def encode_frames(model, frames: np.ndarray, transform: ImageTransform, device, chunk: int = 128) -> torch.Tensor:
    """uint8 frames (N, H, W, 3) -> latents (N, P, D) on `device`."""
    out = []
    for i in range(0, len(frames), chunk):
        x = transform(frames[i : i + chunk]).to(device, non_blocking=True)
        out.append(model.encode(x[:, None])[:, 0].float())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
#  Episodes at latent-step resolution
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    """One episode sampled every `frameskip` rows: frames 0..n and action blocks 0..n-1.

    Block k carries frame k to frame k + 1, as in training.
    """

    frames: np.ndarray  # (n+1, H, W, 3) uint8
    targets: torch.Tensor  # (n+1, K) probe targets of the true state (Task.targets)
    blocks: torch.Tensor  # (n, frameskip * A) normalised action blocks


def action_blocks(reader: H5Reader, ep: int, normalizer: Normalizer, frameskip: int) -> torch.Tensor:
    """(n, frameskip * A) normalised action blocks of one episode."""
    start = int(reader.ep_offset[ep])
    n = (int(reader.ep_len[ep]) - 1) // frameskip
    act = reader.span("action", start, start + n * frameskip)  # the last row's NaN is excluded
    act = normalizer.normalize("action", np.nan_to_num(act, nan=0.0)).astype(np.float32)
    return torch.from_numpy(act.reshape(n, -1))


def load_episode(reader: H5Reader, ep: int, task: Task, normalizer: Normalizer, frameskip: int) -> Episode:
    start = int(reader.ep_offset[ep])
    n = (int(reader.ep_len[ep]) - 1) // frameskip
    stop = start + n * frameskip + 1
    frames = reader.span("pixels", start, stop)[::frameskip]
    obs = torch.from_numpy(reader.span("observation", start, stop)[::frameskip].astype(np.float32))
    return Episode(frames, task.targets(obs), action_blocks(reader, ep, normalizer, frameskip))


# --------------------------------------------------------------------------- #
#  Linear probe: latent -> the task's probe targets
# --------------------------------------------------------------------------- #
def probe_features(emb: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """(..., P, D) latent -> (..., F) probe input.

    One token is used as is (F = D). A patch grid is average-pooled to
    `grid` x `grid` cells (F = grid^2 * D): averaging all 196 DINO patches would
    wash the thin poles out into the background, and the full grid (75k
    numbers) is too wide for a closed-form ridge.
    """
    p, d = emb.shape[-2:]
    if p == 1:
        return emb[..., 0, :].float()
    side = int(round(math.sqrt(p)))
    lead = emb.shape[:-2]
    x = emb.reshape(-1, side, side, d).permute(0, 3, 1, 2).float()  # (N, D, side, side)
    x = F.adaptive_avg_pool2d(x, grid)
    return x.flatten(1).reshape(*lead, d * grid * grid)


class LinearProbe(nn.Module):
    """Standardised features -> standardised targets, one linear map, fitted in closed form."""

    def __init__(self, in_dim: int, out_dim: int, grid: int = 4):
        super().__init__()
        self.grid = grid
        self.net = nn.Linear(in_dim, out_dim)
        for name, n in (("x_mean", in_dim), ("x_std", in_dim), ("y_mean", out_dim), ("y_std", out_dim)):
            self.register_buffer(name, torch.zeros(n) if "mean" in name else torch.ones(n))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Latent (..., P, D) -> targets (..., K)."""
        return self.from_features(probe_features(emb, self.grid))

    def from_features(self, x: torch.Tensor) -> torch.Tensor:
        """Already pooled features (..., F) -> targets (..., K)."""
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean

    @classmethod
    def fit(cls, X: torch.Tensor, Y: torch.Tensor, ridge: float, grid: int, device) -> "LinearProbe":
        """Ridge regression on features X (N, F) -> targets Y (N, K); `ridge` is relative to N."""
        probe = cls(X.size(1), Y.size(1), grid).to(device)
        X, Y = X.to(device), Y.to(device)
        probe.x_mean.copy_(X.mean(0))
        probe.x_std.copy_(X.std(0) + 1e-6)
        probe.y_mean.copy_(Y.mean(0))
        probe.y_std.copy_(Y.std(0) + 1e-6)
        Xn, Yn = (X - probe.x_mean) / probe.x_std, (Y - probe.y_mean) / probe.y_std
        A = Xn.T @ Xn + ridge * len(Xn) * torch.eye(Xn.size(1), device=device)
        probe.net.weight.data.copy_(torch.linalg.solve(A, Xn.T @ Yn).T)
        probe.net.bias.data.zero_()
        return probe.eval()

    def save(self, path: Path, **meta) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "in_dim": self.net.in_features,
                    "out_dim": self.net.out_features, "grid": self.grid, **meta}, path)

    @classmethod
    def load(cls, path: str | Path, device) -> "LinearProbe":
        blob = torch.load(path, map_location="cpu")
        probe = cls(blob["in_dim"], blob["out_dim"], blob.get("grid", 4))
        probe.load_state_dict(blob["state_dict"])
        return probe.to(device).eval()


# --------------------------------------------------------------------------- #
#  Pixel decoder: latent -> frame
# --------------------------------------------------------------------------- #
class Decoder(nn.Module):
    """(..., P, D) latent -> (..., 3, res, res) frame in [0, 1].

    One token goes through a linear layer onto a 7x7 map; a patch grid
    (side x side) enters as a feature map through a 1x1 conv. Either is then
    upsampled x2 until it reaches `res`, which must be (7 or side) * 2^k.
    """

    def __init__(self, dim: int, tokens: int = 1, res: int = 112, base: int = 256):
        super().__init__()
        self.dim, self.tokens, self.res, self.base = dim, tokens, res, base
        self.side = 7 if tokens == 1 else int(round(math.sqrt(tokens)))
        ups = int(round(math.log2(res / self.side)))
        if self.side * 2**ups != res or ups < 1:
            raise ValueError(f"res must be {self.side} * 2^k (k >= 1), got {res}")

        self.register_buffer("mean", torch.zeros(dim))  # per-channel latent standardisation
        self.register_buffer("std", torch.ones(dim))
        if tokens == 1:
            self.fc = nn.Linear(dim, base * 7 * 7)
        else:
            self.stem = nn.Conv2d(dim, base, 1)

        layers, ch = [], base
        for _ in range(ups):
            out = max(ch // 2, 32)
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(ch, out, 3, padding=1), nn.GroupNorm(8, out), nn.GELU(),
                       nn.Conv2d(out, out, 3, padding=1), nn.GroupNorm(8, out), nn.GELU()]
            ch = out
        layers.append(nn.Conv2d(ch, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        lead = z.shape[:-2]
        z = (z.reshape(-1, self.tokens, self.dim).float() - self.mean) / self.std
        if self.tokens == 1:
            x = self.fc(z[:, 0]).view(-1, self.base, 7, 7)
        else:
            x = self.stem(z.transpose(1, 2).reshape(-1, self.dim, self.side, self.side))
        return torch.sigmoid(self.net(x)).view(*lead, 3, self.res, self.res)

    def save(self, path: Path, **meta) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "dim": self.dim, "tokens": self.tokens,
                    "res": self.res, "base": self.base, **meta}, path)

    @classmethod
    def load(cls, path: str | Path, device) -> "Decoder":
        blob = torch.load(path, map_location="cpu")
        dec = cls(blob.get("dim", blob.get("in_dim")), blob.get("tokens", 1), blob["res"], blob["base"])
        dec.load_state_dict(blob["state_dict"])
        return dec.to(device).eval()


def resize_frames(frames: np.ndarray | torch.Tensor, res: int, device) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float (N, 3, res, res) in [0, 1] on `device`."""
    x = torch.as_tensor(frames).to(device).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(res, res), mode="area") / 255


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(..., 3, H, W) in [0, 1] -> (..., H, W, 3) uint8."""
    return (x.clamp(0, 1) * 255).round().byte().movedim(-3, -1).cpu().numpy()


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PSNR of [0, 1] images (..., 3, H, W) -> (...)."""
    mse = (a.float() - b.float()).pow(2).flatten(-3).mean(-1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
