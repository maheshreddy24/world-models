"""Acrobot: a pixel decoder on the frozen world model's latents, for looking at rollouts.

    python ablations/decoder_acrobot.py --ckpt checkpoints/<acrobot run>/epoch_0NN.pt
    python ablations/decoder_acrobot.py --ckpt <...> --res 224 --steps 30000
    python ablations/decoder_acrobot.py --ckpt <...> --latent-noise 0.1    # tolerate off-manifold rollout latents

LeWM has no decoder and never needs one; this one only exists to *show* what a
latent holds. The world model is frozen (BatchNorm recalibrated as in the probe);
every `--row-stride`-th frame of its training episodes is encoded once, and a
small conv decoder is fitted from the latent (the same post-projector space the
predictor outputs) to the frame at `--res`. Nothing flows back into the model.

The poles are thin and the background is most of every frame, so plain MSE
would happily decode an empty sky. Pixels that differ from the mean training
frame (i.e. where a pole can be) get `--fg-weight` times the loss.

Held-out episodes are reconstructed for the PSNR and the sample grid. Writes
<ckpt dir>/acrobot_decoder/{decoder.pt, decoder.json, samples.png};
`ablations/rollout_acrobot.py --videos N` then decodes rollouts with it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_acrobot import encode_frames  # noqa: E402  (same folder)
from src.data import H5Reader, ImageTransform, build_datasets, split_episodes  # noqa: E402
from src.utils import load_model, recalibrate_bn, save_json, set_seed  # noqa: E402


class Decoder(nn.Module):
    """(..., P, D) latent -> (..., 3, res, res) in [0, 1]. res must be 7 * 2^k."""

    def __init__(self, in_dim: int, res: int = 112, base: int = 256):
        super().__init__()
        ups = int(round(math.log2(res / 7)))
        if 7 * 2**ups != res:
            raise ValueError(f"res must be 7 * 2^k (56, 112, 224), got {res}")
        self.in_dim, self.res, self.base = in_dim, res, base
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.fc = nn.Linear(in_dim, base * 7 * 7)
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
        x = (z.flatten(-2).reshape(-1, self.in_dim).float() - self.mean) / self.std
        x = self.fc(x).view(-1, self.base, 7, 7)
        return torch.sigmoid(self.net(x)).view(*lead, 3, self.res, self.res)


def load_decoder(path: str | Path, device) -> Decoder:
    blob = torch.load(path, map_location="cpu")
    dec = Decoder(blob["in_dim"], blob["res"], blob["base"])
    dec.load_state_dict(blob["state_dict"])
    return dec.to(device).eval()


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(..., 3, H, W) in [0, 1] -> (..., H, W, 3) uint8."""
    return (x.clamp(0, 1) * 255).round().byte().movedim(-3, -1).cpu().numpy()


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR of [0, 1] images (..., 3, H, W) -> (...)."""
    mse = (a.float() - b.float()).pow(2).flatten(-3).mean(-1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


@torch.no_grad()
def collect(model, reader, episodes, transform, res, stride, device, desc):
    """Latents (N, P, D) on `device` and frames (N, 3, res, res) uint8 on `device`."""
    zs, xs = [], []
    for ep in tqdm(episodes, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        start = int(reader.ep_offset[int(ep)])
        frames = reader.span("pixels", start, start + int(reader.ep_len[int(ep)]))[::stride]
        zs.append(encode_frames(model, transform(frames), device).float())
        x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()
        xs.append(F.interpolate(x, size=(res, res), mode="area").round().byte())
    return torch.cat(zs), torch.cat(xs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--res", type=int, default=112, help="decoded frame side: 56, 112 or 224")
    p.add_argument("--base", type=int, default=256, help="channels at the 7x7 stage")
    p.add_argument("--row-stride", type=int, default=2, help="use every k-th training row")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fg-weight", type=float, default=10.0, help="extra loss weight where a pole can be")
    p.add_argument("--latent-noise", type=float, default=0.0,
                   help="gaussian noise on the standardised latent during training (0 = clean)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="default: <ckpt dir>/acrobot_decoder")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device))
    if cfg.data.obs_key != "pixels":
        sys.exit("expects a pixel model (preset acrobot_pixels)")
    set_seed(cfg.seed)
    windows, _, store, _ = build_datasets(cfg)
    recalibrate_bn(model, windows, device)  # same latents the probe and rollout use
    store.close()

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    transform = ImageTransform(cfg.data.img_size)
    Z, X = collect(model, reader, train_eps, transform, args.res, args.row_stride, device, "encode train")
    Zv, Xv = collect(model, reader, val_eps, transform, args.res, 5, device, "encode held-out")
    reader.close()
    del model
    torch.cuda.empty_cache()

    dec = Decoder(Z[0].numel(), args.res, args.base).to(device)
    flat = Z.flatten(1)
    dec.mean.copy_(flat.mean(0))
    dec.std.copy_(flat.std(0) + 1e-6)

    # where the background is not: anything far from the mean frame is a pole somewhere
    mean_img = X[:: max(1, len(X) // 5000)].float().mean(0) / 255  # (3, res, res)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    bar = tqdm(range(args.steps), desc="fit decoder", dynamic_ncols=True, disable=not sys.stderr.isatty())
    for step in bar:
        idx = torch.randint(len(Z), (args.batch_size,), device=device, generator=gen)
        z, x = Z[idx], X[idx].float() / 255
        if args.latent_noise > 0:
            z = z + args.latent_noise * dec.std.view(z.shape[1:]) * torch.randn_like(z)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            y = dec(z)
        w = 1 + args.fg_weight * ((x - mean_img).abs().amax(1, keepdim=True) > 0.1).float()
        loss = (w * (y.float() - x).pow(2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            bar.set_postfix(loss=f"{loss.item():.5f}")

    dec.eval()
    with torch.no_grad():
        Yv = torch.cat([dec(Zv[i : i + 512]) for i in range(0, len(Zv), 512)])
        p_dec = psnr(Yv, Xv.float() / 255)
        p_mean = psnr(mean_img.expand_as(Yv), Xv.float() / 255)
    res = {"ckpt": str(args.ckpt), "res": args.res, "steps": args.steps, "train_frames": len(Z),
           "heldout_frames": len(Zv), "latent_noise": args.latent_noise, "fg_weight": args.fg_weight,
           "heldout_psnr": float(p_dec.mean()), "mean_frame_psnr": float(p_mean.mean())}

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "acrobot_decoder"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dec.state_dict(), "in_dim": dec.in_dim, "res": args.res, "base": args.base,
                "ckpt": str(args.ckpt)}, out / "decoder.pt")
    save_json(out / "decoder.json", res)

    # held-out samples: truth on top, decode(encode(truth)) below
    pick = torch.linspace(0, len(Zv) - 1, 10).long()
    grid = np.concatenate([np.concatenate(list(to_uint8(Xv[pick].float() / 255)), 1),
                           np.concatenate(list(to_uint8(Yv[pick])), 1)], 0)
    Image.fromarray(grid).save(out / "samples.png")

    print(f"\nheld-out PSNR  decoder {res['heldout_psnr']:.2f} dB | mean frame {res['mean_frame_psnr']:.2f} dB")
    print(f"saved {out / 'decoder.pt'} and samples.png (top: truth, bottom: decoded)")


if __name__ == "__main__":
    main()
