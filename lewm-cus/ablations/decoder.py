"""Pixel decoder on the frozen world model's latents, for looking at rollouts.

    python ablations/decoder.py --ckpt checkpoints/<task>/vit/epoch_006.pt
    python ablations/decoder.py --ckpt <...> --res 224 --steps 30000
    python ablations/decoder.py --ckpt <...> --latent-noise 0.1   # tolerate off-manifold rollout latents

LeWM never decodes; this decoder only exists to show what a latent holds.
Every `--row-stride`-th frame of the world model's training episodes is encoded
once, and a small conv decoder is fitted from the latent (the space the
predictor outputs) to the frame at `--res`. Nothing flows back into the model.

The poles are thin and most of each frame is background, so plain MSE would
happily decode an empty sky: pixels that differ from the mean training frame
(where a pole can be) get `--fg-weight` times the loss.

Writes <run dir>/decoder/{decoder.pt, decoder.json, samples.png}; samples.png
shows held-out frames (top) and their reconstructions (bottom).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import Decoder, analysis_dir, encode_frames, load_world_model, psnr, resize_frames, to_uint8
from src.data import H5Reader, ImageTransform, split_episodes
from src.utils import save_json, set_seed


@torch.no_grad()
def collect(model, transform, reader, episodes, res, stride, device, desc):
    """Latents (N, P, D) as fp16 and frames (N, 3, res, res) as uint8, both on the CPU."""
    zs, xs = [], []
    for ep in tqdm(episodes, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        start = int(reader.ep_offset[int(ep)])
        frames = reader.span("pixels", start, start + int(reader.ep_len[int(ep)]))[::stride]
        zs.append(encode_frames(model, frames, transform, device).half().cpu())
        xs.append((resize_frames(frames, res, device) * 255).round().byte().cpu())
    return torch.cat(zs), torch.cat(xs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--res", type=int, default=112, help="decoded frame side (7 or grid side, times 2^k)")
    p.add_argument("--base", type=int, default=256, help="channels at the first map")
    p.add_argument("--row-stride", type=int, default=2, help="use every k-th training row")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fg-weight", type=float, default=10.0, help="extra loss weight where a pole can be")
    p.add_argument("--latent-noise", type=float, default=0.0,
                   help="gaussian noise on the standardised latent while training (0 = clean)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_world_model(args.ckpt, device)
    set_seed(cfg.seed)
    transform = ImageTransform(cfg.data.img_size)

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    Z, X = collect(model, transform, reader, train_eps, args.res, args.row_stride, device, "encode train")
    Zv, Xv = collect(model, transform, reader, val_eps, args.res, 5, device, "encode held-out")
    reader.close()
    del model
    torch.cuda.empty_cache()

    tokens, dim = Z.shape[1:]
    dec = Decoder(dim, tokens, args.res, args.base).to(device)
    sample = Z[torch.randperm(len(Z))[:4096]].float()
    dec.mean.copy_(sample.mean((0, 1)))
    dec.std.copy_(sample.std((0, 1)) + 1e-6)

    # pixels far from the mean frame are where a pole can be
    mean_img = X[torch.randperm(len(X))[:5000]].float().mean(0).to(device) / 255  # (3, res, res)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(cfg.seed)
    bar = tqdm(range(args.steps), desc="fit decoder", dynamic_ncols=True, disable=not sys.stderr.isatty())
    for step in bar:
        idx = torch.randint(len(Z), (args.batch_size,), generator=gen)
        z = Z[idx].to(device, non_blocking=True).float()
        x = X[idx].to(device, non_blocking=True).float() / 255
        if args.latent_noise > 0:
            z = z + args.latent_noise * dec.std * torch.randn_like(z)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            y = dec(z)
        weight = 1 + args.fg_weight * ((x - mean_img).abs().amax(1, keepdim=True) > 0.1).float()
        loss = (weight * (y.float() - x).pow(2)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            bar.set_postfix(loss=f"{loss.item():.5f}")

    dec.eval()
    with torch.no_grad():
        Yv = torch.cat([dec(Zv[i : i + 256].to(device).float()).cpu() for i in range(0, len(Zv), 256)])
    truth = Xv.float() / 255
    results = {
        "ckpt": str(args.ckpt), "res": args.res, "steps": args.steps, "latent_noise": args.latent_noise,
        "fg_weight": args.fg_weight, "train_frames": len(Z), "heldout_frames": len(Zv),
        "heldout_psnr": float(psnr(Yv, truth).mean()),
        "mean_frame_psnr": float(psnr(mean_img.cpu().expand_as(truth), truth).mean()),
    }

    out = analysis_dir(args.ckpt, "decoder")
    dec.save(out / "decoder.pt", ckpt=str(args.ckpt))
    save_json(out / "decoder.json", results)
    pick = torch.linspace(0, len(Zv) - 1, 10).long()
    grid = np.concatenate([np.concatenate(list(to_uint8(truth[pick])), 1),
                           np.concatenate(list(to_uint8(Yv[pick])), 1)], 0)
    Image.fromarray(grid).save(out / "samples.png")

    print(f"\nheld-out PSNR  decoder {results['heldout_psnr']:.2f} dB | mean frame {results['mean_frame_psnr']:.2f} dB")
    print(f"saved {out / 'decoder.pt'} and samples.png (top: truth, bottom: decoded)")


if __name__ == "__main__":
    main()
