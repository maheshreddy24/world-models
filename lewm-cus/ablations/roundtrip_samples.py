"""Pictures of a predicted latent before and after a decode -> encode round trip.

    python ablations/roundtrip_samples.py                              # vit epoch_006, decoder/
    python ablations/roundtrip_samples.py --decoder decoder_224        # another decoder next to the ckpt

Samples are autoregressive rollouts on heldout episodes (`<task>-valtest.h5`): `history`
encoded real frames as context, the recorded actions, and the prediction z_hat at horizon h.
Each sample is one row of three frames:

    real                    the recorded frame at t + h
    dec(z_hat)              the prediction, decoded
    dec(enc(dec(z_hat)))    the decoded prediction encoded again (upsampled to the encoder's
                            size if the decoder draws smaller frames), then decoded

Writes <run dir>/roundtrip_samples/<decoder>/sample<k>_ep<e>_t<t>_h<h>.png and grid.png.

With --strip H every sample is one rollout shown at every step 1..H instead: rows real,
dec(z_hat), dec(enc(dec(z_hat))), columns h = 1..H, written to
roundtrip_samples/<decoder>_strip<H>/sample<k>_ep<e>_t<t>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ablations"))

from common import Decoder, analysis_dir, encode_frames, load_episode, load_world_model, to_uint8  # noqa: E402
from config import heldout_h5  # noqa: E402
from conveyor_goal import label  # noqa: E402
from src.data import H5Reader, ImageTransform, get_normalizer  # noqa: E402


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "cartpole-swingup" / "vit" / "epoch_006.pt")
    p.add_argument("--decoder", default="decoder", help="decoder folder next to the checkpoint")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20], help="cycled over the samples")
    p.add_argument("--strip", type=int, default=0, help="show every horizon step 1..H of each rollout")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load_world_model(args.ckpt, device)
    decoder = Decoder.load(analysis_dir(args.ckpt, args.decoder) / "decoder.pt", device)
    transform = ImageTransform(cfg.data.img_size)
    mean, std = transform.mean.to(device), transform.std.to(device)
    train_reader = H5Reader(cfg.data.h5_path)
    normalizer = get_normalizer(cfg, train_reader, ("action",))
    train_reader.close()
    fs, hist, res = cfg.data.frameskip, cfg.data.history, decoder.res

    def encode01(img):  # (N, 3, res, res) in [0, 1] -> (N, P, D)
        if img.shape[-1] != transform.size:
            img = F.interpolate(img, size=(transform.size, transform.size), mode="bilinear", align_corners=False)
        return model.encode(((img - mean) / std)[:, None])[:, 0].float()

    reader = H5Reader(heldout_h5(cfg.data.task))
    rng = np.random.default_rng(args.seed)
    episodes = np.linspace(0, reader.num_episodes - 1, args.samples).round().astype(int)  # mixed and expert
    out = analysis_dir(args.ckpt, "roundtrip_samples") / (args.decoder + (f"_strip{args.strip}" if args.strip else ""))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, ep in enumerate(episodes):
        h = args.strip or args.horizons[k % len(args.horizons)]
        episode = load_episode(reader, int(ep), task, normalizer, fs)
        s = int(rng.integers(0, len(episode.blocks) + 1 - hist - h + 1))  # first context frame
        ctx = episode.frames[s : s + hist]
        z_ctx = encode_frames(model, ctx, transform, device)
        acts = episode.blocks[s : s + hist + h - 1].to(device)
        z_hat = model.rollout(z_ctx[None], acts[None])[0]  # (h, P, D): frames s + hist .. s + hist - 1 + h
        if args.strip:
            img = decoder(z_hat)
            img_rt = decoder(encode01(img))
            real = F.interpolate(torch.as_tensor(episode.frames[s + hist : s + hist + h], device=device)
                                 .permute(0, 3, 1, 2).float() / 255, size=(res, res), mode="area")
            t0 = (s + hist - 1) * fs  # row of the last context frame
            grid = np.concatenate([
                np.concatenate([label(f, [f"real h={j + 1}"] if j else [f"real ep {ep} row {t0 + fs}", "h=1"])
                                for j, f in enumerate(to_uint8(real))], 1),
                np.concatenate([label(f, [f"dec(pred) h={j + 1}"]) for j, f in enumerate(to_uint8(img))], 1),
                np.concatenate([label(f, [f"round trip h={j + 1}"]) for j, f in enumerate(to_uint8(img_rt))], 1)], 0)
            Image.fromarray(grid).save(out / f"sample{k:02d}_ep{ep}_t{t0}.png")
            continue
        z_hat = z_hat[-1:]  # (1, P, D) prediction of frame s + hist - 1 + h
        img = decoder(z_hat)  # (1, 3, res, res)
        img_rt = decoder(encode01(img))
        t = (s + hist - 1 + h) * fs  # row of the predicted frame
        real = F.interpolate(torch.as_tensor(episode.frames[s + hist - 1 + h][None], device=device)
                             .permute(0, 3, 1, 2).float() / 255, size=(res, res), mode="area")
        row = np.concatenate([label(to_uint8(real)[0], [f"real  ep {ep} row {t}"]),
                              label(to_uint8(img)[0], [f"dec(pred)  h={h}"]),
                              label(to_uint8(img_rt)[0], ["dec(enc(dec(pred)))"])], 1)
        Image.fromarray(row).save(out / f"sample{k:02d}_ep{ep}_t{t}_h{h}.png")
        rows.append(row)
    reader.close()
    if args.strip:
        print(f"saved {args.samples} strips to {out}")
        return
    half = (len(rows) + 1) // 2
    left, right = np.concatenate(rows[:half], 0), np.concatenate(rows[half:], 0)
    if len(right) < len(left):
        right = np.concatenate([right, np.zeros((len(left) - len(right), *right.shape[1:]), np.uint8)], 0)
    sep = np.full((len(left), 8, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([left, sep, right], 1)).save(out / "grid.png")
    print(f"saved {len(rows)} samples and grid.png to {out}")


if __name__ == "__main__":
    main()
