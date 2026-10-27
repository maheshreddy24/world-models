"""Is a predicted latent self-consistent: does it read the same after decode -> re-encode?

    python ablations/probe_pred_vs_enc.py                      # vit (SIGReg) epoch_006, heldout episodes
    python ablations/probe_pred_vs_enc.py --ckpt checkpoints/cartpole-swingup/vit_ema/epoch_006.pt

For every window of every heldout episode (`<task>-valtest.h5`, never trained on): `history`
encoded real frames as context, then `model.rollout` with the recorded actions for --horizon
latent steps. At each horizon step h, readings of frame t + h:

    pred      probe(z_hat)                       the predictor's autoregressive latent
    redec     probe(enc(dec(z_hat)))             that latent decoded to a frame and encoded again
    enc       probe(enc(x))                      the real frame's latent
    recon     probe(enc(dec(enc(x))))            control: the same round trip on the real latent
    true      the recorded state

The decoder (decoder.py) draws 112px frames; they are upsampled to the encoder's 224px.
Gaps, per quantity (cart cm, pole deg):

    pred_vs_redec    what the round trip changes on a prediction (the question)
    enc_vs_recon     what it changes on a real latent: the decoder's own loss, no prediction
    pred_vs_true, redec_vs_true, enc_vs_true

plus the latent MSE (per token, as in training) of pred vs redec and enc vs recon.

Writes <run dir>/probe_pred_vs_enc/{result.json, result.png}: mean gaps per horizon step, and
the pole-angle readings along one rollout of a few episodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ablations"))

from common import Decoder, LinearProbe, analysis_dir, encode_frames, load_episode  # noqa: E402
from config import heldout_h5, task_h5  # noqa: E402
from src.data import H5Reader, ImageTransform, build_datasets, get_normalizer  # noqa: E402
from src.models import latent_sq_dist  # noqa: E402
from src.tasks import get_task  # noqa: E402
from src.utils import load_model, recalibrate_bn, save_json, set_seed  # noqa: E402

PAIRS = ("pred_vs_redec", "enc_vs_recon", "pred_vs_true", "redec_vs_true", "enc_vs_true")
LATENT_PAIRS = ("pred_vs_redec", "enc_vs_recon")


@torch.no_grad()
def reencode(model, decoder, transform, z: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """enc(dec(z)): (..., P, D) -> (..., P, D)."""
    flat, out = z.reshape(-1, *z.shape[-2:]), []
    mean, std = transform.mean.to(z.device), transform.std.to(z.device)
    for i in range(0, len(flat), chunk):
        img = decoder(flat[i : i + chunk])  # (N, 3, res, res) in [0, 1]
        img = F.interpolate(img, size=(transform.size, transform.size), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
            out.append(model.encode(((img - mean) / std)[:, None])[:, 0].float())
    return torch.cat(out).view(z.shape)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / "cartpole-swingup" / "vit" / "epoch_006.pt")
    p.add_argument("--h5", type=Path, default=None, help="default: the task's heldout table")
    p.add_argument("--horizon", type=int, default=50, help="latent steps (x frameskip rows)")
    p.add_argument("--start-stride", type=int, default=1, help="a window every this many latent steps")
    p.add_argument("--examples", type=int, nargs="+", default=[0, 5, 20, 30], help="episodes drawn in the figure")
    p.add_argument("--example-start", type=int, default=0, help="first context frame of the drawn rollouts")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device))
    cfg.data.h5_path = str(task_h5(cfg.data.task))  # the checkpoint holds the training machine's path
    train_set, _, train_reader, _ = build_datasets(cfg)
    recalibrate_bn(model, train_set, device)  # as common.load_world_model: same latents the probe was fit on
    normalizer = get_normalizer(cfg, train_reader, ("action",))
    train_reader.close()
    set_seed(cfg.seed)
    task = get_task(cfg.data.task)
    probe = LinearProbe.load(analysis_dir(args.ckpt, "probe") / "probe.pt", device)
    decoder = Decoder.load(analysis_dir(args.ckpt, "decoder") / "decoder.pt", device)
    transform = ImageTransform(cfg.data.img_size)
    fs, hist, hor = cfg.data.frameskip, cfg.data.history, args.horizon

    reader = H5Reader(args.h5 or heldout_h5(cfg.data.task))
    sums = {k: torch.zeros(hor, len(task.quantities), device=device, dtype=torch.float64) for k in PAIRS}
    mse = {k: torch.zeros(hor, device=device, dtype=torch.float64) for k in LATENT_PAIRS}
    windows, examples = 0, {}
    for ep in range(reader.num_episodes):
        episode = load_episode(reader, ep, task, normalizer, fs)
        emb = encode_frames(model, episode.frames, transform, device)  # (n+1, P, D)
        recon_all = reencode(model, decoder, transform, emb)
        blocks, truth = episode.blocks.to(device), episode.targets.to(device)
        n = len(blocks)
        starts = torch.arange(0, n + 1 - hist - hor + 1, args.start_stride)
        ctx = starts[:, None] + torch.arange(hist)
        acts = starts[:, None] + torch.arange(hist + hor - 1)
        tgt = starts[:, None] + hist + torch.arange(hor)  # (W, hor)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z_hat = model.rollout(emb[ctx], blocks[acts]).float()  # (W, hor, P, D)
        lat = {"pred": z_hat, "redec": reencode(model, decoder, transform, z_hat), "enc": emb[tgt],
               "recon": recon_all[tgt]}
        read = {k: probe(v) for k, v in lat.items()} | {"true": truth[tgt]}  # (W, hor, K) probe targets
        for pair in PAIRS:
            a, b = pair.split("_vs_")
            sums[pair] += task.errors(read[a], read[b]).sum(0)
        for pair in LATENT_PAIRS:
            a, b = pair.split("_vs_")
            mse[pair] += latent_sq_dist(lat[a], lat[b]).sum(0)
        windows += len(starts)
        if ep in args.examples:
            s = int((starts == args.example_start).nonzero()[0, 0]) if args.example_start in starts else 0
            examples[ep] = {k: task.read(v[s]).cpu().numpy() for k, v in read.items()}  # (hor, Q)
    reader.close()

    res = {"ckpt": str(args.ckpt), "h5": str(args.h5 or heldout_h5(cfg.data.task)), "episodes": reader.num_episodes,
           "windows": windows, "frameskip": fs, "history": hist, "horizon": list(range(1, hor + 1)),
           "units": {q.name: q.unit for q in task.quantities}}
    for pair in PAIRS:
        for k, q in enumerate(task.quantities):
            res[f"{q.name}_{pair}"] = (sums[pair][:, k] / windows).tolist()
    for pair in LATENT_PAIRS:
        res[f"latent_mse_{pair}"] = (mse[pair] / windows).tolist()
    out = analysis_dir(args.ckpt, "probe_pred_vs_enc")
    save_json(out / "result.json", res)
    plot(res, task, examples, out / "result.png")

    show = [h for h in (1, 2, 5, 10, 20, 30, 50) if h <= hor]
    print(f"{windows} windows of {res['episodes']} heldout episodes | horizon in latent steps of {fs} rows")
    print(f"{'':28s}" + "".join(f"{f'h={h}':>9s}" for h in show))
    for q in task.quantities:
        for pair in PAIRS:
            print(f"{q.name + ' ' + pair + f' ({q.unit})':28s}"
                  + "".join(f"{res[f'{q.name}_{pair}'][h - 1]:9.2f}" for h in show))
    for pair in LATENT_PAIRS:
        print(f"{'latent mse ' + pair:28s}" + "".join(f"{res[f'latent_mse_{pair}'][h - 1]:9.1f}" for h in show))
    print(f"saved {out}/result.{{json,png}}")


def plot(res: dict, task, examples: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(res["horizon"])
    style = {"pred_vs_redec": ("probe(pred) vs probe(enc(dec(pred)))", "#eb6834", "o", "-"),
             "enc_vs_recon": ("control: probe(enc(x)) vs probe(enc(dec(enc(x))))", "#7b4fd6", None, "--"),
             "pred_vs_true": ("probe(pred) vs true", "#2a78d6", None, "-"),
             "enc_vs_true": ("probe(enc(x)) vs true", "gray", None, ":")}
    nq = len(task.quantities)
    cols = max(nq + 1, len(examples))
    fig, axes = plt.subplots(2, cols, figsize=(4.8 * cols, 7.8))
    for k, q in enumerate(task.quantities):
        ax = axes[0, k]
        for pair, (lab, color, marker, ls) in style.items():
            ax.plot(h, res[f"{q.name}_{pair}"], color=color, marker=marker, ms=3, ls=ls, label=lab)
        ax.set(title=f"{q.name}: mean gap", xlabel="horizon (latent steps)", ylabel=f"abs difference ({q.unit})")
        ax.legend(fontsize=7)
    ax = axes[0, nq]
    ax.plot(h, res["latent_mse_pred_vs_redec"], color="#eb6834", marker="o", ms=3, label="pred vs enc(dec(pred))")
    ax.plot(h, res["latent_mse_enc_vs_recon"], color="#7b4fd6", ls="--", label="control: enc(x) vs enc(dec(enc(x)))")
    ax.set(title="latent MSE", xlabel="horizon (latent steps)", ylabel="latent MSE")
    ax.legend(fontsize=7)
    for ax in axes[0, nq + 1:]:
        ax.axis("off")

    angle = [k for k, q in enumerate(task.quantities) if q.kind == "angle"][0]
    for ax, (ep, r) in zip(axes[1], sorted(examples.items())):
        for key, color, lab, ls in (("true", "black", "true", "-"), ("pred", "#eb6834", "probe(pred)", "-"),
                                    ("redec", "#2a9d5c", "probe(enc(dec(pred)))", "--")):
            ax.plot(h, np.degrees(r[key][:, angle]), color=color, lw=1.5, label=lab, ls=ls)
        ax.set(title=f"heldout ep {ep}: {task.quantities[angle].name} angle along one rollout",
               xlabel="horizon (latent steps)", ylabel="deg (0 = upright)", ylim=(-185, 185))
        ax.legend(fontsize=7)
    for ax in axes[1, len(examples):]:
        ax.axis("off")
    fig.suptitle(f"{task.name}: probe on predicted latents vs after decode -> re-encode ({Path(res['ckpt']).parent.name})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
