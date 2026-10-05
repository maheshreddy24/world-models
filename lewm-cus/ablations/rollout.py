"""Compounding error of autoregressive rollouts, four ways.

    python ablations/rollout.py --ckpt checkpoints/<task>/vit/epoch_006.pt
    python ablations/rollout.py --ckpt <...> --extra-h5 <task>-valtest.h5   # more unseen episodes
    python ablations/rollout.py --ckpt <...> --videos 4                     # + decoded videos
    python ablations/rollout.py --ckpt <...> --videos 4 --videos-only

Needs probe.py run on the same checkpoint (its probe turns latents into the
task's angles / positions) and, for --videos, decoder.py.

No planner, no simulator. A window gives the predictor `history` encoded real
frames and predicts `--horizon` latent steps ahead. A window starts at every
`--start-stride`-th step of every unseen episode and every window runs the
full horizon, so each horizon step averages the same windows. Unseen episodes
are the model's held-out episodes plus every episode of each `--extra-h5`.

Conditions, per horizon step h (target frame t+h):

    copy            the last context latent, repeated ("nothing moves")
    tf              teacher-forced: one step from the *real* latents of the `history`
                    frames before t+h; nothing can pile up, so this is the per-step error
    ar              autoregressive, `model.rollout` with the recorded actions: after
                    `history` steps the predictor sees only its own predictions
    shuf            the same rollout, but every action block that drives a predicted
                    frame comes from a different random unseen episode (random
                    offset); the blocks between the context frames stay. If it
                    matches ar, the predictor ignores the actions.

Metrics (means over windows):

    <cond>_mse      squared L2 to the real frame's latent (per token, as in training)
    <q>_<cond>      probe error of quantity q (src/tasks.py), in degrees or cm
    <q>_floor       the probe on the real frame's latent: no prediction at all
    accum_mse, <q>_accum   ar - tf: the accumulated part

At h = 1 ar and tf are the same computation.

Writes <run dir>/rollout/{rollout.json, rollout.png} and, with --videos,
rollout/videos/ep<i>.{mp4,png}: truth | decode(enc(truth)) | teacher-forced |
autoregressive | shuffled, with the `history` context frames first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import (
    Decoder, LinearProbe, action_blocks, analysis_dir, encode_frames, load_episode, load_world_model,
    resize_frames, to_uint8,
)
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
from src.models import latent_sq_dist
from src.tasks import Task
from src.utils import save_json, save_video, set_seed

PREDICTIONS = ("tf", "ar", "shuf")
LABELS = {"tf": "teacher-forced", "ar": "autoregressive", "shuf": "shuffled actions", "copy": "copy last context"}
STYLES = {"ar": "o-", "tf": "s-", "shuf": "^-"}


def summed_metrics(task: Task) -> list[str]:
    """Metrics accumulated as sums over windows (means are taken at the end)."""
    names = ["copy_mse", *(f"{c}_mse" for c in PREDICTIONS)]
    return names + [f"{q.name}_{c}" for q in task.quantities for c in (*PREDICTIONS, "floor")]


# --------------------------------------------------------------------------- #
#  Predictions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def teacher_forced(model, emb: torch.Tensor, blocks: torch.Tensor, hist: int, chunk: int) -> torch.Tensor:
    """(n+1, P, D): frame j >= hist predicted from real frames j-hist..j-1; rows < hist are zero."""
    out = torch.zeros_like(emb)
    targets = torch.arange(hist, len(emb))
    for i in range(0, len(targets), chunk):
        j = targets[i : i + chunk]
        ctx = j[:, None] - hist + torch.arange(hist)  # (J, hist)
        out[j] = model.predict(emb[ctx], model.encode_action(blocks[ctx]))[:, -1]
    return out


def donor_segments(donors: dict[int, torch.Tensor], ep: int, count: int, length: int, rng) -> torch.Tensor:
    """`count` runs of `length` consecutive action blocks, each from a random episode other than `ep`."""
    pool = [d for d, b in donors.items() if d != ep and len(b) >= length]
    if not pool:
        raise SystemExit("shuffled actions need at least two unseen episodes long enough for the horizon")
    segments = []
    for d in rng.choice(pool, size=count):
        b = donors[int(d)]
        u = int(rng.integers(0, len(b) - length + 1))
        segments.append(b[u : u + length])
    return torch.stack(segments)


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def episode_sums(model, probe, task, episode, emb, donors, ep, hist, horizon, stride, chunk, rng, device):
    """Per-horizon metric sums over every full-horizon window of one episode.

    Returns:
        ({metric: (horizon,) float64}, number of windows), or (None, 0) if no window fits.
    """
    n = len(episode.blocks)
    last_start = n + 1 - hist - horizon  # episodes vary in length (PushT); too short ones give no window
    if last_start < 0:
        return None, 0
    starts = torch.arange(0, last_start + 1, stride)

    blocks = episode.blocks.to(device)
    truth = episode.targets.to(device)  # (n+1, K)
    floor_err = task.errors(probe(emb), truth)  # (n+1, Q): probe on the real frames
    tf_all = teacher_forced(model, emb, blocks, hist, chunk)

    sums = {k: torch.zeros(horizon, device=device, dtype=torch.float64) for k in summed_metrics(task)}
    for s in starts.split(chunk):
        ctx = s[:, None] + torch.arange(hist)  # (W, hist) context frames
        acts = s[:, None] + torch.arange(hist + horizon - 1)  # (W, ...) blocks the rollout consumes
        tgt = s[:, None] + hist + torch.arange(horizon)  # (W, horizon) target frames

        shuffled = blocks[acts].clone()
        shuffled[:, hist - 1 :] = donor_segments(donors, ep, len(s), horizon, rng).to(device)
        preds = {
            "tf": tf_all[tgt],
            "ar": model.rollout(emb[ctx], blocks[acts]),
            "shuf": model.rollout(emb[ctx], shuffled),
        }

        target = emb[tgt]
        sums["copy_mse"] += latent_sq_dist(emb[ctx[:, -1:]].expand_as(target), target).sum(0)
        for c, z in preds.items():
            sums[f"{c}_mse"] += latent_sq_dist(z, target).sum(0)
            err = task.errors(probe(z), truth[tgt])  # (W, horizon, Q)
            for k, q in enumerate(task.quantities):
                sums[f"{q.name}_{c}"] += err[..., k].sum(0)
        for k, q in enumerate(task.quantities):
            sums[f"{q.name}_floor"] += floor_err[tgt][..., k].sum(0)
    return {k: v.cpu().numpy() for k, v in sums.items()}, len(starts)


def evaluate(model, probe, task, reader, episodes, normalizer, transform, cfg, args, device, desc) -> dict:
    """Metric sums over every window of `episodes` in one recording."""
    fs, hist = cfg.data.frameskip, cfg.data.history
    # donors for the shuffled-action rollouts: the other episodes of the same set
    donors = {int(e): action_blocks(reader, int(e), normalizer, fs) for e in episodes}
    total = {k: np.zeros(args.horizon) for k in summed_metrics(task)}
    windows = used = 0
    for ep in tqdm(episodes, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        ep = int(ep)
        episode = load_episode(reader, ep, task, normalizer, fs)
        emb = encode_frames(model, episode.frames, transform, device)
        rng = np.random.default_rng([cfg.seed, ep])
        sums, w = episode_sums(model, probe, task, episode, emb, donors, ep, hist, args.horizon,
                               args.start_stride, args.window_batch, rng, device)
        if w:
            for k in total:
                total[k] += sums[k]
            windows += w
            used += 1
    if not windows:
        raise SystemExit(f"{desc}: no episode fits history {hist} + horizon {args.horizon}")
    return {"episodes": used, "windows": windows, "sums": total}


def summarize(task: Task, block: dict) -> dict:
    """Sums -> means, plus the accumulated (ar - tf) error."""
    m = {k: v / block["windows"] for k, v in block["sums"].items()}
    m["accum_mse"] = m["ar_mse"] - m["tf_mse"]
    for q in task.quantities:
        m[f"{q.name}_accum"] = m[f"{q.name}_ar"] - m[f"{q.name}_tf"]
    return {"episodes": block["episodes"], "windows": block["windows"], **{k: v.tolist() for k, v in m.items()}}


def merge(blocks: list[dict]) -> dict:
    return {"episodes": sum(b["episodes"] for b in blocks), "windows": sum(b["windows"] for b in blocks),
            "sums": {k: sum(b["sums"][k] for b in blocks) for k in blocks[0]["sums"]}}


# --------------------------------------------------------------------------- #
#  Outputs
# --------------------------------------------------------------------------- #
def plot(res: dict, task: Task, frameskip: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(res["horizon"])
    panels = 2 + len(task.quantities)
    fig, axes = plt.subplots(1, panels, figsize=(5 * panels, 4))
    for c in PREDICTIONS:
        axes[0].plot(h, res[f"{c}_mse"], STYLES[c], ms=3, label=LABELS[c])
    axes[0].plot(h, res["copy_mse"], "--", color="gray", label=LABELS["copy"])
    axes[0].set(ylabel="latent MSE", title="Latent error")
    axes[1].plot(h, res["accum_mse"], "o-", ms=3, color="crimson")
    axes[1].set(ylabel="autoregressive - teacher-forced", title="Accumulated latent error")
    for ax, q in zip(axes[2:], task.quantities):
        for c in PREDICTIONS:
            ax.plot(h, res[f"{q.name}_{c}"], STYLES[c], ms=3, label=LABELS[c])
        ax.plot(h, res[f"{q.name}_floor"], "--", color="gray", label="probe on the real frame")
        ax.set(ylabel=f"abs error ({q.unit})", title=q.name)
    for ax in axes:
        ax.set_xlabel(f"horizon (latent steps of {frameskip} rows)")
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(task.name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def write_videos(model, decoder, task, reader, episodes, donor_pool, normalizer, transform, cfg, args, out, device):
    """One decoded window per episode: truth | decode(enc(truth)) | teacher-forced | autoregressive | shuffled."""
    import cv2

    fs, hist, res = cfg.data.frameskip, cfg.data.history, decoder.res
    donors = {int(e): action_blocks(reader, int(e), normalizer, fs) for e in donor_pool}
    out.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        ep = int(ep)
        episode = load_episode(reader, ep, task, normalizer, fs)
        n = len(episode.blocks)
        h = min(args.horizon, n + 1 - hist)
        s = min(args.video_start, n + 1 - hist - h)
        emb = encode_frames(model, episode.frames, transform, device)
        blocks = episode.blocks.to(device)

        ctx = emb[s : s + hist]
        acts = blocks[s : s + hist + h - 1]
        shuffled = acts.clone()
        shuffled[hist - 1 :] = donor_segments(donors, ep, 1, h, np.random.default_rng([cfg.seed, ep]))[0].to(device)
        tf = teacher_forced(model, emb, blocks, hist, args.window_batch)[s + hist : s + hist + h]
        ar = model.rollout(ctx[None], acts[None])[0]
        shuf = model.rollout(ctx[None], shuffled[None])[0]

        context = decoder(ctx)
        streams = {
            "truth": resize_frames(episode.frames[s : s + hist + h], res, device),
            "decode(enc(truth))": decoder(emb[s : s + hist + h]),
            "teacher-forced": torch.cat([context, decoder(tf)]),
            "autoregressive": torch.cat([context, decoder(ar)]),
            "shuffled actions": torch.cat([context, decoder(shuf)]),
        }
        imgs = {k: to_uint8(v).copy() for k, v in streams.items()}  # (T, res, res, 3) each
        for name, frames in imgs.items():
            for t, img in enumerate(frames):
                tag = "context" if t < hist else f"h={t - hist + 1}"
                for text, y in ((name, 12), (tag, res - 6)):
                    cv2.putText(img, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)

        save_video(out / f"ep{ep}.mp4", np.concatenate(list(imgs.values()), axis=2), fps=5)
        cols = np.unique(np.linspace(0, hist + h - 1, 12).round().astype(int))
        still = np.concatenate([np.concatenate([frames[c] for c in cols], 1) for frames in imgs.values()], 0)
        Image.fromarray(still).save(out / f"ep{ep}.png")
    print(f"videos  {out}  (window starts at latent step {args.video_start}, {args.horizon} steps)")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--horizon", type=int, default=50, help="latent steps predicted per window")
    p.add_argument("--start-stride", type=int, default=1, help="latent steps between window starts")
    p.add_argument("--extra-h5", nargs="*", default=[], help="more unseen recordings; every episode is used")
    p.add_argument("--window-batch", type=int, default=32, help="windows per forward pass (lower for DINO if OOM)")
    p.add_argument("--videos", type=int, default=0, help="decoded videos for this many held-out episodes")
    p.add_argument("--videos-only", action="store_true", help="skip the metrics (implies --videos 4 if unset)")
    p.add_argument("--video-start", type=int, default=0, help="first context frame (latent step) of the video window")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.videos_only and not args.videos:
        args.videos = 4

    probe_path = analysis_dir(args.ckpt, "probe") / "probe.pt"
    decoder_path = analysis_dir(args.ckpt, "decoder") / "decoder.pt"
    if not args.videos_only and not probe_path.exists():
        sys.exit(f"no probe at {probe_path}; run: python ablations/probe.py --ckpt {args.ckpt}")
    if args.videos and not decoder_path.exists():
        sys.exit(f"no decoder at {decoder_path}; run: python ablations/decoder.py --ckpt {args.ckpt}")
    missing = [path for path in args.extra_h5 if not Path(path).exists()]
    if missing:
        sys.exit(f"--extra-h5 not found: {missing}\n"
                 "make it with: python datasets/prepare_mmbench.py --task <task> --heldout")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, task = load_world_model(args.ckpt, device)
    set_seed(cfg.seed)
    transform = ImageTransform(cfg.data.img_size)
    out = analysis_dir(args.ckpt, "rollout")

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action",))  # training statistics, also for --extra-h5
    _, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)

    if args.videos:
        write_videos(model, Decoder.load(decoder_path, device), task, reader, np.sort(val_eps)[: args.videos],
                     val_eps, normalizer, transform, cfg, args, out / "videos", device)
        if args.videos_only:
            reader.close()
            return

    probe = LinearProbe.load(probe_path, device)
    blocks = {"heldout": evaluate(model, probe, task, reader, val_eps, normalizer, transform, cfg, args, device,
                                  "held-out")}
    reader.close()
    for path in args.extra_h5:
        extra = H5Reader(path, rdcc_mb=cfg.data.rdcc_mb)
        blocks[Path(path).stem] = evaluate(model, probe, task, extra, np.arange(extra.num_episodes), normalizer,
                                           transform, cfg, args, device, Path(path).stem)
        extra.close()

    res = {"ckpt": str(args.ckpt), "task": task.name, "frameskip": cfg.data.frameskip, "history": cfg.data.history,
           "units": {q.name: q.unit for q in task.quantities},
           "horizon": list(range(1, args.horizon + 1)), **summarize(task, merge(list(blocks.values()))),
           "per_source": {k: summarize(task, b) for k, b in blocks.items()}}
    save_json(out / "rollout.json", res)
    plot(res, task, cfg.data.frameskip, out / "rollout.png")

    show = sorted({1, 2, 5, 10, 20, 30, 50, args.horizon} & set(res["horizon"]))
    sources = ", ".join(f"{k}: {b['episodes']} eps" for k, b in res["per_source"].items())
    print(f"\n{task.name}: {res['episodes']} unseen episodes, {res['windows']} windows ({sources})")
    print(f"{'horizon':20s}" + "".join(f"{h:>10d}" for h in show))
    rows = ["copy_mse", "tf_mse", "ar_mse", "shuf_mse", "accum_mse"]
    rows += [f"{q.name}_{c}" for q in task.quantities for c in ("floor", "tf", "ar", "shuf", "accum")]
    for k in rows:
        print(f"{k:20s}" + "".join(f"{res[k][h - 1]:10.3f}" for h in show))
    print(f"saved {out / 'rollout.json'} and rollout.png")


if __name__ == "__main__":
    main()
