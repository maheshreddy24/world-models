"""Acrobot: compounding error of autoregressive rollouts driven by the recorded actions.

    python ablations/rollout_acrobot.py --ckpt checkpoints/<acrobot run>/epoch_0NN.pt
    python ablations/rollout_acrobot.py --ckpt <...> --horizon 80 --start-stride 2
    python ablations/rollout_acrobot.py --ckpt <...> --extra-h5 <mmbench val+test>.h5   # more unseen episodes

Run `ablations/probe_acrobot.py --ckpt <same ckpt>` first; its linear probe reads
the joint angles off the latents here.

No planner and no simulator. Each window gives the predictor `data.history`
encoded frames of a real episode plus the action blocks that were actually
executed, and it predicts the next `--horizon` latents on its own output
(`model.rollout`). Every window starts at a different step of the episode, all
windows run the full horizon, and the numbers are averaged over every window
of every unseen episode. Per horizon step h:

    latent_mse      ||z_hat(t+h) - enc(x(t+h))||^2   autoregressive (per token, as in training)
    shuf_mse        the same rollout, but every action block that drives a *predicted* frame
                    is replaced by a contiguous segment from a different random unseen
                    episode (random offset). The blocks between the context frames stay.
                    If it matches latent_mse, the predictor is not using the actions.
    tf_mse          the same for the teacher-forced one-step prediction of frame t+h: the
                    predictor fed the *real* latents of the `history` frames before it.
                    No error can pile up, so it is the per-step error at those frames.
    accum_mse       latent_mse - tf_mse: what feeding predictions back in has added
    copy_mse        ||enc(x(t)) - enc(x(t+h))||^2    "nothing moves": the last context latent
    <joint>_deg     |angle(probe(z_hat(t+h))) - true angle(t+h)|, degrees (autoregressive)
    <joint>_tf      the same on the teacher-forced prediction
    <joint>_shuf    the same on the shuffled-action rollout
    <joint>_accum   <joint>_deg - <joint>_tf
    <joint>_floor   |angle(probe(enc(x(t+h)))) - true angle(t+h)|: the probe on the real
                    frame, no prediction at all.

At h = 1 the autoregressive and teacher-forced predictions are the same computation.

Unseen episodes = the world model's held-out episodes of its own h5
(`data.val_episodes`, never trained on) plus every episode of each `--extra-h5`.
Writes <ckpt dir>/acrobot_rollout/{rollout.json, rollout.png}.

`--videos N` also decodes one window of each of the first N held-out episodes
with the decoder from `ablations/decoder_acrobot.py` (train it first) into
videos/ep<i>.mp4 and ep<i>.png, five streams side by side:

    truth | decode(enc(truth)) | teacher-forced | autoregressive | shuffled actions

The first `history` frames are the context (the same in every stream). The
second stream is what the decoder can do at best; the gap between the last two
is the accumulated error, in pixels. `--videos-only` skips the metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_acrobot import JOINTS, TARGET_COLS, angle_error_deg, encode_frames, features, joint_angles, load_probe  # noqa: E402
from src.data import H5Reader, ImageTransform, build_datasets, get_normalizer, split_episodes  # noqa: E402
from src.models import latent_sq_dist  # noqa: E402
from src.utils import load_model, recalibrate_bn, save_json, set_seed  # noqa: E402

METRICS = ("latent_mse", "tf_mse", "shuf_mse", "copy_mse") + tuple(
    f"{j}{s}" for j in JOINTS for s in ("_deg", "_tf", "_shuf", "_floor"))


def action_blocks(reader, ep: int, normalizer, fs: int) -> torch.Tensor:
    """(n, fs*A) normalised action blocks of one episode; block k drives frame k -> k+1."""
    start = int(reader.ep_offset[ep])
    n = (int(reader.ep_len[ep]) - 1) // fs
    act = reader.span("action", start, start + n * fs)  # the last row's NaN is excluded
    act = normalizer.normalize("action", np.nan_to_num(act, nan=0.0)).astype(np.float32)
    return torch.from_numpy(act.reshape(n, -1))


def donor_segments(donors: dict, ep: int, count: int, length: int, rng) -> torch.Tensor:
    """`count` segments of `length` consecutive blocks, each from a random episode other than `ep`."""
    pool = [d for d, b in donors.items() if d != ep and len(b) >= length]
    if not pool:
        raise SystemExit("shuffled actions need at least two episodes long enough for the horizon")
    out = []
    for d in rng.choice(pool, size=count):
        b = donors[int(d)]
        u = int(rng.integers(0, len(b) - length + 1))
        out.append(b[u : u + length])
    return torch.stack(out)


@torch.no_grad()
def episode_sums(model, probe, reader, ep, normalizer, transform, cfg, horizon, stride, device, donors, rng):
    """Summed per-horizon metrics over every full-horizon window of one episode.

    Returns:
        ({metric: (horizon,) float64 sums}, number of windows)
    """
    fs, hist = cfg.data.frameskip, cfg.data.history
    start = int(reader.ep_offset[ep])
    n = (int(reader.ep_len[ep]) - 1) // fs  # action blocks; frames 0..n at rows 0, fs, ..., n*fs
    starts = np.arange(0, n + 1 - hist - horizon + 1, stride)
    if len(starts) == 0:
        return None, 0

    frames = reader.span("pixels", start, start + n * fs + 1)[::fs]  # (n+1, H, W, 3)
    obs = reader.span("observation", start, start + n * fs + 1)[::fs][:, TARGET_COLS]
    true_ang = joint_angles(torch.from_numpy(obs.astype(np.float32))).to(device)  # (n+1, 2)
    blocks = donors[ep].to(device)  # (n, fs*A), block k drives frame k -> k+1

    emb = encode_frames(model, transform(frames), device)  # (n+1, P, D)
    floor_ang = joint_angles(probe(features(emb)))  # (n+1, 2): probe on the real frames

    # teacher-forced: frame j (j >= hist) predicted from real frames j-hist..j-1 and their blocks
    j = torch.arange(hist, n + 1)
    idx_tf = j[:, None] - hist + torch.arange(hist)  # (n+1-hist, hist)
    tf = torch.zeros_like(emb)
    tf[hist:] = model.predict(emb[idx_tf], model.encode_action(blocks[idx_tf]))[:, -1]
    tf_ang = joint_angles(probe(features(tf)))  # rows < hist are never a target

    # window s: context frames s..s+hist-1, blocks s..s+hist+horizon-2 -> frames s+hist..s+hist+horizon-1
    idx_ctx = torch.as_tensor(starts)[:, None] + torch.arange(hist)
    idx_act = torch.as_tensor(starts)[:, None] + torch.arange(hist + horizon - 1)
    idx_tgt = torch.as_tensor(starts)[:, None] + hist + torch.arange(horizon)  # (W, horizon)

    pred = model.rollout(emb[idx_ctx], blocks[idx_act])  # (W, horizon, P, D)
    target = emb[idx_tgt]
    copy = emb[idx_ctx[:, -1:]].expand_as(target)
    pred_ang = joint_angles(probe(features(pred)))  # (W, horizon, 2)

    # shuffled: blocks hist-1 .. (the ones that drive predicted frames) come from another episode
    shuf_act = blocks[idx_act].clone()
    shuf_act[:, hist - 1 :] = donor_segments(donors, ep, len(starts), horizon, rng).to(device)
    shuf = model.rollout(emb[idx_ctx], shuf_act)
    shuf_ang = joint_angles(probe(features(shuf)))

    err = angle_error_deg(pred_ang, true_ang[idx_tgt])
    err_shuf = angle_error_deg(shuf_ang, true_ang[idx_tgt])
    err_tf = angle_error_deg(tf_ang[idx_tgt], true_ang[idx_tgt])
    floor = angle_error_deg(floor_ang[idx_tgt], true_ang[idx_tgt])
    sums = {"latent_mse": latent_sq_dist(pred, target), "tf_mse": latent_sq_dist(tf[idx_tgt], target),
            "shuf_mse": latent_sq_dist(shuf, target), "copy_mse": latent_sq_dist(copy, target)}
    for k, name in enumerate(JOINTS):
        sums[f"{name}_deg"], sums[f"{name}_tf"] = err[..., k], err_tf[..., k]
        sums[f"{name}_shuf"], sums[f"{name}_floor"] = err_shuf[..., k], floor[..., k]
    return {k: v.sum(0).double().cpu().numpy() for k, v in sums.items()}, len(starts)


def evaluate(model, probe, reader, episodes, normalizer, transform, cfg, args, device, desc) -> dict:
    total = {k: np.zeros(args.horizon) for k in METRICS}
    windows = used = 0
    # donors for the shuffled-action rollouts: the other episodes of the same unseen set
    donors = {int(e): action_blocks(reader, int(e), normalizer, cfg.data.frameskip) for e in episodes}
    for ep in tqdm(episodes, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        rng = np.random.default_rng([cfg.seed, int(ep)])
        sums, w = episode_sums(model, probe, reader, int(ep), normalizer, transform, cfg,
                               args.horizon, args.start_stride, device, donors, rng)
        if w:
            for k in METRICS:
                total[k] += sums[k]
            windows += w
            used += 1
    if not windows:
        raise SystemExit(f"{desc}: no episode is long enough for history {cfg.data.history} + horizon {args.horizon}")
    return {"episodes": used, "windows": windows, "sums": total}


@torch.no_grad()
def decode_videos(model, decoder, reader, episodes, donor_pool, normalizer, transform, cfg, horizon, start, out, device):
    """truth | decode(enc(truth)) | teacher-forced | autoregressive | shuffled, one window per episode."""
    import cv2

    from decoder_acrobot import to_uint8
    from src.utils import save_video

    fs, hist, res = cfg.data.frameskip, cfg.data.history, decoder.res
    out.mkdir(parents=True, exist_ok=True)
    donors = {int(e): action_blocks(reader, int(e), normalizer, fs) for e in donor_pool}
    for ep in episodes:
        ep = int(ep)
        off = int(reader.ep_offset[ep])
        n = (int(reader.ep_len[ep]) - 1) // fs
        h = min(horizon, n + 1 - hist)
        s = min(start, n + 1 - hist - h)
        frames = reader.span("pixels", off, off + n * fs + 1)[::fs]
        act = reader.span("action", off, off + n * fs)
        act = normalizer.normalize("action", np.nan_to_num(act, nan=0.0)).astype(np.float32)
        blocks = torch.from_numpy(act.reshape(n, -1)).to(device)
        emb = encode_frames(model, transform(frames), device)  # (n+1, P, D)

        ctx = emb[s : s + hist]
        ar = model.rollout(ctx[None], blocks[None, s : s + hist + h - 1])[0]  # frames s+hist .. s+hist+h-1
        shuf_act = blocks[None, s : s + hist + h - 1].clone()
        shuf_act[:, hist - 1 :] = donor_segments(donors, ep, 1, h, np.random.default_rng([cfg.seed, ep])).to(device)
        shuf = model.rollout(ctx[None], shuf_act)[0]
        j = torch.arange(s + hist, s + hist + h)
        idx = j[:, None] - hist + torch.arange(hist)
        tf = model.predict(emb[idx], model.encode_action(blocks[idx]))[:, -1]
        ctx_dec = decoder(ctx)

        truth = torch.from_numpy(frames[s : s + hist + h]).to(device).permute(0, 3, 1, 2).float()
        truth = F.interpolate(truth, size=(res, res), mode="area") / 255
        streams = {
            "truth": truth,
            "decode(enc(truth))": decoder(emb[s : s + hist + h]),
            "teacher-forced": torch.cat([ctx_dec, decoder(tf)]),
            "autoregressive": torch.cat([ctx_dec, decoder(ar)]),
            "shuffled actions": torch.cat([ctx_dec, decoder(shuf)]),
        }
        imgs = {k: to_uint8(v).copy() for k, v in streams.items()}  # (T, res, res, 3)
        for k, v in imgs.items():
            for t in range(len(v)):
                tag = "context" if t < hist else f"h={t - hist + 1}"
                for txt, y in ((k, 12), (tag, res - 6)):
                    cv2.putText(v[t], txt, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.concatenate(list(imgs.values()), axis=2)  # (T, res, 4*res, 3)
        save_video(out / f"ep{ep}.mp4", panel, fps=5)

        # still: one row per stream, ~12 evenly spaced steps
        cols = np.unique(np.linspace(0, len(panel) - 1, 12).round().astype(int))
        still = np.concatenate([np.concatenate([v[c] for c in cols], 1) for v in imgs.values()], 0)
        Image.fromarray(still).save(out / f"ep{ep}.png")
    print(f"videos       {out}  (truth | decode(enc(truth)) | teacher-forced | autoregressive | shuffled), "
          f"start frame {start}, {horizon} steps")


def finish(block: dict) -> dict:
    m = {k: v / block["windows"] for k, v in block["sums"].items()}
    m["accum_mse"] = m["latent_mse"] - m["tf_mse"]
    for j in JOINTS:
        m[f"{j}_accum"] = m[f"{j}_deg"] - m[f"{j}_tf"]
    return {"episodes": block["episodes"], "windows": block["windows"], **{k: v.tolist() for k, v in m.items()}}


def plot(res: dict, fs: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(res["horizon"])
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.9))
    axes[0].plot(h, res["latent_mse"], "o-", ms=3, label="autoregressive")
    axes[0].plot(h, res["tf_mse"], "s-", ms=3, label="teacher-forced (real frames)")
    axes[0].plot(h, res["shuf_mse"], "^-", ms=3, label="autoregressive, shuffled actions")
    axes[0].plot(h, res["copy_mse"], "--", color="gray", label="copy last context latent")
    axes[0].set(ylabel="latent MSE", title="Latent error")
    axes[1].plot(h, res["accum_mse"], "o-", ms=3, color="crimson", label="latent MSE")
    axes[1].set(ylabel="autoregressive - teacher-forced", title="Accumulated latent error")
    for ax, j in zip(axes[2:], JOINTS):
        ax.plot(h, res[f"{j}_deg"], "o-", ms=3, label="probe(autoregressive)")
        ax.plot(h, res[f"{j}_tf"], "s-", ms=3, label="probe(teacher-forced)")
        ax.plot(h, res[f"{j}_shuf"], "^-", ms=3, label="probe(shuffled actions)")
        ax.plot(h, res[f"{j}_floor"], "--", color="gray", label="probe(true frame): floor")
        ax.set(ylabel="abs error (deg)", title=f"{j} angle")
    for ax in axes:
        ax.set_xlabel(f"horizon (latent steps, x{fs} rows)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--probe", default=None, help="default: <ckpt dir>/acrobot_probe/probe.pt")
    p.add_argument("--horizon", type=int, default=50, help="latent steps predicted per window")
    p.add_argument("--start-stride", type=int, default=1, help="latent steps between window starts")
    p.add_argument("--extra-h5", nargs="*", default=[], help="more unseen recordings, every episode used")
    p.add_argument("--videos", type=int, default=0, help="decode rollouts of this many held-out episodes")
    p.add_argument("--videos-only", action="store_true", help="skip the metrics, just make the videos")
    p.add_argument("--video-start", type=int, default=0, help="first context frame (latent step) of the video window")
    p.add_argument("--decoder", default=None, help="default: <ckpt dir>/acrobot_decoder/decoder.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="default: <ckpt dir>/acrobot_rollout")
    args = p.parse_args()
    if args.videos_only and not args.videos:
        args.videos = 4

    dec_path = Path(args.decoder or Path(args.ckpt).parent / "acrobot_decoder" / "decoder.pt")
    if args.videos and not dec_path.exists():
        sys.exit(f"no decoder at {dec_path}; train it: python ablations/decoder_acrobot.py --ckpt {args.ckpt}")
    missing = [p for p in args.extra_h5 if not Path(p).exists()]
    if missing:
        sys.exit(f"--extra-h5 not found: {missing}\n"
                 "make it with: python datasets/prepare_mmbench.py --splits val test --out <path>.h5")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device=str(device))
    if cfg.data.obs_key != "pixels" or model.oracle_encoder is not None:
        sys.exit("expects a pixel-only model (preset acrobot_pixels)")
    set_seed(cfg.seed)
    # the same recalibration probe_acrobot.py did, so probe and rollout see identical latents
    windows, _, store, _ = build_datasets(cfg)
    recalibrate_bn(model, windows, device)
    store.close()
    probe = load_probe(args.probe or Path(args.ckpt).parent / "acrobot_probe" / "probe.pt", device)
    transform = ImageTransform(cfg.data.img_size)

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action",))  # the training stats, also for --extra-h5
    _, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "acrobot_rollout"

    if args.videos:
        from decoder_acrobot import load_decoder

        decode_videos(model, load_decoder(dec_path, device), reader, np.sort(val_eps)[: args.videos], val_eps, normalizer,
                      transform, cfg, args.horizon, args.video_start, out / "videos", device)
        if args.videos_only:
            reader.close()
            return

    blocks = {"heldout": evaluate(model, probe, reader, val_eps, normalizer, transform, cfg, args, device, "held-out")}
    reader.close()
    for path in args.extra_h5:
        r = H5Reader(path, rdcc_mb=cfg.data.rdcc_mb)
        blocks[Path(path).stem] = evaluate(model, probe, r, np.arange(r.num_episodes), normalizer, transform,
                                           cfg, args, device, Path(path).stem)
        r.close()

    every = {"episodes": sum(b["episodes"] for b in blocks.values()),
             "windows": sum(b["windows"] for b in blocks.values()),
             "sums": {k: sum(b["sums"][k] for b in blocks.values()) for k in METRICS}}
    res = {"ckpt": str(args.ckpt), "frameskip": cfg.data.frameskip, "history": cfg.data.history,
           "horizon": list(range(1, args.horizon + 1)), **finish(every),
           "per_source": {k: finish(b) for k, b in blocks.items()}}

    save_json(out / "rollout.json", res)
    plot(res, cfg.data.frameskip, out / "rollout.png")

    show = sorted({1, 2, 5, 10, 20, 30, 50, args.horizon} & set(res["horizon"]))
    sources = ", ".join(f"{k}: {b['episodes']} eps" for k, b in res["per_source"].items())
    print(f"\n{res['episodes']} unseen episodes, {res['windows']} windows ({sources})")
    print(f"{'horizon':16s}" + "".join(f"{h:>8d}" for h in show))
    rows = ["latent_mse", "shuf_mse", "tf_mse", "accum_mse", "copy_mse"]
    rows += [f"{j}{s}" for j in JOINTS for s in ("_deg", "_shuf", "_tf", "_accum", "_floor")]
    for k in rows:
        print(f"{k:16s}" + "".join(f"{res[k][h - 1]:8.3f}" for h in show))
    print(f"saved {out / 'rollout.json'} and rollout.png")


if __name__ == "__main__":
    main()
