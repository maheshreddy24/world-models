"""Open-loop latent rollout diagnostics.

    python rollout.py --ckpt <run>/best.pt
    python rollout.py --ckpt <run>/best.pt rollout.horizon=20

No planner and no simulator: the model is given the first `data.history`
observations of a held-out trajectory plus the actions the expert actually
took, and it predicts the rest of the latent trajectory on its own output.
Two questions get answered.

*How far does prediction hold up?*  Latent error is reported per horizon step
against the encoder's own embedding of the true future, next to a "nothing
moves" baseline that repeats the last observed embedding.  A ratio below 1
means the predictor genuinely models dynamics; where the curve crosses 1 is
roughly how far ahead it is worth planning.

*What does it think it is imagining?*  A JEPA has no decoder, so each predicted
latent is matched to its nearest neighbour among the embeddings of the real
frames of that episode, and the retrieved frames are written out as a video
beside the truth.  Sensible retrievals mean the latent kept the information
that distinguishes one moment from another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from src.data import H5Reader, ImageTransform, SequenceDataset, get_normalizer, split_episodes
from src.utils import load_model, make_panel, save_json, save_video, set_seed


@torch.no_grad()
def latent_error_curves(model, loader, cfg, device, horizon: int, history: int) -> dict:
    """Per-horizon latent error of the rollout, and of the static baseline."""
    sums = {k: torch.zeros(horizon, device=device) for k in ("mse", "cosine", "static_mse")}
    count = 0

    for batch in tqdm(loader, desc="rollout", dynamic_ncols=True, disable=not sys.stderr.isatty()):
        obs = batch[cfg.data.obs_key].to(device)
        actions = batch["action"].to(device)

        emb = model.encode(obs)  # (B, T, D) ground-truth latents
        # Actions 0..T-2 carry steps 0..T-1, so every prediction has a target.
        preds = model.rollout(emb[:, :history], actions[:, :-1])  # (B, horizon, D)
        target = emb[:, history:]
        static = emb[:, history - 1 : history].expand_as(target)  # "nothing moves"

        sums["mse"] += (preds - target).pow(2).sum(-1).sum(0)
        sums["static_mse"] += (static - target).pow(2).sum(-1).sum(0)
        sums["cosine"] += torch.cosine_similarity(preds, target, dim=-1).sum(0)
        count += obs.size(0)

    mse = (sums["mse"] / count).cpu().numpy()
    static = (sums["static_mse"] / count).cpu().numpy()
    return {
        "horizon": np.arange(1, horizon + 1).tolist(),
        "latent_mse": mse.tolist(),
        "static_mse": static.tolist(),
        # < 1 means the predictor beats assuming the world is frozen
        "mse_vs_static": (mse / np.maximum(static, 1e-8)).tolist(),
        "cosine": (sums["cosine"] / count).cpu().numpy().tolist(),
        "num_sequences": count,
    }


@torch.no_grad()
def imagination_video(model, reader, cfg, device, episode: int, horizon: int, history: int, normalizer, transform):
    """Decode a latent rollout by nearest-neighbour retrieval over real frames."""
    fs = cfg.data.frameskip
    start = int(reader.ep_offset[episode])
    length = int(reader.ep_len[episode]) - 1
    n_steps = min(history + horizon, length // fs)

    frames = reader.span("pixels", start, start + n_steps * fs)[::fs]  # (n, H, W, 3)
    raw_actions = reader.span("action", start, start + n_steps * fs)
    raw_actions = normalizer.normalize("action", np.nan_to_num(raw_actions, nan=0.0))
    actions = torch.from_numpy(raw_actions.astype(np.float32).reshape(1, n_steps, -1)).to(device)

    pixels = transform(frames).unsqueeze(0).to(device)  # (1, n, 3, H, W)
    emb = model.encode(pixels)  # (1, n, D)

    preds = model.rollout(emb[:, :history], actions[:, :-1])[0]  # (n - history, D)
    bank = emb[0]  # every real frame of the episode is a retrieval candidate

    # Nearest neighbour in latent space = the frame the model thinks it is at.
    distances = torch.cdist(preds, bank)  # (n - history, n)
    retrieved = distances.argmin(dim=1).cpu().numpy()

    truth = frames[history:]
    return make_panel(truth, frames[retrieved]), retrieved


def plot_curves(curves: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = curves["horizon"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(steps, curves["latent_mse"], "o-", label="rollout")
    axes[0].plot(steps, curves["static_mse"], "s--", color="gray", label="static baseline")
    axes[0].set(xlabel="horizon (latent steps)", ylabel="latent MSE", title="Prediction error")
    axes[0].legend(frameon=False)

    axes[1].plot(steps, curves["mse_vs_static"], "o-", color="crimson")
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set(xlabel="horizon (latent steps)", ylabel="rollout / static", title="Skill vs. doing nothing")

    axes[2].plot(steps, curves["cosine"], "o-", color="seagreen")
    axes[2].set(xlabel="horizon (latent steps)", ylabel="cosine similarity", title="Direction agreement")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    cfg, args = get_config(argv, ckpt=None, out=None)
    if not args.ckpt:
        raise SystemExit("pass --ckpt <path>")
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model, train_cfg = load_model(args.ckpt, device=str(device))
    cfg.data, cfg.model = train_cfg.data, train_cfg.model

    history, horizon = cfg.data.history, cfg.rollout.horizon
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action", "observation"))
    transform = ImageTransform(cfg.data.img_size)

    _, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)

    dataset = SequenceDataset(
        reader=reader,
        episodes=val_eps,
        seq_len=history + horizon,
        frameskip=cfg.data.frameskip,
        keys=(cfg.data.obs_key,),
        normalizer=normalizer,
        image_transform=transform,
        stride=cfg.data.frameskip,
    )
    picks = np.random.default_rng(cfg.seed).choice(
        len(dataset), size=min(cfg.rollout.num_sequences, len(dataset)), replace=False
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, picks.tolist()),
        batch_size=cfg.rollout.batch_size,
        num_workers=min(4, cfg.optim.num_workers),
    )

    curves = latent_error_curves(model, loader, cfg, device, horizon, history)
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "rollout"
    save_json(out_dir / "curves.json", curves)
    plot_curves(curves, out_dir / "latent_rollout.png")

    print(f"\nhorizon      {'  '.join(f'{h:>7}' for h in curves['horizon'][:8])}")
    print(f"latent mse   {'  '.join(f'{v:7.3f}' for v in curves['latent_mse'][:8])}")
    print(f"vs static    {'  '.join(f'{v:7.3f}' for v in curves['mse_vs_static'][:8])}")
    print(f"cosine       {'  '.join(f'{v:7.3f}' for v in curves['cosine'][:8])}")

    if cfg.rollout.retrieval_video and cfg.data.obs_key == "pixels":
        for i, episode in enumerate(val_eps[: cfg.rollout.num_videos]):
            panel, retrieved = imagination_video(
                model, reader, cfg, device, int(episode), horizon, history, normalizer, transform
            )
            save_video(out_dir / f"imagination_ep{int(episode)}.mp4", panel, fps=4)
        print(f"videos       {out_dir}  (truth | nearest-neighbour of the predicted latent)")

    print(f"curves       {out_dir / 'latent_rollout.png'}")
    reader.close()


if __name__ == "__main__":
    main()
