"""Train LeWM on OGBench trajectories.

    python train.py                              # pixels, default settings
    python train.py --preset cube_oracle         # oracle-state ablation
    python train.py --preset cube_dino           # frozen DINOv2 patches, no SIGReg (DINO-WM)
    python train.py --preset debug               # 20-second smoke test
    python train.py optim.lr=1e-4 model.depth=8  # override anything from config.py
    python train.py --resume                     # continue the last checkpoint

The objective is to predict the next embedding and, when the encoder is
trainable, keep the embedding distribution Gaussian so it cannot collapse
(`loss.use_sigreg`).  Watch `emb_std` — if it heads for zero the representation
is collapsing and `loss.sigreg_weight` is too low; `pred_vs_static` below 1
means the predictor beats assuming nothing moves.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from src.data import build_datasets
from src.models import SIGReg, build_model
from src.utils import (
    Logger,
    Meters,
    WarmupCosine,
    amp_dtype,
    build_optimizer,
    count_params,
    load_checkpoint,
    recalibrate_bn,
    save_checkpoint,
    set_seed,
)


def make_loaders(cfg, train_set, val_set):
    """Loaders whose epoch length is fixed by `optim.steps_per_epoch`.

    The dataset holds millions of overlapping windows, so a literal pass over it
    would take hours and produce one checkpoint. Sampling a fixed number of
    random windows per epoch keeps checkpoints and validation frequent.
    """
    common = dict(
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.optim.num_workers,
        pin_memory=cfg.optim.pin_memory,
    )
    if cfg.optim.num_workers > 0:
        common.update(
            persistent_workers=cfg.optim.persistent_workers,
            prefetch_factor=cfg.optim.prefetch_factor,
        )

    sampler = None
    if cfg.optim.steps_per_epoch:
        sampler = RandomSampler(
            train_set, replacement=True, num_samples=cfg.optim.steps_per_epoch * cfg.optim.batch_size
        )
    # Training drops the ragged tail so every step sees a full batch; validation
    # keeps it, or a val split smaller than one batch would yield no batches at
    # all and leave the epoch with nothing to select a checkpoint on.
    train_loader = DataLoader(
        train_set, shuffle=sampler is None, sampler=sampler, drop_last=True, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """Newest `epoch_*.pt` across every experiment directory under `out_dir`.

    Checkpoints are per-epoch now, so "resume" means the highest epoch of the
    most recent experiment rather than a fixed `last.pt`.
    """
    runs = sorted((p for p in Path(out_dir).glob("exp_*") if p.is_dir()), reverse=True)
    for run in runs:
        ckpts = sorted(run.glob("epoch_*.pt"))
        if ckpts:
            return ckpts[-1]
    return None


def run_epoch(model, sigreg, loader, cfg, device, autocast_dtype, optimizer=None, scheduler=None, logger=None, epoch=0, global_step=0, max_steps=None):
    """One pass. Training when `optimizer` is given, validation otherwise."""
    training = optimizer is not None
    model.train(training)
    meters, window = Meters(), Meters()
    stage = "train" if training else "val"
    total = max_steps or len(loader)
    started = time.time()

    progress = tqdm(
        loader, total=total, desc=f"{stage} {epoch}", leave=False,
        dynamic_ncols=True, disable=not sys.stderr.isatty(),
    )
    for step, batch in enumerate(progress):
        if max_steps and step >= max_steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            out = model.loss(batch, sigreg, cfg.loss.sigreg_weight)

        if training:
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            lr = scheduler.step()
            global_step += 1
            out = {**out, "grad_norm": grad_norm, "lr": lr}

        stats = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in out.items()}
        meters.update(stats)
        window.update(stats)
        progress.set_postfix(loss=f"{stats['loss']:.4f}", emb_std=f"{stats['emb_std']:.3f}")

        if training and logger and global_step % cfg.optim.log_every == 0:
            metrics = window.average()
            metrics["samples_per_sec"] = (step + 1) * cfg.optim.batch_size / (time.time() - started)
            logger.log(metrics, step=global_step, prefix="train/", stdout=False)
            window.reset()

    progress.close()
    return meters.average(), global_step


def main(argv=None) -> None:
    cfg, args = get_config(argv, resume=False, ckpt=None)
    set_seed(cfg.seed)

    # A resumed run continues inside the experiment directory it started in,
    # so this has to be settled before `cfg.run_dir` is read below.
    resume_from = None
    if args.resume:
        resume_from = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(cfg.out_dir)
        if resume_from is None:
            raise SystemExit(f"--resume found no epoch_*.pt under {cfg.out_dir}")
        if not resume_from.exists():
            raise SystemExit(f"--resume asked for {resume_from}, which does not exist")
        cfg.exp_id = resume_from.parent.name

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    autocast = amp_dtype(cfg.optim.amp_dtype) if device.type == "cuda" else None
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    # -- data ------------------------------------------------------------- #
    train_set, val_set, reader, normalizer = build_datasets(cfg)
    train_loader, val_loader = make_loaders(cfg, train_set, val_set)
    action_dim = normalizer.dim("action")
    state_dim = normalizer.dim(cfg.data.obs_key) if cfg.data.obs_key in normalizer else None
    print(
        f"data  | {len(train_set):,} train windows / {len(val_set):,} val windows"
        f" | obs={cfg.data.obs_key} action_dim={action_dim} x frameskip={cfg.data.frameskip}"
    )

    # -- model ------------------------------------------------------------ #
    model = build_model(cfg, action_dim=action_dim, state_dim=state_dim).to(device)
    sigreg = SIGReg(cfg.loss.sigreg_knots, cfg.loss.sigreg_proj).to(device) if cfg.loss.use_sigreg else None
    if cfg.compile:
        model = torch.compile(model)
    params = count_params(model)
    print(
        f"model | {cfg.model.encoder} encoder{' (frozen)' if cfg.model.freeze_encoder else ''}"
        f", sigreg {'on' if sigreg is not None else 'off'}"
        f", {params['total_M']:.1f}M params ({params['trainable_M']:.1f}M trainable) -> {run_dir}"
    )

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = cfg.optim.steps_per_epoch or len(train_loader)
    scheduler = WarmupCosine(
        optimizer,
        base_lr=cfg.optim.lr,
        warmup_steps=int(cfg.optim.warmup_epochs * steps_per_epoch),
        total_steps=cfg.optim.epochs * steps_per_epoch,
        min_lr_scale=cfg.optim.min_lr_scale,
    )

    # -- resume ----------------------------------------------------------- #
    start_epoch, global_step = 0, 0
    if resume_from is not None:
        ckpt = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["step"]
        print(f"resume| from {resume_from} at epoch {start_epoch}")

    logger = Logger(run_dir, cfg)
    # Model selection uses `pred_vs_static`, not the raw prediction loss: a
    # collapsed encoder maps everything to one point, which makes pred_loss
    # ~0 and would otherwise be crowned the best checkpoint. The ratio is
    # scale-invariant, so collapse scores badly instead of perfectly.
    metric = cfg.optim.select_metric
    best_val, best_epoch = float("inf"), -1

    def checkpoint(epoch: int) -> Path:
        """One file per epoch — nothing is ever overwritten.

        `best_val`/`best_epoch` still ride along in every checkpoint so the
        best-scoring epoch is identifiable after the fact without re-running
        validation, but selection no longer decides what gets written.
        """
        path = run_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            path, model, cfg, optimizer, scheduler,
            epoch=epoch, step=global_step,
            extra={
                "action_dim": action_dim,
                "state_dim": state_dim,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "select_metric": metric,
            },
        )
        return path

    # -- loop ------------------------------------------------------------- #
    try:
        for epoch in range(start_epoch, cfg.optim.epochs):
            train_stats, global_step = run_epoch(
                model, sigreg, train_loader, cfg, device, autocast,
                optimizer=optimizer, scheduler=scheduler, logger=logger,
                epoch=epoch, global_step=global_step, max_steps=cfg.optim.steps_per_epoch,
            )
            logger.log({**train_stats, "epoch": epoch}, step=global_step, prefix="train_epoch/")

            # BatchNorm running stats collected with dropout on do not fit eval mode;
            # recompute them so val/* and the saved checkpoint reflect the model.
            recalibrate_bn(model, train_set, device, num_workers=min(4, cfg.optim.num_workers))
            with torch.no_grad():
                val_stats, _ = run_epoch(
                    model, sigreg, val_loader, cfg, device, autocast,
                    epoch=epoch, max_steps=cfg.optim.val_steps,
                )
            logger.log({**val_stats, "epoch": epoch}, step=global_step, prefix="val/")

            if metric not in val_stats:
                raise SystemExit(
                    f"validation produced no {metric!r} to select on. "
                    + (
                        f"The val split ran {len(val_loader)} batches; "
                        "check optim.select_metric against what model.loss returns: "
                        f"{sorted(val_stats)}"
                        if val_stats
                        else f"The val split yielded no batches at all "
                        f"({len(val_set):,} windows, batch_size={cfg.optim.batch_size}) — "
                        "raise data.val_episodes or lower optim.batch_size."
                    )
                )

            if val_stats[metric] < best_val:
                best_val, best_epoch = val_stats[metric], epoch

            if (epoch + 1) % cfg.optim.ckpt_every == 0:
                path = checkpoint(epoch)
                logger.write(
                    f"epoch {epoch:3d} | val {metric} {val_stats[metric]:.5f} "
                    f"| best {best_val:.5f} @ epoch {best_epoch} | saved {path.name}"
                )
    except KeyboardInterrupt:
        print("\ninterrupted — saving before exit")
        checkpoint(epoch)
    finally:
        logger.finish()
        reader.close()

    print(
        f"done  | best val {metric} {best_val:.5f} at epoch {best_epoch} "
        f"(epoch_{best_epoch:03d}.pt) | checkpoints in {run_dir}"
    )


if __name__ == "__main__":
    main()
