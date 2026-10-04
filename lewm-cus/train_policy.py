"""Train the goal-conditioned diffusion policy on top of a frozen LeWM.

    python train_policy.py --ckpt checkpoints/<exp>/epoch_007.pt
    python train_policy.py --preset scene_policy --ckpt checkpoints/<scene exp>/epoch_019.pt
    python train_policy.py --ckpt <...> policy.use_proprio=false   # no proprio token
    python train_policy.py --ckpt <...> policy.use_contact=true    # + a gripper-contact token
    python train_policy.py --ckpt <...> policy.goal=latent policy.proprio=ee   # goal frame + ee proprio
    python train_policy.py --ckpt <...> policy.random_backbone=true   # baseline: untrained world model
    python train_policy.py --ckpt <...> optim.batch_size=64 optim.epochs=5

The world model is only an encoder here.  Each sample is the current frame, a
goal further along the same episode, the expert's next `policy.chunk` raw
actions and the proprio at the current frame; the policy learns to denoise
those actions given c = [current latent, proprio, goal (, contact)].  On the
cube the goal defaults to the cube's position at the goal row
(`policy.goal=cube_xyz`, 3 numbers) and the proprio to joint pos + joint vel +
gripper opening (`policy.proprio=arm`, 13).

Where the goals come from is the one thing the two recordings do differently,
and it lives entirely in `src.data`:

    cube (h5)     goals `policy.goal_offset_min..max` raw steps ahead, and the
                  train/val episode split of the world model
    scene         goals drawn from the mined v1 pairs in `train_pairs.npz`
                  (`scene.p_task`) or hindsight, validated on the held-out
                  recording

Data, model and the split all come from the world-model checkpoint, so the
policy never trains on what evaluation draws from.  Optimisation defaults come
from the `policy` / `scene_policy` presets in config.py.

Watch `val/action_mse`: the error of sampled chunks against the expert's, in
raw env action units (actions live in [-1, 1]).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from src.data import build_policy_datasets
from src.planner import build_policy
from src.utils import (
    Logger,
    Meters,
    WarmupCosine,
    amp_dtype,
    build_optimizer,
    count_params,
    load_checkpoint,
    load_model,
    save_checkpoint,
    set_seed,
)
from train import make_loaders

@torch.no_grad()
def update_ema(ema: torch.nn.Module, policy: torch.nn.Module, decay: float) -> None:
    for e, p in zip(ema.parameters(), policy.parameters()):
        e.lerp_(p, 1 - decay)


def run_epoch(policy, ema, lewm, loader, cfg, device, autocast_dtype, optimizer=None, scheduler=None,
              logger=None, epoch=0, global_step=0, max_steps=None):
    """One pass. Training (updates `policy` and `ema`) when `optimizer` is given, else validation of `ema`."""
    training = optimizer is not None
    policy.train(training)
    meters, window = Meters(), Meters()
    stage = "train" if training else "val"
    started = time.time()
    use_amp = autocast_dtype is not None

    progress = tqdm(
        loader, total=max_steps or len(loader), desc=f"{stage} {epoch}", leave=False,
        dynamic_ncols=True, disable=not sys.stderr.isatty(),
    )
    for step, batch in enumerate(progress):
        if max_steps and step >= max_steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
            z = lewm.encode(batch[cfg.data.obs_key])  # (B, 1 | 2, P, D): current frame (+ goal frame)
        z_cur = z[:, 0].float()
        goal = z[:, 1].float() if cfg.policy.goal == "latent" else batch["goal"]  # (B, P, D) | (B, 3)
        cond = {"proprio": batch.get("proprio"), "contact": batch.get("contact")}

        if training:
            if cfg.policy.goal_noise > 0:
                goal = goal + cfg.policy.goal_noise * torch.randn_like(goal)
            with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
                loss = policy.loss(batch["action"], z_cur, goal, **cond)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip)
            optimizer.step()
            update_ema(ema, policy, cfg.policy.ema_decay)
            stats = {"loss": loss, "grad_norm": grad_norm, "lr": scheduler.step()}
            global_step += 1
        else:
            with torch.no_grad():
                loss = ema.loss(batch["action"], z_cur, goal, **cond)
                sampled = ema.sample(z_cur, goal, **cond, steps=cfg.plan.ddim_steps)
            stats = {"loss": loss, "action_mse": (sampled - batch["action"]).pow(2).mean()}

        stats = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in stats.items()}
        meters.update(stats)
        window.update(stats)
        progress.set_postfix(loss=f"{stats['loss']:.4f}")

        if training and logger and global_step % cfg.optim.log_every == 0:
            metrics = window.average()
            metrics["samples_per_sec"] = (step + 1) * cfg.optim.batch_size / (time.time() - started)
            logger.log(metrics, step=global_step, prefix="train/", stdout=False)
            window.reset()

    progress.close()
    return meters.average(), global_step


def main(argv=None) -> None:
    cfg, args = get_config(argv, default_preset="policy", ckpt=None)
    if not args.ckpt:
        raise SystemExit("pass --ckpt <world model checkpoint> to train the policy on")
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    autocast = amp_dtype(cfg.optim.amp_dtype) if device.type == "cuda" else None

    # -- frozen world model: it decides what an observation and a latent are -- #
    lewm, lewm_cfg = load_model(args.ckpt, device=str(device), random_init=cfg.policy.random_backbone)
    cfg.data, cfg.model = lewm_cfg.data, lewm_cfg.model
    cfg.validate()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    lewm_ckpt = Path(args.ckpt).resolve()
    if cfg.policy.random_backbone:
        # eval.py rebuilds the world model from the path the policy records, so
        # the untrained one is saved into this run and recorded instead of --ckpt.
        source = load_checkpoint(args.ckpt)
        lewm_ckpt = run_dir / "random_world_model.pt"
        save_checkpoint(
            lewm_ckpt, lewm, lewm_cfg,
            extra={
                "action_dim": source.get("action_dim", 5),
                "state_dim": source.get("state_dim"),
                "random_init_of": str(Path(args.ckpt).resolve()),
            },
        )

    # -- data -------------------------------------------------------------- #
    train_set, val_set, store, stats, action_dim = build_policy_datasets(cfg)
    train_loader, val_loader = make_loaders(cfg, train_set, val_set)
    goals = (
        f"{cfg.scene.p_task:.0%} mined v1 pairs, rest hindsight "
        f"{cfg.scene.hindsight_min_gap}-{cfg.scene.hindsight_max_gap} steps ahead"
        if cfg.data.backend == "scene"
        else f"{cfg.policy.goal} goals {cfg.policy.goal_offset_min}-{cfg.policy.goal_offset_max} steps ahead"
    )
    print(
        f"data  | {cfg.data.backend} | {len(train_set):,} train / {len(val_set):,} val samples"
        f" | chunk {cfg.policy.chunk} raw actions | {goals}"
    )

    # -- policy ------------------------------------------------------------ #
    num_tokens = lewm.predictor.num_patches
    policy = build_policy(cfg, action_dim=action_dim, num_tokens=num_tokens)
    policy.set_stats(**stats)
    store.close()  # dataloader workers open their own handle; never share one across a fork
    policy.to(device)
    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    print(
        f"model | diffusion policy {count_params(policy)['total_M']:.1f}M params on "
        f"{num_tokens}x{cfg.model.embed_dim} latents from {args.ckpt}"
        + (" (architecture only: random-init weights)" if cfg.policy.random_backbone else "")
        + f", goal {cfg.policy.goal}, proprio {cfg.policy.proprio if cfg.policy.use_proprio else 'off'}"
        f", contact {'on' if cfg.policy.use_contact else 'off'} -> {run_dir}"
    )

    optimizer = build_optimizer(policy, cfg)
    steps_per_epoch = cfg.optim.steps_per_epoch or len(train_loader)
    scheduler = WarmupCosine(
        optimizer,
        base_lr=cfg.optim.lr,
        warmup_steps=int(cfg.optim.warmup_epochs * steps_per_epoch),
        total_steps=cfg.optim.epochs * steps_per_epoch,
        min_lr_scale=cfg.optim.min_lr_scale,
    )

    logger = Logger(run_dir, cfg)
    best_val, best_epoch, global_step, epoch = float("inf"), -1, 0, 0

    def checkpoint(epoch: int) -> Path:
        """The EMA weights, which is what eval.py acts with."""
        path = run_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            path, ema, cfg, epoch=epoch, step=global_step,
            extra={
                "lewm_ckpt": str(lewm_ckpt),
                "action_dim": action_dim,
                "num_tokens": num_tokens,
                "best_val": best_val,
                "best_epoch": best_epoch,
            },
        )
        return path

    try:
        for epoch in range(cfg.optim.epochs):
            train_stats, global_step = run_epoch(
                policy, ema, lewm, train_loader, cfg, device, autocast,
                optimizer=optimizer, scheduler=scheduler, logger=logger,
                epoch=epoch, global_step=global_step, max_steps=cfg.optim.steps_per_epoch,
            )
            logger.log({**train_stats, "epoch": epoch}, step=global_step, prefix="train_epoch/")

            val_stats, _ = run_epoch(
                policy, ema, lewm, val_loader, cfg, device, autocast,
                epoch=epoch, max_steps=cfg.optim.val_steps,
            )
            logger.log({**val_stats, "epoch": epoch}, step=global_step, prefix="val/")

            if val_stats["action_mse"] < best_val:
                best_val, best_epoch = val_stats["action_mse"], epoch
            if (epoch + 1) % cfg.optim.ckpt_every == 0:
                path = checkpoint(epoch)
                logger.write(
                    f"epoch {epoch:3d} | val action_mse {val_stats['action_mse']:.5f} "
                    f"| best {best_val:.5f} @ epoch {best_epoch} | saved {path.name}"
                )
    except KeyboardInterrupt:
        print("\ninterrupted — saving before exit")
        checkpoint(epoch)
    finally:
        logger.finish()
        store.close()

    print(
        f"done  | best val action_mse {best_val:.5f} at epoch {best_epoch} "
        f"(epoch_{best_epoch:03d}.pt) | checkpoints in {run_dir}"
    )


if __name__ == "__main__":
    main()
