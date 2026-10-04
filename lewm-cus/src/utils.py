"""Seeding, optimisation schedules, checkpoints, logging and video helpers."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def count_params(model: torch.nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}


# --------------------------------------------------------------------------- #
#  Optimisation
# --------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """AdamW with weight decay switched off for norms, biases and embeddings.

    Decaying a LayerNorm gain or a position embedding toward zero fights the
    thing it is there to do, so those tensors get their own group.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".pos_embedding") or "pos_embed" in name or "cls_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.optim.lr, betas=tuple(cfg.optim.betas))


class WarmupCosine:
    """Linear warmup then cosine decay, stepped once per optimiser step."""

    def __init__(self, optimizer, base_lr: float, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.01):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.min_lr = base_lr * min_lr_scale
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]


def amp_dtype(name: str):
    """Map the config string to a torch dtype (None disables autocast)."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[name]


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
class Meters:
    """Running means, reset between logging windows."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, values: dict, n: int = 1) -> None:
        for key, value in values.items():
            value = float(value.detach()) if torch.is_tensor(value) else float(value)
            self.sums[key] = self.sums.get(key, 0.0) + value * n
            self.counts[key] = self.counts.get(key, 0) + n

    def average(self) -> dict[str, float]:
        return {k: self.sums[k] / max(1, self.counts[k]) for k in self.sums}

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


class Logger:
    """Writes metrics to stdout, a CSV file, a plain-text log, and optionally wandb.

    The CSV is for plotting, `train.log` is for reading: it keeps every line
    that reached stdout, timestamped, so a detached run can be inspected after
    the fact without a wandb round-trip.
    """

    def __init__(self, run_dir: Path, cfg, filename: str = "metrics.csv", log_name: str = "train.log"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename
        self.log_path = self.run_dir / log_name
        self.fields: list[str] | None = None
        self.wandb = None

        # Appended, not truncated, so a resumed run keeps its history.
        self.write(f"=== run {cfg.run_name} ({cfg.exp_id}) started {_now()} ===")
        self.write(f"config: {json.dumps(cfg.to_dict(), default=str)}")
        if cfg.wandb.enabled:
            import wandb

            self.wandb = wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name or cfg.run_name,
                mode=cfg.wandb.mode,
                dir=str(self.run_dir),
                config=cfg.to_dict(),
            )

    def log(self, metrics: dict, step: int, prefix: str = "", stdout: bool = True) -> None:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        row = {"step": step, **{k: _scalar(v) for k, v in metrics.items()}}

        if self.fields is None:
            self.fields = list(row)
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        with self.path.open("a", newline="") as f:
            # Unseen keys would break the header contract; drop them from the CSV
            # (they still reach wandb and stdout).
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)

        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

        body = "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        line = f"[{step:>7}] {body}"
        self.write(line, stamp=True)
        if stdout:
            print(line, flush=True)

    def write(self, text: str, stamp: bool = True) -> None:
        """Append a free-form line to `train.log` (never to the CSV or wandb)."""
        prefix = f"{_now()} " if stamp else ""
        with self.log_path.open("a") as f:
            f.write(f"{prefix}{text}\n")

    def finish(self) -> None:
        self.write(f"=== finished {_now()} ===")
        if self.wandb is not None:
            self.wandb.finish()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _scalar(value):
    return float(value.detach()) if torch.is_tensor(value) else value


def _fmt(value) -> str:
    value = _scalar(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# --------------------------------------------------------------------------- #
#  Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, model, cfg, optimizer=None, scheduler=None, epoch: int = 0, step: int = 0, extra: dict | None = None) -> None:
    """Write weights plus everything needed to rebuild the model later.

    The config travels with the weights, so eval and rollout never have to be
    told again how the model was built.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "step": step,
        **(extra or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a killed job never leaves a half-written ckpt


@torch.no_grad()
def recalibrate_bn(model, dataset, device, batches: int = 40, batch_size: int = 64, seed: int = 0,
                   num_workers: int = 4) -> int:
    """Re-estimate every BatchNorm's running statistics, with dropout off, on `dataset`.

    The projector heads use BatchNorm, whose running statistics accumulate in
    training mode with dropout on. In eval mode (dropout off) they no longer
    match the activations and the embeddings come out inflated: on acrobot,
    eval-mode prediction loss was ~9x the train-mode loss on the same windows.
    Recomputing them as a plain average over `batches` x `batch_size` training
    windows, with the rest of the model in eval mode, fixes that.

    Leaves the model in eval mode. Returns the number of BatchNorm layers reset
    (0: nothing to do).
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        model.eval()
        return 0
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the recalibration batches
    model.eval()
    for m in bns:
        m.train()

    n = min(batches * batch_size, len(dataset))
    picks = np.random.default_rng(seed).choice(len(dataset), size=n, replace=False)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, picks.tolist()),
                                         batch_size=batch_size, drop_last=n >= batch_size, num_workers=num_workers)
    device = torch.device(device)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            model.loss(batch)  # encode + predict: passes every BatchNorm in the model

    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.eval()
    return len(bns)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model(path: str | Path, device: str = "cuda", random_init: bool = False):
    """Rebuild a trained model from a checkpoint (eval mode, no gradients).

    `random_init` builds the same architecture but keeps its fresh weights
    (the untrained-backbone baseline). Eval-mode BatchNorm statistics still
    need `recalibrate_bn` before the latents can be trusted.

    Returns:
        (model in eval mode on `device`, the config it was trained with)
    """
    from config import Config
    from src.models import build_model

    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg, action_dim=ckpt["action_dim"], state_dim=ckpt.get("state_dim"))
    if not random_init:
        model.load_state_dict(ckpt["model"])
    model.to(device).eval().requires_grad_(False)
    return model, cfg


# --------------------------------------------------------------------------- #
#  Output helpers
# --------------------------------------------------------------------------- #
def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def save_video(path: str | Path, frames, fps: int = 10) -> None:
    """Write (T, H, W, 3) uint8 frames to an mp4."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1, quality=8)
