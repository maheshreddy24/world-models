"""Every knob in the project, as plain dataclasses.

Three ways to change a setting:

    1. edit the default here,
    2. pick a preset:              python train.py --preset dino
    3. override on the CLI:        python train.py data.task=cartpole-swingup optim.lr=1e-4

CLI overrides are dotted paths into the config tree, cast to the type of the
field they replace (`optim.epochs=50` is an int, `wandb.enabled=true` a bool).

A checkpoint stores its config as a dict; `Config.from_dict` ignores keys it
does not know, so checkpoints written before a field was removed still load.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints


def _find_home() -> Path:
    """The data store: `$STABLEWM_HOME`, else the nearest `.stable_worldmodel` with a `datasets/`."""
    if "STABLEWM_HOME" in os.environ:
        return Path(os.environ["STABLEWM_HOME"]).expanduser()
    here = Path(__file__).resolve().parent
    candidates = [p / ".stable_worldmodel" for p in (here, *here.parents[:2])]
    candidates.append(Path.home() / ".stable_worldmodel")
    for path in candidates:
        if (path / "datasets").is_dir():
            return path
    return candidates[-1]


HOME = _find_home()

# MMBench2 tasks with a converted recording and a probe spec (src/tasks.py).
TASKS = ("acrobot-swingup", "cartpole-swingup", "pendulum-swingup", "reacher-easy")


def mmbench_h5(task: str, heldout: bool = False) -> Path:
    """Where datasets/prepare_mmbench.py writes a task's training (or val + test) table."""
    return HOME / "datasets" / "mmbench" / f"{task}{'-valtest' if heldout else ''}.h5"


# --------------------------------------------------------------------------- #
#  Sections
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """The recording and how it is cut into training sequences.

    Every task is an MMBench2 DMControl recording converted by
    `datasets/prepare_mmbench.py`: 224x224 frames, 260 episodes of 501 rows,
    each row 2 simulator steps.
    """

    task: str = "acrobot-swingup"  # one of config.TASKS
    h5_path: str | None = None  # None: mmbench_h5(task)

    # One latent step spans `frameskip` rows. Its action is the flattened block
    # of the `frameskip` raw actions it covers (action_dim * frameskip numbers).
    frameskip: int = 5
    history: int = 3  # latent frames the predictor sees
    num_preds: int = 1  # how far past the history a training sequence reaches

    img_size: int = 224
    obs_key: str = "pixels"  # "pixels", or "observation" for the oracle state

    # Rows between window starts. Episodes are 501 rows, so 2 still covers
    # every phase of the swing.
    train_stride: int = 2
    val_stride: int = 5

    max_episodes: int | None = None  # cap for quick experiments
    val_episodes: int = 26  # held out by episode (10% of 260), never by row

    # z-score statistics, computed once and cached next to the h5 file
    stats_path: str | None = None
    stats_max_rows: int = 200_000

    rdcc_mb: int = 256  # HDF5 chunk cache per dataloader worker

    @property
    def seq_len(self) -> int:
        """Latent steps per training sequence."""
        return self.history + self.num_preds


@dataclass
class ModelConfig:
    """LeWM = encoder + action encoder + autoregressive latent predictor."""

    # "vit"  pixels, ViT trained from scratch (LeWM)
    # "dino" pixels, frozen pretrained DINOv2 patch features (DINO-WM)
    # "mlp"  the oracle state
    encoder: str = "vit"
    freeze_encoder: bool = False  # no gradients, always in eval mode

    vit_name: str = "vit_tiny_patch16_224"
    vit_pretrained: bool = False
    vit_pool: str = "cls"  # "cls" or "mean"

    # Frames are resized to dino_img_size inside the encoder.
    dino_name: str = "dinov2_vits14"  # torch.hub name; 384 wide
    dino_tokens: str = "patch"  # "patch": a 14x14 token grid | "cls": one vector
    dino_img_size: int = 196  # multiple of the 14px patch

    mlp_hidden: int = 512  # "mlp" encoder only

    embed_dim: int = 192  # = encoder width (vit-tiny 192, dinov2_vits14 384)

    # action encoder
    act_smooth_dim: int = 64
    act_mlp_scale: int = 4

    # predictor (widths from the LeWM reference, independent of embed_dim)
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1

    # MLP heads after the encoder and after the predictor. Must be off without
    # SIGReg: a trainable head can collapse even on top of a frozen encoder.
    use_projector: bool = True
    proj_hidden: int = 2048
    # "batch" as in the reference. Its running statistics do not match eval
    # mode out of the box; see src.utils.recalibrate_bn.
    proj_norm: str = "batch"


@dataclass
class OptimConfig:
    epochs: int = 100  # length of the LR schedule (warmup + cosine)
    # Stop after this epoch index while keeping the `epochs`-long schedule, e.g.
    # 6: train epochs 0..6 at the LR a 100-epoch run has there. None: run them all.
    stop_epoch: int | None = None
    batch_size: int = 128
    lr: float = 5e-5
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: float = 2.0
    min_lr_scale: float = 0.01  # final lr = lr * min_lr_scale
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"  # "bf16" | "fp16" | "off"

    num_workers: int = 12
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True

    # An "epoch" is a fixed number of random windows, so checkpoints come often.
    steps_per_epoch: int | None = 2000
    val_steps: int | None = 100
    log_every: int = 20
    ckpt_every: int = 1  # epochs
    # Validation metric tracked as "best". Not the raw pred_loss: a collapsed
    # encoder predicts a constant perfectly.
    select_metric: str = "pred_vs_static"


@dataclass
class LossConfig:
    """Next-embedding prediction, plus SIGReg when the target could collapse."""

    use_sigreg: bool = True
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_proj: int = 1024


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "lewm-cus"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"


@dataclass
class Config:
    run_name: str = "lewm"  # set to <task>_<preset> unless overridden
    out_dir: str = str(Path(__file__).resolve().parent / "checkpoints")
    # exp_<unix time>, stamped once per launch by `get_config` so no run
    # overwrites another; carried in every checkpoint so a resume reuses it.
    exp_id: str | None = None
    seed: int = 3072
    device: str = "cuda"
    compile: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / (self.exp_id or self.run_name)

    def validate(self) -> None:
        if self.data.task not in TASKS:
            raise ValueError(f"data.task must be one of {TASKS}, got {self.data.task!r}")
        if self.model.encoder == "mlp" and self.data.obs_key == "pixels":
            raise ValueError("the mlp encoder needs data.obs_key='observation'")
        if self.model.encoder in ("vit", "dino") and self.data.obs_key != "pixels":
            raise ValueError(f"the {self.model.encoder} encoder needs data.obs_key='pixels'")
        if not self.loss.use_sigreg and (not self.model.freeze_encoder or self.model.use_projector):
            raise ValueError(
                "loss.use_sigreg=false needs a target that cannot collapse: set "
                "model.freeze_encoder=true and model.use_projector=false"
            )

    # (de)serialisation
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _build(cls, d)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
#  Presets: named bundles of overrides
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict[str, Any]] = {
    # LeWM from pixels: ViT-tiny trained from scratch with SIGReg.
    "vit": {},
    # DINO-WM: predict frozen DINOv2 ViT-S/14 patch features (14x14 grid, 196
    # tokens of 384). Frozen features cannot collapse: no SIGReg, no projector.
    "dino": {
        "model.encoder": "dino",
        "model.freeze_encoder": True,
        "model.embed_dim": 384,
        "model.use_projector": False,
        "loss.use_sigreg": False,
        "optim.batch_size": 32,  # DINO-WM's batch and predictor lr
        "optim.lr": 5e-4,
    },
    # Oracle: an MLP over the task's true state.
    "oracle": {
        "data.obs_key": "observation",
        "data.train_stride": 1,
        "model.encoder": "mlp",
        "optim.batch_size": 512,
        "optim.lr": 3e-4,
        "optim.num_workers": 4,
    },
    # 20-second smoke test.
    "debug": {
        "data.max_episodes": 40,
        "data.val_episodes": 8,
        "optim.epochs": 2,
        "optim.batch_size": 8,
        "optim.steps_per_epoch": 10,
        "optim.val_steps": 5,
        "optim.num_workers": 2,
        "optim.log_every": 1,
    },
}


# --------------------------------------------------------------------------- #
#  Plumbing
# --------------------------------------------------------------------------- #
def _build(cls, d: dict):
    """Recursively instantiate a dataclass tree from a dict, skipping unknown keys."""
    hints = get_type_hints(cls)  # resolves the string annotations
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        value, ftype = d[f.name], hints[f.name]
        if is_dataclass(ftype) and isinstance(value, dict):
            value = _build(ftype, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _coerce(value: str, current: Any) -> Any:
    """Cast a CLI string to the type of the value it replaces."""
    if value.lower() in ("none", "null"):
        return None
    if current is None:
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value
    if isinstance(current, bool):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot read {value!r} as a bool")
    if isinstance(current, (tuple, list)):
        parts = [p for p in value.strip("[]()").split(",") if p != ""]
        inner = type(current[0]) if len(current) else float
        return type(current)(inner(p) for p in parts)
    return type(current)(value)


def set_by_path(cfg: Any, path: str, value: Any) -> None:
    """`set_by_path(cfg, "optim.lr", "1e-4")`: dotted assignment with type casting."""
    *sections, leaf = path.split(".")
    node = cfg
    for name in sections:
        if not hasattr(node, name):
            raise KeyError(f"unknown config section {name!r} in {path!r}")
        node = getattr(node, name)
    if not hasattr(node, leaf):
        raise KeyError(f"unknown config field {path!r}")
    if isinstance(value, str):
        value = _coerce(value, getattr(node, leaf))
    setattr(node, leaf, value)


def get_config(argv: list[str] | None = None, **extra_flags: Any) -> tuple[Config, argparse.Namespace]:
    """Build a Config from `--preset`, an optional `--config file.json` and `key=value` overrides.

    `extra_flags` become `--flag` options of the script (e.g. `resume=False`).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="vit", choices=sorted(PRESETS))
    parser.add_argument("--config", default=None, help="json config to start from")
    for name, default in extra_flags.items():
        flag = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(flag, dest=name, action="store_true")
        else:
            parser.add_argument(flag, dest=name, default=default)
    args, overrides = parser.parse_known_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    for path, value in PRESETS[args.preset].items():
        set_by_path(cfg, path, value)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"cannot parse override {item!r}; expected key=value")
        path, value = item.split("=", 1)
        set_by_path(cfg, path.lstrip("-"), value)

    if cfg.data.h5_path is None:
        cfg.data.h5_path = str(mmbench_h5(cfg.data.task))
    if cfg.run_name == Config.run_name:
        cfg.run_name = f"{cfg.data.task}_{args.preset}"
    if cfg.exp_id is None:
        cfg.exp_id = f"exp_{int(time.time())}"
    cfg.validate()
    return cfg, args
