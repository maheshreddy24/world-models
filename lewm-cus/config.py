"""Single source of truth for every knob in the project.

Everything is a plain dataclass, so editing a default is a one-line change and
your editor can autocomplete the fields.  Three ways to change a setting:

    1. edit the default here,
    2. pick a preset:              python train.py --preset cube_oracle
    3. override on the CLI:        python train.py optim.lr=1e-4 model.depth=8

CLI overrides are dotted paths into the config tree and are cast to the type of
the field they replace, so `optim.epochs=50` gives an int and `wandb.enabled=true`
a bool.
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
    """Locate the data/checkpoint store.

    `STABLEWM_HOME` wins if set; otherwise prefer a `.stable_worldmodel` next to
    this repo (where the OGBench dumps usually land) and fall back to the one in
    the home directory.
    """
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


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the trajectories come from and how they are cut into sequences."""

    h5_path: str = str(HOME / "datasets/ogbench/cube_single_expert.h5")

    # A "latent step" spans `frameskip` raw env steps.  The action of a latent
    # step is the flattened block of the `frameskip` raw actions it covers, so
    # the model sees action_dim * frameskip numbers per step.
    frameskip: int = 5
    history: int = 3  # context length fed to the predictor
    num_preds: int = 1  # how far past the context a sequence reaches

    img_size: int = 224
    obs_key: str = "pixels"  # "pixels" or "observation" (oracle state)

    # Cap episodes for quick experiments (None = all 10k).
    max_episodes: int | None = None
    val_episodes: int = 200  # held out by episode, never by row

    # Normalisation statistics are computed once and cached next to the h5 file.
    stats_path: str | None = None
    stats_max_rows: int = 200_000  # rows subsampled to estimate mean/std

    # HDF5 chunk cache per worker.  Pixels are stored in 100-frame chunks; a
    # bigger cache means fewer repeated decompressions.
    rdcc_mb: int = 256

    @property
    def seq_len(self) -> int:
        """Latent steps per training sequence."""
        return self.history + self.num_preds


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """LeWM = encoder + action encoder + autoregressive latent predictor."""

    # -- encoder ----------------------------------------------------------- #
    encoder: str = "vit"  # "vit" (pixels) or "mlp" (oracle state)
    vit_name: str = "vit_tiny_patch16_224"
    vit_pretrained: bool = False
    vit_pool: str = "cls"  # "cls" or "mean"
    mlp_hidden: int = 512  # only used by the "mlp" encoder

    embed_dim: int = 192  # latent width; vit-tiny is 192

    # -- action encoder ---------------------------------------------------- #
    act_smooth_dim: int = 64  # width of the 1x1 conv that mixes action dims
    act_mlp_scale: int = 4

    # -- predictor --------------------------------------------------------- #
    depth: int = 6
    heads: int = 6
    dim_head: int = 64
    mlp_dim: int = 768
    dropout: float = 0.1

    # -- projectors -------------------------------------------------------- #
    # Applied after the encoder (projector) and after the predictor (pred_proj).
    # Set use_projector=False to regularise/predict raw encoder features.
    use_projector: bool = True
    proj_hidden: int = 1024
    # "batch" matches the LeWM reference, but BatchNorm makes the embedding
    # batch-dependent in training and running-stat-dependent in eval, so
    # val/emb_std lags train/emb_std until the running stats settle.
    # Switch to "layer" for identical train/eval behaviour.
    proj_norm: str = "batch"

    # -- planning cost ----------------------------------------------------- #
    cost: str = "final_mse"  # "final_mse" | "mean_mse" | "final_cosine"


# --------------------------------------------------------------------------- #
#  Optimisation
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 5e-5
    weight_decay: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: float = 2.0
    min_lr_scale: float = 0.01  # final lr = lr * min_lr_scale
    grad_clip: float = 1.0
    amp_dtype: str = "bf16"  # "bf16" | "fp16" | "off"

    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True

    # Cap steps per epoch to get frequent checkpoints on a huge dataset.
    steps_per_epoch: int | None = 2000
    val_steps: int | None = 100
    log_every: int = 20
    ckpt_every: int = 1  # epochs
    # Which val metric picks best.pt. Keep "pred_vs_static": raw "pred_loss"
    # is minimised by a collapsed encoder, which predicts a constant perfectly.
    select_metric: str = "pred_vs_static"


@dataclass
class LossConfig:
    """Two terms only: next-embedding prediction + isotropic-Gaussian regulariser."""

    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_proj: int = 1024


# --------------------------------------------------------------------------- #
#  Planning / evaluation
# --------------------------------------------------------------------------- #
@dataclass
class PlanConfig:
    """Model-predictive control settings."""

    solver: str = "cem"  # "cem" | "mppi" | "random"
    horizon: int = 5  # latent steps planned ahead
    receding_horizon: int = 5  # latent steps executed before replanning
    action_block: int = 5  # raw env steps per latent step (= data.frameskip)
    history_len: int = 1  # observed frames given to the model when planning
    warm_start: bool = True

    # solver internals
    num_samples: int = 300
    n_iters: int = 30
    topk: int = 30
    var_scale: float = 1.0
    momentum: float = 0.1  # elite-mean smoothing, 0 = no smoothing
    min_std: float = 0.01
    temperature: float = 1.0  # MPPI only
    chunk_size: int = 0  # split candidates across forward passes (0 = all at once)

    @property
    def plan_steps(self) -> int:
        """Raw env steps covered by one plan."""
        return self.horizon * self.action_block


@dataclass
class EvalConfig:
    num_eval: int = 50  # parallel eval episodes
    goal_offset: int = 25  # goal is this many raw steps ahead of the start
    eval_budget: int = 50  # raw env steps allowed per episode
    # Metres the cube must travel between start and goal for a task to count.
    # The simulator calls success at 0.04m, so anything near that is solved (or
    # nearly solved) before the planner acts: a random policy scores 31% at
    # 0.04 but only ~5% at 0.10. See src/data/ogbench.sample_eval_episodes.
    min_goal_distance: float = 0.10
    env_id: str = "swm/OGBCube-v0"
    env_type: str = "single"
    terminate_at_goal: bool = True
    save_video: bool = True
    video_fps: int = 10


@dataclass
class RolloutConfig:
    """Open-loop latent rollout diagnostics (no planning, no env)."""

    num_sequences: int = 256
    horizon: int = 10  # latent steps to roll out
    batch_size: int = 32
    retrieval_video: bool = True  # nearest-neighbour "imagination" video
    retrieval_bank: int = 2048  # frames encoded to retrieve from
    num_videos: int = 4


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "lewm-cus"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"


# --------------------------------------------------------------------------- #
#  Root
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    run_name: str = "lewm_cube"  # human label for the run; names the wandb run
    # Checkpoints live next to the code, not in the data store, so an
    # experiment's weights, config and logs sit together in the repo.
    out_dir: str = str(Path(__file__).resolve().parent / "checkpoints")
    # Stamped once per process by `get_config` as exp_<unix time>, so every
    # launch gets a fresh directory and nothing is ever silently overwritten.
    # It travels in config.json and in every checkpoint, so resuming or
    # reloading a run reuses the directory it was created with.
    exp_id: str | None = None
    seed: int = 3072
    device: str = "cuda"
    compile: bool = False  # torch.compile the model

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ---- derived -------------------------------------------------------- #
    @property
    def run_dir(self) -> Path:
        """`<out_dir>/exp_<timestamp>` — one directory per experiment."""
        return Path(self.out_dir) / (self.exp_id or self.run_name)

    def validate(self) -> None:
        if self.plan.action_block != self.data.frameskip:
            raise ValueError(
                f"plan.action_block ({self.plan.action_block}) must match "
                f"data.frameskip ({self.data.frameskip}): the model only ever "
                "sees actions grouped in blocks of that size."
            )
        if self.plan.receding_horizon > self.plan.horizon:
            raise ValueError("plan.receding_horizon must be <= plan.horizon")
        if self.plan.plan_steps > self.eval.eval_budget:
            raise ValueError(
                f"one plan covers {self.plan.plan_steps} env steps but the eval "
                f"budget is only {self.eval.eval_budget}"
            )
        if self.model.encoder == "mlp" and self.data.obs_key == "pixels":
            raise ValueError("the mlp encoder needs data.obs_key='observation'")

    # ---- (de)serialisation ---------------------------------------------- #
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
#  Presets — named bundles of overrides
# --------------------------------------------------------------------------- #
PRESETS: dict[str, dict[str, Any]] = {
    # LeWM from pixels on OGBench cube-single (the default paper setting).
    "cube_pixels": {},
    # Oracle variant: an MLP over the 28-d state instead of the ViT.  Trains in
    # minutes and is the right thing to debug the planner against.
    "cube_oracle": {
        "run_name": "lewm_cube_oracle",
        "data.obs_key": "observation",
        "model.encoder": "mlp",
        "optim.batch_size": 512,
        "optim.lr": 3e-4,
        "optim.num_workers": 4,
    },
    # Tiny smoke test: a handful of episodes, a few steps, no video.
    "debug": {
        "run_name": "debug",
        "data.max_episodes": 40,
        "data.val_episodes": 8,
        "optim.epochs": 2,
        "optim.batch_size": 8,
        "optim.steps_per_epoch": 10,
        "optim.val_steps": 5,
        "optim.num_workers": 2,
        "optim.log_every": 1,
        "eval.num_eval": 2,
        "plan.num_samples": 16,
        "plan.n_iters": 2,
        "plan.topk": 4,
        "rollout.num_sequences": 16,
        "rollout.retrieval_bank": 128,
    },
}


# --------------------------------------------------------------------------- #
#  Plumbing
# --------------------------------------------------------------------------- #
def _build(cls, d: dict):
    """Recursively instantiate a dataclass tree from a plain dict."""
    # `from __future__ import annotations` makes f.type a string, so resolve the
    # real classes once through the module namespace.
    hints = get_type_hints(cls)
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
    if current is None:
        if value.lower() in ("none", "null"):
            return None
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
    if value.lower() in ("none", "null"):
        return None
    return type(current)(value)


def set_by_path(cfg: Any, path: str, value: Any) -> None:
    """`set_by_path(cfg, "optim.lr", 1e-4)` — dotted assignment with type casting."""
    node = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if not hasattr(node, p):
            raise KeyError(f"unknown config section {p!r} in {path!r}")
        node = getattr(node, p)
    leaf = parts[-1]
    if not hasattr(node, leaf):
        raise KeyError(f"unknown config field {path!r}")
    if isinstance(value, str):
        value = _coerce(value, getattr(node, leaf))
    setattr(node, leaf, value)


def get_config(argv: list[str] | None = None, **defaults: Any) -> tuple[Config, argparse.Namespace]:
    """Build a Config from `--preset`, a `--config file.json` and `key=value` args.

    Returns the config plus the parsed namespace, so scripts can add their own
    flags (`--ckpt`, `--policy`, ...) through the `known` extras.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--preset", default="cube_pixels", choices=sorted(PRESETS))
    parser.add_argument("--config", default=None, help="json config to start from")
    for name, default in defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=default)
    args, overrides = parser.parse_known_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    for path, value in PRESETS[args.preset].items():
        set_by_path(cfg, path, value)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"cannot parse override {item!r}; expected key=value")
        path, value = item.split("=", 1)
        set_by_path(cfg, path.lstrip("-"), value)

    # A run that did not inherit an id from a config/checkpoint gets a fresh one.
    if cfg.exp_id is None:
        cfg.exp_id = f"exp_{int(time.time())}"

    cfg.validate()
    return cfg, args
