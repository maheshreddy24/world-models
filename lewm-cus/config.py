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

    # Which recording, and therefore which reader, the scripts use:
    #   "h5"     OGBench cube-single, one HDF5 table -> `data.h5_path`
    #   "scene"  OGBench scene-play, a memory-mapped npz store -> the `scene` section
    backend: str = "h5"

    h5_path: str = str(HOME / "datasets/ogbench/cube_single_expert.h5")

    # A "latent step" spans `frameskip` raw env steps.  The action of a latent
    # step is the flattened block of the `frameskip` raw actions it covers, so
    # the model sees action_dim * frameskip numbers per step.
    frameskip: int = 5
    history: int = 3  # context length fed to the predictor
    num_preds: int = 1  # how far past the context a sequence reaches

    img_size: int = 224
    obs_key: str = "pixels"  # "pixels" or "observation" (oracle state)

    # Rows between window starts (h5 backend). 13/18 suit the 201-step cube
    # episodes; short episodes (BallCatch, 61 rows) want 1 so no offset is skipped.
    train_stride: int = 13
    val_stride: int = 18

    # Cap episodes for quick experiments (None = all 10k).
    max_episodes: int | None = None
    # total episodes are 10k, 
    val_episodes: int = 1000  # held out by episode, never by row

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


@dataclass
class SceneConfig:
    """The OGBench scene-play recording, its store, and the v1 tasks on top of it.

    Paths are resolved against `data_dir` unless they are absolute, so pointing
    the whole thing somewhere else is one override:

        python train.py --preset scene_pixels scene.data_dir=/mnt/data/scene

    The recording ships a separate validation file, so stage 1 and stage 2 both
    validate on episodes the model has never seen rather than on a slice of the
    training ones.
    """

    data_dir: str = str(Path(__file__).resolve().parent / "ogbench_scene_single")
    train_npz: str = "visual-scene-play-v0.npz"
    val_npz: str = "visual-scene-play-v0-val.npz"
    # Where `datasets/prepare_scene.py` writes the memory-mapped copies (~25 GB for the
    # training recording). None puts each one beside its npz as `<name>_store`.
    store_dir: str | None = None

    # -- stage 2: mined start/goal pairs ---------------------------------- #
    # Pairs index the recording they were mined from, so `train_pairs` must come
    # from `train_npz`. `val_pairs` is optional: without it the validation goals
    # are hindsight goals, which need no task labels (see datasets/scene_v1.py).
    train_pairs: str | None = "train_pairs.npz"
    val_pairs: str | None = None
    # The held-out evaluation set. Self-contained: it carries the simulator state
    # to reset to and both frames, so it never indexes into a recording.
    v1_pairs: str = "v1_pairs.npz"

    # -- stage 2: how goals are drawn ------------------------------------- #
    p_task: float = 0.5  # rest of the draws are hindsight goals
    balance_tasks: bool = True  # toggle_lock has 80k pairs, cube_into_drawer 3k
    hindsight_min_gap: int = 30  # must be >= policy.chunk
    hindsight_max_gap: int = 160  # the longest gap the mined pairs contain
    # Redraw the start row inside [start, goal - chunk] so the policy sees the
    # whole way to a goal, not just its first chunk. See src/data/scene.py.
    resample_start: bool = True
    samples_per_epoch: int = 200_000
    val_samples: int = 20_000

    # -- v1 evaluation ----------------------------------------------------- #
    env_id: str = "visual-scene-v0"  # gym id behind the `visual-scene-play-v0` data
    num_eval: int | None = None  # tasks to run; None = all 350 pairs
    # Envs alive at once. Each holds its own EGL render context, ~124 MB of GPU
    # memory, so all 350 at once would need ~43 GB. The pool is built once and
    # reused for every batch; building an env costs about a second.
    eval_batch: int = 50
    budget: int = 250  # raw env steps per episode (mined gaps reach 160)
    cube_tol: float = 0.04  # success tolerances, see scene_tasks.is_success
    slide_tol: float = 0.03
    render_size: int = 64  # video resolution; the policy always sees 64x64

    # ---- derived --------------------------------------------------------- #
    def _path(self, name: str | None) -> Path | None:
        if not name:
            return None
        path = Path(name)
        return path if path.is_absolute() else Path(self.data_dir) / path

    def _store(self, npz: str) -> Path:
        if self.store_dir:
            return Path(self.store_dir) / (Path(npz).name.replace(".npz", "") + "_store")
        npz_path = self._path(npz)
        return npz_path.with_name(npz_path.name.replace(".npz", "") + "_store")

    @property
    def train_npz_path(self) -> Path:
        return self._path(self.train_npz)

    @property
    def val_npz_path(self) -> Path:
        return self._path(self.val_npz)

    @property
    def train_store_dir(self) -> Path:
        return self._store(self.train_npz)

    @property
    def val_store_dir(self) -> Path:
        return self._store(self.val_npz)

    @property
    def train_pairs_path(self) -> Path | None:
        return self._path(self.train_pairs)

    @property
    def val_pairs_path(self) -> Path | None:
        return self._path(self.val_pairs)

    @property
    def v1_pairs_path(self) -> Path:
        return self._path(self.v1_pairs)


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """LeWM = encoder + action encoder + autoregressive latent predictor."""

    # -- encoder ----------------------------------------------------------- #
    # "vit"  pixels, ViT trained from scratch (LeWM)
    # "dino" pixels, pretrained DINOv2 (DINO-WM)
    # "mlp"  oracle state
    encoder: str = "vit"
    # A frozen encoder gets no gradients and stays in eval mode.  It is also
    # what makes loss.use_sigreg=false safe: its features cannot collapse.
    freeze_encoder: bool = False

    vit_name: str = "vit_tiny_patch16_224"
    vit_pretrained: bool = False
    vit_pool: str = "cls"  # "cls" or "mean"

    # Frames are resized to dino_img_size inside the encoder, so data.img_size
    # (what is stored and rendered) stays untouched.
    dino_name: str = "dinov2_vits14"  # torch.hub name; vits14 is 384 wide
    dino_tokens: str = "patch"  # "patch": 14x14 token grid (DINO-WM) | "cls": one vector
    dino_img_size: int = 196  # multiple of the 14px patch; 196 -> 14x14 grid

    mlp_hidden: int = 512  # only used by the "mlp" encoder
    # Cube oracle-state groups the mlp encoder never sees, comma-separated names
    # from src.data.ogbench.OBS_GROUPS, e.g. "joint_vel,gripper_contact". The
    # data and normaliser stay 28-d; the encoder drops the columns on the way in.
    drop_obs: str = ""
    # Cube oracle-state groups read *next to* the pixels, same names as drop_obs,
    # e.g. "ee_pos,gripper_opening". Their own MLP turns them into one extra
    # token per frame (P -> P + 1), which the predictor predicts like any other
    # token, so planning needs the goal's state too. Empty = pixels only.
    oracle_obs: str = ""
    oracle_hidden: int = 512

    embed_dim: int = 192  # latent width = encoder width (vit-tiny 192, dinov2_vits14 384)

    # -- action encoder ---------------------------------------------------- #
    act_smooth_dim: int = 64  # width of the 1x1 conv that mixes action dims
    act_mlp_scale: int = 4

    # -- predictor --------------------------------------------------------- #
    # Widths match the LeWM reference (config/train/model/lewm_oracle.yaml):
    # 16 heads x 64 dim_head and a 2048-wide MLP, both independent of
    # embed_dim. Sizing them to embed_dim instead (6 heads, 768) gives a
    # visibly smaller predictor and is not what the paper trained.
    depth: int = 6
    heads: int = 16
    dim_head: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1

    # -- projectors -------------------------------------------------------- #
    # Applied after the encoder (projector) and after the predictor (pred_proj).
    # Set use_projector=False to regularise/predict raw encoder features.  Must
    # be False without SIGReg: a trainable projector can collapse even on top
    # of a frozen encoder.
    use_projector: bool = True
    proj_hidden: int = 2048
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
    batch_size: int = 128
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
    """Next-embedding prediction, plus SIGReg when the target could collapse.

    SIGReg keeps the embedding isotropic Gaussian so a *trainable* encoder
    cannot map every frame to one point.  A frozen encoder with no projector
    (DINO-WM) predicts fixed features, so the term can be switched off.
    """

    use_sigreg: bool = True
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_proj: int = 1024


# --------------------------------------------------------------------------- #
#  Planning / evaluation
# --------------------------------------------------------------------------- #
@dataclass
class PlanConfig:
    """How eval.py acts: MPC over the world model, or the diffusion policy."""

    # "mpc"       a solver searches action blocks through the world model
    # "diffusion" the policy from train_policy.py (pass --diffusion-ckpt)
    planner: str = "mpc"

    # -- mpc --------------------------------------------------------------- #
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

    # -- diffusion --------------------------------------------------------- #
    # Chunks sampled per replan.  1 = the policy alone; >1 = the world model
    # rolls every chunk out and the one landing closest to the goal is executed.
    diffusion_samples: int = 1
    ddim_steps: int = 10  # denoising steps per sample

    @property
    def plan_steps(self) -> int:
        """Raw env steps covered by one plan."""
        return self.horizon * self.action_block


@dataclass
class PolicyConfig:
    """Goal-conditioned diffusion policy on a frozen LeWM (WorldDP low level).

    Trained by train_policy.py on top of a world-model checkpoint and used by
    eval.py with plan.planner=diffusion.  Output: the next `chunk` raw actions,
    denoised under the condition

        c = [ z_t      frozen world-model latent of the current frame  (P x D)
              proprio  see `proprio`                                   (13 | 6)
              goal     see `goal`                                      (3 | P x D)
              contact  gripper contact, with `use_contact`             (1) ]

    Each part enters the denoiser as its own token(s).  The world model is
    untouched by all of this; its side of the contact add-on is
    model.drop_obs=gripper_contact.
    """

    chunk: int = 5  # raw env actions per sample, executed before replanning
    # Training goals are drawn this many raw steps ahead of the current frame.
    # Eval goals sit eval.goal_offset (25) away, so the range must cover it.
    goal_offset_min: int = 5
    goal_offset_max: int = 50
    # "cube_xyz"  the cube's target position in metres (3), the simulator's own
    #             success target. cube-single only.
    # "latent"    the world model's latent of the goal frame (scene-play needs this)
    goal: str = "cube_xyz"
    # "arm"  joint pos (6) + joint vel (6) + gripper opening (1). cube-single only.
    # "ee"   end-effector position + per-step velocity (6)
    proprio: str = "arm"
    use_proprio: bool = True
    use_contact: bool = False  # gripper contact (1) as its own token. cube-single only.
    # Baseline: build the --ckpt world model's architecture but keep its random
    # initial weights, so z_t comes from an untrained backbone. train_policy.py
    # saves it into the policy run and eval.py loads it from there.
    random_backbone: bool = False
    # Std of Gaussian noise on the goal while training: latent units, or metres for cube_xyz.
    goal_noise: float = 0.0

    # -- denoiser (transformer Diffusion Policy; widths from WorldDP) ------- #
    width: int = 256
    depth: int = 8
    heads: int = 4
    cond_layers: int = 2
    dropout: float = 0.1

    train_timesteps: int = 100  # DDPM noise levels (cosine schedule)
    ema_decay: float = 0.999  # checkpoints hold the EMA weights


@dataclass
class EvalConfig:
    num_eval: int = 50  # eval episodes (tasks)
    # Episodes run in parallel; num_eval runs in batches of this many. Each env
    # holds ~120 MB of GPU memory for rendering, so this caps that cost.
    num_envs: int = 50
    goal_offset: int = 25  # goal is this many raw steps ahead of the start
    eval_budget: int = 50  # raw env steps allowed per episode
    # Metres the cube must travel between start and goal for a task to count.
    # The simulator calls success at 0.04m, so anything near that is solved (or
    # nearly solved) before the planner acts: a random policy scores 31% at
    # 0.04 but only ~5% at 0.10. See src/data/ogbench.sample_eval_episodes.
    min_goal_distance: float = 0.10
    cube_tol: float = 0.04  # cube-double success radius per cube (eval_cube_double.py)
    env_id: str = "swm/OGBCube-v0"
    env_type: str = "single"
    terminate_at_goal: bool = True
    # Grasp diagnostics, reported next to the success rate. `gripper_contact` is
    # near-binary in the recording (41% of frames above 0.9, median 0), the cube
    # rests at z = 0.02 m, and a contact frame has the cube ~0.5 cm from the
    # effector, so these thresholds are far from any boundary.
    grasp_contact: float = 0.5  # contact reading that counts as "touching"
    grasp_dist_cm: float = 3.0  # cube must be this close, so the contact is with it
    lift_cm: float = 2.0  # cube this far above its start height counts as lifted

    save_video: bool = True
    num_videos: int = 8  # episodes written as mp4, capped at num_eval
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
    scene: SceneConfig = field(default_factory=SceneConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # ---- derived -------------------------------------------------------- #
    @property
    def run_dir(self) -> Path:
        """`<out_dir>/exp_<timestamp>` — one directory per experiment."""
        return Path(self.out_dir) / (self.exp_id or self.run_name)

    @property
    def eval_budget(self) -> int:
        """Raw env steps allowed per evaluation episode, for the active backend."""
        return self.scene.budget if self.data.backend == "scene" else self.eval.eval_budget

    def validate(self) -> None:
        if self.data.backend not in ("h5", "scene"):
            raise ValueError(f"data.backend must be 'h5' or 'scene', got {self.data.backend!r}")
        if self.data.backend == "scene":
            if self.data.obs_key not in ("pixels", "observation"):
                raise ValueError(
                    "the scene backend offers data.obs_key='pixels' or 'observation' "
                    f"(the oracle state, see datasets/prepare_scene.py --state); got {self.data.obs_key!r}"
                )
            if not self.scene.hindsight_min_gap <= self.scene.hindsight_max_gap:
                raise ValueError("need scene.hindsight_min_gap <= scene.hindsight_max_gap")
            if self.scene.hindsight_min_gap < self.policy.chunk:
                raise ValueError(
                    f"scene.hindsight_min_gap ({self.scene.hindsight_min_gap}) must be >= "
                    f"policy.chunk ({self.policy.chunk}): a goal closer than one chunk "
                    "leaves no actions to supervise"
                )
        if self.plan.action_block != self.data.frameskip:
            raise ValueError(
                f"plan.action_block ({self.plan.action_block}) must match "
                f"data.frameskip ({self.data.frameskip}): the model only ever "
                "sees actions grouped in blocks of that size."
            )
        if self.plan.planner not in ("mpc", "diffusion"):
            raise ValueError(f"plan.planner must be 'mpc' or 'diffusion', got {self.plan.planner!r}")
        if not 1 <= self.policy.goal_offset_min <= self.policy.goal_offset_max:
            raise ValueError("need 1 <= policy.goal_offset_min <= policy.goal_offset_max")
        if self.policy.goal not in ("cube_xyz", "latent"):
            raise ValueError(f"policy.goal must be 'cube_xyz' or 'latent', got {self.policy.goal!r}")
        if self.policy.proprio not in ("arm", "ee"):
            raise ValueError(f"policy.proprio must be 'arm' or 'ee', got {self.policy.proprio!r}")
        if self.plan.diffusion_samples > 1 and self.policy.chunk % self.data.frameskip:
            raise ValueError(
                f"plan.diffusion_samples > 1 ranks chunks with the world model, which reads "
                f"actions in blocks of data.frameskip ({self.data.frameskip}); "
                f"policy.chunk ({self.policy.chunk}) must be a multiple of it"
            )
        if self.plan.receding_horizon > self.plan.horizon:
            raise ValueError("plan.receding_horizon must be <= plan.horizon")
        if self.plan.plan_steps > self.eval_budget:
            raise ValueError(
                f"one plan covers {self.plan.plan_steps} env steps but the eval "
                f"budget is only {self.eval_budget}"
            )
        if self.model.encoder == "mlp" and self.data.obs_key == "pixels":
            raise ValueError("the mlp encoder needs data.obs_key='observation'")
        if self.model.encoder == "dino" and self.data.obs_key != "pixels":
            raise ValueError("the dino encoder needs data.obs_key='pixels'")
        if self.model.drop_obs and (self.model.encoder != "mlp" or self.data.backend != "h5"):
            raise ValueError(
                "model.drop_obs hides groups of the cube oracle state; it needs "
                "model.encoder='mlp' on data.backend='h5'"
            )
        if self.model.oracle_obs:
            if self.model.encoder == "mlp" or self.data.backend != "h5":
                raise ValueError(
                    "model.oracle_obs adds groups of the cube oracle state next to the pixels; "
                    "it needs a pixel encoder (vit or dino) on data.backend='h5'"
                )
            if not self.loss.use_sigreg:
                raise ValueError(
                    "model.oracle_obs trains an MLP for the extra token, and without SIGReg "
                    "it can collapse to a constant; keep loss.use_sigreg=true"
                )
        if not self.loss.use_sigreg and (not self.model.freeze_encoder or self.model.use_projector):
            raise ValueError(
                "loss.use_sigreg=false needs a target that cannot collapse: set "
                "model.freeze_encoder=true and model.use_projector=false. A trainable "
                "encoder or projector can map every frame to one point and drive "
                "the prediction loss to zero."
            )

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
    # DINO-WM style: predict frozen DINOv2 ViT-S/14 patch features directly.
    # Frozen features cannot collapse, so no SIGReg and no projector.
    # `model.dino_tokens=cls` swaps the 196-token grid for one vector per frame.
    "cube_dino": {
        "run_name": "lewm_cube_dino",
        "model.encoder": "dino",
        "model.freeze_encoder": True,
        "model.embed_dim": 384,
        "model.use_projector": False,
        "loss.use_sigreg": False,
        "optim.batch_size": 32,  # DINO-WM's batch and predictor lr
        "optim.lr": 5e-4,
        # 50 envs x 300 samples x 588 tokens does not fit at once on a 24GB L4;
        # 10 candidates per pass peaks at ~13GB. Pass --preset cube_dino to eval.py.
        "plan.chunk_size": 10,
    },
    # ---- BallCatch (src/envs/ball.py, datasets/make_ball_data.py) -------- #
    # 2D pymunk catching task: 128x128 frames of the whole scene, 1-d action
    # (basket velocity), 61-row episodes at 20 Hz. frameskip 5 = 0.25 s per
    # latent step; a window (history 3 + 1) spans 20 steps, one second.
    # Stride 1 uses all 41 window offsets per episode.
    "ball_pixels": {
        "run_name": "lewm_ball",
        "data.h5_path": str(HOME / "datasets/ballcatch/ballcatch_v2_20k.h5"),
        "data.img_size": 128,
        "data.obs_key": "pixels",
        "data.train_stride": 1,
        "data.val_stride": 3,
        # 2000 held-out episodes (10%) -> ~400 eligible eval tasks, 200 used.
        "data.val_episodes": 2000,
        "model.encoder": "vit",
        "model.embed_dim": 192,
        "optim.batch_size": 128,
        "optim.num_workers": 12,
        # eval_ball.py: start at the launch frame of a held-out expert catch, the
        # goal is the frame `eval.goal_offset` steps after the catch (ball in the
        # basket), and the basket must travel at least `min_goal_distance` metres.
        "eval.num_eval": 200,
        "eval.goal_offset": 3,
        "eval.eval_budget": 50,
        "eval.min_goal_distance": 0.5,
        "eval.num_envs": 100,
        "eval.video_fps": 20,
    },
    # Oracle: an MLP over the 6-d state (ball x, y, vx, vy, basket x, vx).
    "ball_oracle": {
        "run_name": "lewm_ball_oracle",
        "data.h5_path": str(HOME / "datasets/ballcatch/ballcatch_v2_20k.h5"),
        "data.img_size": 128,
        "data.obs_key": "observation",
        "data.train_stride": 1,
        "data.val_stride": 3,
        # 2000 held-out episodes (10%) -> ~400 eligible eval tasks, 200 used.
        "data.val_episodes": 2000,
        "model.encoder": "mlp",
        "optim.batch_size": 512,
        "optim.lr": 3e-4,
        "optim.num_workers": 4,
        "eval.num_eval": 200,
        "eval.goal_offset": 3,
        "eval.eval_budget": 50,
        "eval.min_goal_distance": 0.5,
        "eval.num_envs": 100,
        "eval.video_fps": 20,
    },
    # ---- MMBench2 acrobot-swingup (datasets/prepare_mmbench.py) ---------- #
    # DMControl acrobot from nicklashansen/mmbench2: 224x224 frames, 1-d action
    # (elbow torque; the 15 padding columns are cut), 501-row episodes, each row
    # already 2 sim steps (TD-MPC2 action repeat). frameskip 5 = 10 sim steps per
    # latent step; a window (history 3 + 1) spans 20 rows. expert + mixed-small +
    # mixed-large + zeros = 260 episodes; 26 (10%) held out. No closed-loop eval
    # yet (lewm-cus has no acrobot env), so train.py's val loss and rollout.py
    # are the measures.
    "acrobot_pixels": {
        "run_name": "lewm_acrobot",
        "data.h5_path": str(HOME / "datasets/mmbench/acrobot-swingup.h5"),
        "data.img_size": 224,
        "data.obs_key": "pixels",
        "data.train_stride": 2,
        "data.val_stride": 5,
        "data.val_episodes": 26,
        "model.encoder": "vit",
        "model.embed_dim": 192,
        "optim.batch_size": 128,
        "optim.num_workers": 12,
    },
    # Oracle: an MLP over the 6-d state (cos/sin of both joint angles, 2 joint velocities).
    "acrobot_oracle": {
        "run_name": "lewm_acrobot_oracle",
        "data.h5_path": str(HOME / "datasets/mmbench/acrobot-swingup.h5"),
        "data.obs_key": "observation",
        "data.train_stride": 1,
        "data.val_stride": 5,
        "data.val_episodes": 26,
        "model.encoder": "mlp",
        "optim.batch_size": 512,
        "optim.lr": 3e-4,
        "optim.num_workers": 4,
    },
    # ---- OGBench scene-play (data.backend=scene) ------------------------- #
    # Stage 1: the dynamics predictor on visual-scene-play-v0. Frames are 64x64,
    # so the ViT runs at 64 and sees a 4x4 patch grid; nothing is upsampled.
    "scene_pixels": {
        "run_name": "lewm_scene",
        "data.backend": "scene",
        "data.img_size": 64,
        "data.obs_key": "pixels",
        "model.encoder": "vit",
        "model.embed_dim": 192,
        "optim.batch_size": 128,
        "optim.num_workers": 12,
    },
    # Stage 1 on visual-cube-double-play, through the same npz store as scene
    # (`python datasets/prepare_scene.py --preset cube_double_pixels`). The recording is
    # 124x124; 128 is the nearest multiple of the 16px patch, an 8x8 grid.
    "cube_double_pixels": {
        "run_name": "lewm_cube_double",
        "data.backend": "scene",
        "data.img_size": 128,
        "data.obs_key": "pixels",
        "model.encoder": "vit",
        "model.embed_dim": 192,
        "optim.batch_size": 128,
        "optim.num_workers": 12,
        "scene.data_dir": "/home/world-models/ogbench/data_gen_scripts/data",
        "scene.train_npz": "visual-cube-double-play-224.npz",
        "scene.val_npz": "visual-cube-double-play-224-val.npz",
        "scene.env_id": "visual-cube-double-v0",
        # eval_cube_double.py: goals 100 steps ahead in the val recording, 2x that
        # to reach them. A pick-and-place takes about that long: with both cubes
        # at rest at start and goal, 40% of 100-step windows move a cube >= 10 cm
        # (13% at 50 steps). Nearly all of them move one cube; ~15% end stacked.
        "eval.num_eval": 200,
        "eval.goal_offset": 100,
        "eval.eval_budget": 200,
        "eval.min_goal_distance": 0.10,
    },
    # Stage 1, oracle: an MLP over the 39-d scene state instead of the ViT.
    # Needs `python datasets/prepare_scene.py --state`. Stage 2 is unchanged — train_policy.py
    # adopts the observation from whichever world model --ckpt points at.
    "scene_oracle": {
        "run_name": "lewm_scene_oracle",
        "data.backend": "scene",
        "data.img_size": 64,
        "data.obs_key": "observation",
        "model.encoder": "mlp",
        "optim.batch_size": 512,
        "optim.lr": 3e-4,
        "optim.num_workers": 8,
    },
    # Stage 1, DINO-WM variant: frozen DINOv2 patches. 64 -> 224 inside the
    # encoder (a multiple of the 14px patch), so a frame becomes a 16x16 grid.
    "scene_dino": {
        "run_name": "lewm_scene_dino",
        "data.backend": "scene",
        "data.img_size": 64,
        "model.encoder": "dino",
        "model.freeze_encoder": True,
        "model.dino_img_size": 224,
        "model.embed_dim": 384,
        "model.use_projector": False,
        "loss.use_sigreg": False,
        "optim.batch_size": 32,
        "optim.lr": 5e-4,
        "optim.num_workers": 12,
        "plan.chunk_size": 10,
    },
    # Stage 2: the goal-conditioned diffusion planner on the mined pairs.
    # data/model come from the --ckpt world model; chunk 15 = 3 latent steps at
    # frameskip 5, so plan.diffusion_samples>1 can rank chunks with that model.
    "scene_policy": {
        "run_name": "scene_diffusion_policy",
        "data.backend": "scene",
        "data.img_size": 64,  # overwritten by the --ckpt world model; set for clarity
        "plan.planner": "diffusion",
        "policy.chunk": 15,
        # Seven tasks over several objects: no single cube target to condition on.
        "policy.goal": "latent",
        "policy.proprio": "ee",
        "optim.lr": 1e-4,
        "optim.weight_decay": 1e-6,
        "optim.epochs": 20,
        "optim.warmup_epochs": 0.5,
        "optim.num_workers": 12,
        "optim.steps_per_epoch": 2000,
        "optim.val_steps": 50,
    },
    # Scene smoke test: a few episodes, a few steps, a handful of eval tasks.
    "scene_debug": {
        "run_name": "scene_debug",
        "data.backend": "scene",
        "data.img_size": 64,
        "data.max_episodes": 8,
        "optim.epochs": 2,
        "optim.batch_size": 8,
        "optim.steps_per_epoch": 10,
        "optim.val_steps": 5,
        "optim.num_workers": 2,
        "optim.log_every": 1,
        "policy.chunk": 15,
        "policy.goal": "latent",
        "policy.proprio": "ee",
        "scene.samples_per_epoch": 200,
        "scene.val_samples": 80,
        "scene.num_eval": 4,
        "scene.budget": 60,
        "plan.num_samples": 16,
        "plan.n_iters": 2,
        "plan.topk": 4,
        "rollout.num_sequences": 16,
        "rollout.retrieval_bank": 128,
    },
    # Diffusion policy (train_policy.py). data/model come from the --ckpt world
    # model, so only the optimisation differs from the defaults. WorldDP used
    # lr 1e-4 and ~3 passes over the data; 20 x 2000 steps x 128 is ~2.8 passes.
    "policy": {
        "run_name": "diffusion_policy",
        "optim.lr": 1e-4,
        "optim.weight_decay": 1e-6,
        "optim.epochs": 20,
        "optim.warmup_epochs": 0.5,
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


def get_config(
    argv: list[str] | None = None, default_preset: str = "cube_pixels", **defaults: Any
) -> tuple[Config, argparse.Namespace]:
    """Build a Config from `--preset`, a `--config file.json` and `key=value` args.

    Returns the config plus the parsed namespace, so scripts can add their own
    flags (`--ckpt`, `--policy`, ...) through the `known` extras.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--preset", default=default_preset, choices=sorted(PRESETS))
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
