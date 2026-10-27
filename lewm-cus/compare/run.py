"""Regulariser comparison on TwoRoom and PushT: 5 trained world models + frozen DINOv2 CLS.

    python compare/run.py                                  # everything: Sub-JEPA's training, both tasks, all models
    python compare/run.py --tasks tworoom --stages train   # one task, training only
    python compare/run.py --models sigreg_0.01 visreg      # some models
    python compare/run.py --stages analyse cem report      # evaluate what is trained
    python compare/run.py --stages analyse cem report --epoch 4   # an earlier epoch, reported separately
    python compare/run.py --recipe short --stages report   # the earlier 10 x 2000-step experiment
    python compare/run.py --smoke                          # tiny end-to-end check (minutes)

Models (the same LeWM world model and data; only what prevents collapse differs):

    sigreg_0.01   SIGReg, lambda = 0.01
    sigreg_0.2    SIGReg, lambda = 0.2
    subjepa       Sub-JEPA: SIGReg in K frozen orthogonal subspaces, lambda = 0.09, K = 32 (TwoRoom) / 16 (PushT)
    visreg        VISReg (scale + sliced-Wasserstein shape + centre), (1 - lambda) pred + lambda reg, lambda = 0.6
    ema           no regulariser, EMA target (momentum 0.996 -> 1)
    dino_cls      frozen DINOv2-small CLS token; predictor only

Recipes (--recipe), each in its own checkpoints/<root>/<task>/<model>/:

    subjepa  (default, checkpoints/compare_subjepa) Sub-JEPA's training exactly: 10 full passes over 90% of
             all 4-frame clips (TwoRoom 5,138, PushT 13,933 steps per epoch) in shuffled batches of 128;
             HF ViT-tiny with 14 px patches, action encoder to 10 channels, randn position embedding;
             AdamW lr 5e-5, weight decay 1e-3 on every parameter, clip 1, bf16; warm-up from 0 over 1%
             of the steps, cosine to 0. dino_cls trains its predictor the same way. Verified: with
             LeWM's released weights loaded, this model's outputs equal LeWM's own implementation.
    short    (checkpoints/compare) the first experiment: 10 x 2000 steps, timm ViT-tiny/16, DINO-WM's
             optimiser for dino_cls.

Stages, in order (an output that exists is skipped, unless --force):

    setup    pip install -r requirements.txt if stable-worldmodel is missing
    data     download LeWM's recordings (quentinll/lewm-tworooms, quentinll/lewm-pusht) into
             $STABLEWM_HOME/datasets/{tworoom,pusht}/ and unpack them
    train    train.py per (task, model); resumes a partial run; keeps every epoch's checkpoint
    analyse  compare/analysis.py: probe error vs horizon, folding, off-manifold, local stretch
    cem      compare/plan.py: CEM success rate under LeWM's protocol, Sub-JEPA's six seeds
    report   compare/report.py: tables, correlations and figures in <root>/report[_epochNNN]/

Evaluations of the last epoch go to <run>/{analysis,cem}/, of any other epoch to <run>/epochNNN/.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import HOME, task_h5  # noqa: E402

TASKS = ("tworoom", "pusht")
STAGES = ("setup", "data", "train", "analyse", "cem", "report")
EPOCHS = 10
SEEDS = (42, 100, 2026, 3407, 1234, 4444)  # Sub-JEPA's CEM evaluation seeds

# --------------------------------------------------------------------------- #
#  Models: a train.py preset and what is specific to the model
# --------------------------------------------------------------------------- #
MODELS: dict[str, tuple[str, list[str]]] = {
    "sigreg_0.01": ("vit", ["loss.sigreg_weight=0.01"]),
    "sigreg_0.2": ("vit", ["loss.sigreg_weight=0.2"]),
    "subjepa": ("vit", ["loss.reg_type=subjepa", "loss.sigreg_weight=0.09", "loss.subspace_init=orthogonal_frozen"]),
    "visreg": ("vit", ["loss.reg_type=visreg", "loss.sigreg_weight=0.6", "loss.pred_weight=0.4",
                       "loss.visreg_proj=2048"]),
    "ema": ("vit_ema", []),
    "dino_cls": ("dino_cls", []),
}
SUBSPACES = {"tworoom": 32, "pusht": 16}  # Sub-JEPA's K per environment


def model_overrides(task: str, model: str) -> list[str]:
    extra = list(MODELS[model][1])
    if model == "subjepa":
        extra.append(f"loss.subspaces={SUBSPACES[task]}")
    return extra


# --------------------------------------------------------------------------- #
#  Recipes: how every model is trained
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recipe:
    root: str  # checkpoints/<root>/
    overrides: Callable[[str, str], list[str]]  # (task, model) -> train.py overrides


def subjepa_steps_per_epoch(task: str, history: int = 3, preds: int = 1, frameskip: int = 5, batch: int = 128,
                            train_split: float = 0.9) -> int:
    """Gradient steps in one Sub-JEPA epoch: clips of (history + preds) * frameskip rows at every start row
    (stable-worldmodel's clip_indices), 90% of them for training, full batches only (drop_last)."""
    import h5py

    span = (history + preds) * frameskip
    with h5py.File(task_h5(task), "r") as f:
        lengths = f["ep_len"][:]
    clips = int(sum(max(0, int(n) - span + 1) for n in lengths))
    return int(train_split * clips) // batch


def subjepa_recipe(task: str, model: str) -> list[str]:
    """Sub-JEPA's training (its train.py on LeWM's code and stable-pretraining), applied to every model.

    The held-out split stays by episode, unlike Sub-JEPA's split of clips (whose "val" overlaps the
    training episodes); only the step count is taken from its split. Every optimiser setting is
    explicit, so no preset (dino_cls's lr 5e-4) leaks through.
    """
    return [
        # optimiser and schedule
        "optim.lr=5e-5", "optim.weight_decay=1e-3", "optim.decay_all=true", "optim.betas=(0.9,0.999)",
        "optim.batch_size=128", "optim.grad_clip=1.0", "optim.amp_dtype=bf16",
        f"optim.epochs={EPOCHS}", f"optim.steps_per_epoch={subjepa_steps_per_epoch(task)}",
        "optim.sample_without_replacement=true",
        "optim.warmup_epochs=0.1", "optim.min_lr_scale=0.0", "optim.lr_start_zero=true",
        "optim.ckpt_mode=resume",  # every epoch kept, weights only; resume.pt holds the full state
        # data
        "data.train_stride=1", "data.window_to_last_row=true",
        "data.stats_max_rows=1000000000", f"data.stats_path={task_h5(task).with_suffix('.full-stats.npz')}",
        # model
        "model.vit_impl=hf", "model.vit_patch=14", "model.act_smooth_dim=10", "model.pos_init_std=1.0",
        *model_overrides(task, model),
    ]


def short_recipe(task: str, model: str) -> list[str]:
    """The first experiment: 10 x 2000 steps; dino_cls with DINO-WM's optimiser (batch 32, lr 5e-4, constant)."""
    common = [f"optim.epochs={EPOCHS}", "optim.steps_per_epoch=2000", "optim.warmup_epochs=0.1",
              "optim.min_lr_scale=0.0"]
    if model == "dino_cls":
        common += ["optim.batch_size=32", "optim.lr=5e-4", "optim.weight_decay=0.01", "optim.warmup_epochs=0",
                   "optim.min_lr_scale=1.0", "optim.grad_clip=inf"]
    return [*common, *model_overrides(task, model)]


RECIPES = {
    "subjepa": Recipe("compare_subjepa", subjepa_recipe),
    "short": Recipe("compare", short_recipe),
}

DATA = {
    "tworoom": {"repo": "quentinll/lewm-tworooms", "file": "tworoom.tar.zst", "kind": "tar", "gb": 14},
    "pusht": {"repo": "quentinll/lewm-pusht", "file": "pusht_expert_train.h5.zst", "kind": "zst", "gb": 57},
}

SMOKE = {
    "train": ["optim.epochs=1", "optim.steps_per_epoch=5", "optim.val_steps=2", "data.max_episodes=200",
              "optim.num_workers=2", "wandb.enabled=false"],
    "analyse": ["--windows", "64", "--min-windows", "16", "--bank-episodes", "40", "--heldout-frames", "400",
                "--pairs", "5000", "--tc-frames", "300", "--workers", "2"],
    "cem": ["--seeds", "42", "--num-eval", "2", "--samples", "30", "--iters", "2", "--topk", "5"],
}


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def python(script: str, *args) -> None:
    cmd = [sys.executable, script, *map(str, args)]
    print(f"\n$ python {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def skip(name: str, output: Path, force: bool) -> bool:
    if output.exists() and not force:
        print(f"[{name}] skipped: {output} exists")
        return True
    return False


def epoch_ckpt(run_dir: Path, epoch: int) -> Path:
    """A run's checkpoint of `epoch`; runs pruned by the first experiment kept some in keep/."""
    name = f"epoch_{epoch:03d}.pt"
    return run_dir / name if (run_dir / name).exists() else run_dir / "keep" / name


def eval_dir(run_dir: Path, epoch: int, last: int) -> Path:
    """Where the evaluations of `epoch` go: the run dir for the last epoch, <run>/epochNNN/ otherwise."""
    return run_dir if epoch == last else run_dir / f"epoch{epoch:03d}"


# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
def setup() -> None:
    if importlib.util.find_spec("stable_worldmodel") is not None:
        print("[setup] dependencies present")
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)


def zstd_bin() -> str:
    """zstd on PATH, else the conda base install's (an activated env usually hides it)."""
    found = shutil.which("zstd")
    base = Path(sys.executable).resolve().parents[3] / "bin" / "zstd"  # <conda>/envs/<env>/bin/python
    if found is None and base.exists():
        found = str(base)
    if found is None:
        sys.exit("[data] zstd not found (conda install zstd, or apt install zstd)")
    return found


def download(task: str) -> None:
    """LeWM's recording -> task_h5(task); the archive is deleted once unpacked."""
    target = task_h5(task)
    if target.exists():
        print(f"[data] {task}: {target} exists")
        return
    spec = DATA[task]
    dest = target.parent
    dest.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(dest).free / 1e9
    if free_gb < spec["gb"]:
        sys.exit(f"[data] {task} needs ~{spec['gb']} GB free at its peak (archive + unpacked), {free_gb:.0f} GB left")
    zstd = zstd_bin()

    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", local_dir=dest / "_download"))
    print(f"[data] {task}: downloaded {archive} ({archive.stat().st_size / 1e9:.1f} GB), unpacking", flush=True)
    if spec["kind"] == "zst":
        subprocess.run([zstd, "-d", "--rm", "-f", str(archive), "-o", str(target)], check=True)
    else:
        unpack = dest / "_unpack"
        unpack.mkdir(exist_ok=True)
        subprocess.run(["tar", "-I", zstd, "-xf", str(archive), "-C", str(unpack)], check=True)
        archive.unlink()
        found = sorted(unpack.rglob("*.h5"))
        if len(found) != 1:
            sys.exit(f"[data] expected one .h5 in {archive.name}, found {[str(f) for f in found]}")
        found[0].replace(target)
        shutil.rmtree(unpack)
    shutil.rmtree(dest / "_download", ignore_errors=True)
    print(f"[data] {task}: {target} ({target.stat().st_size / 1e9:.1f} GB)")


def train(task: str, model: str, recipe: str, run_root: Path, last: int, args, smoke: dict) -> None:
    run_dir = run_root / task / model
    if skip(f"train {task}/{model}", run_dir / f"epoch_{last:03d}.pt", args.force):
        return
    overrides = [f"data.task={task}", f"out_dir={run_root / task}", f"exp_id={model}",
                 f"wandb.name={task}_{model}_{recipe}", *RECIPES[recipe].overrides(task, model),
                 *smoke.get("train", []), *(["wandb.enabled=false"] if args.no_wandb else [])]
    # resume.pt holds the full state when the epoch files are weights only (optim.ckpt_mode=resume)
    done = sorted(run_dir.glob("epoch_*.pt"))
    latest = run_dir / "resume.pt" if (run_dir / "resume.pt").exists() else (done[-1] if done else None)
    resume = ["--resume", "--ckpt", latest] if latest and not args.force else []
    python("train.py", "--preset", MODELS[model][0], *resume, *overrides)
    if args.prune_epochs:
        for old in sorted(run_dir.glob("epoch_*.pt"))[:-1]:
            old.unlink()


def evaluate(task: str, model: str, run_root: Path, epoch: int, last: int, args, smoke: dict) -> None:
    run_dir = run_root / task / model
    ckpt, out = epoch_ckpt(run_dir, epoch), eval_dir(run_dir, epoch, last)
    if not ckpt.exists():
        print(f"[{task}/{model}] {ckpt.name} missing; skipping its evaluations")
        return
    if "analyse" in args.stages and not skip(f"analyse {task}/{model}", out / "analysis" / "analysis.json", args.force):
        python("compare/analysis.py", "--ckpt", ckpt, "--out", out / "analysis", "--horizon", args.horizon,
               *smoke.get("analyse", []))
    if "cem" in args.stages:
        if args.force or args.smoke or not all((out / "cem" / f"seed{s}.json").exists() for s in args.seeds):
            cem = smoke.get("cem", ["--seeds", *args.seeds])  # plan.py runs only the seeds not done yet
            python("compare/plan.py", "--ckpt", ckpt, "--out", out / "cem", *cem,
                   *(["--video"] if args.cem_video else []), *(["--redo"] if args.force else []))
        else:
            print(f"[cem {task}/{model}] skipped: seeds {list(args.seeds)} done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", choices=sorted(RECIPES), default="subjepa", help="how the models are trained")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--epoch", type=int, default=None, help="checkpoint to evaluate (default: the last)")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="CEM evaluation seeds")
    p.add_argument("--horizon", type=int, default=15, help="rollout horizon of the analysis (latent steps)")
    p.add_argument("--force", action="store_true", help="redo stages whose output exists")
    p.add_argument("--prune-epochs", action="store_true", help="delete a finished run's epochs except the last")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--cem-video", action="store_true", help="also save CEM episode videos")
    p.add_argument("--smoke", action="store_true", help="tiny settings, written to <root>/_smoke")
    args = p.parse_args()

    smoke = SMOKE if args.smoke else {}
    run_root = ROOT / "checkpoints" / RECIPES[args.recipe].root / ("_smoke" if args.smoke else "")
    last = 0 if args.smoke else EPOCHS - 1
    epoch = last if args.epoch is None else args.epoch
    print(f"recipe {args.recipe} | tasks {args.tasks} | models {args.models} | stages {args.stages} | epoch {epoch}"
          f" | {run_root} | data in {HOME / 'datasets'}")

    if "setup" in args.stages:
        setup()
    if "data" in args.stages:
        for task in args.tasks:
            download(task)
    if "train" in args.stages:
        for task in args.tasks:
            for model in args.models:
                train(task, model, args.recipe, run_root, last, args, smoke)
    if "analyse" in args.stages or "cem" in args.stages:
        for task in args.tasks:
            for model in args.models:
                evaluate(task, model, run_root, epoch, last, args, smoke)
    if "report" in args.stages:
        where = [] if epoch == last else ["--subdir", f"epoch{epoch:03d}", "--out", run_root / f"report_epoch{epoch:03d}"]
        python("compare/report.py", "--root", run_root, "--tasks", *args.tasks, "--models", *args.models, *where)


if __name__ == "__main__":
    main()
