"""One task end to end: data -> world model -> probe -> decoder -> rollout ablation.

    python run.py --task acrobot-swingup                  # train up to epoch_006.pt, analyse it
    python run.py --task cartpole-swingup --stop-epoch 9
    python run.py --task reacher-easy --stages probe decoder rollout   # only the analyses
    python run.py --task pendulum-swingup --force                       # redo stages whose output exists
    python run.py --task acrobot-swingup --smoke                        # tiny end-to-end check (minutes)

Stages, in order (each one is a plain script you can also run by hand):

    data     datasets/prepare_mmbench.py   MMBench tasks: training table + the val/test table, if
                                           missing; cube-single reads the LeWM OGBench h5 as is
    train    train.py                      epochs 0..--stop-epoch of 2000 steps each, with the LR
                                           schedule of an `optim.epochs` (100) epoch run; resumes
    probe    ablations/probe.py            linear probe on epoch_<stop-epoch>, plus the untrained baseline
    decoder  ablations/decoder.py          pixel decoder for the videos
    rollout  ablations/rollout.py          copy / teacher-forced / autoregressive / shuffled actions,
                                           on the held-out + val/test episodes, with --videos videos

Everything of one run lives in checkpoints/<task>/<preset>/: epoch_NNN.pt,
train.log, and probe/ probe_random/ decoder/ rollout/. A stage whose output
already exists is skipped unless --force.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import MMBENCH_TASKS, PRESETS, TASKS, heldout_h5, task_h5

ROOT = Path(__file__).resolve().parent
STAGES = ("data", "train", "probe", "decoder", "rollout")

# Per-task analysis settings. cube-single has 10k short episodes: 41 latent
# frames each (a 30-step horizon fits, 50 does not), and every 10th training
# row is plenty for the probe and the decoder.
TASK_ARGS = {
    "cube-single": {
        "probe": ["--row-stride", "10"],
        "decoder": ["--row-stride", "10"],
        "rollout": ["--horizon", "30"],
    },
    # pusht: 18.7k episodes of 49-246 rows (median 123); a 15-step horizon fits 82% of them.
    "pusht": {
        "probe": ["--row-stride", "10"],
        "decoder": ["--row-stride", "10"],
        "rollout": ["--horizon", "15"],
    },
}

# --smoke: a few steps of everything, written to checkpoints/_smoke/
SMOKE = {
    "train": ["data.max_episodes=40", "data.val_episodes=8", "optim.steps_per_epoch=10", "optim.val_steps=5",
              "optim.num_workers=2"],
    "probe": ["--row-stride", "25", "--workers", "2"],
    "decoder": ["--steps", "200", "--row-stride", "25"],
    "rollout": ["--horizon", "10", "--start-stride", "10"],
}


def python(script: str, *args) -> None:
    """Run one of the repo's scripts with this interpreter; stop everything if it fails."""
    cmd = [sys.executable, script, *map(str, args)]
    print(f"\n$ python {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def skip(name: str, output: Path, force: bool) -> bool:
    if output.exists() and not force:
        print(f"\n[{name}] skipped: {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output} exists")
        return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--preset", default="vit", choices=sorted(PRESETS))
    p.add_argument("--stop-epoch", type=int, default=6,
                   help="train epochs 0..N (LR schedule of a 100-epoch run); the analyses use epoch_N")
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--videos", type=int, default=4, help="decoded rollout videos")
    p.add_argument("--force", action="store_true", help="redo stages whose output already exists")
    p.add_argument("--smoke", action="store_true", help="tiny settings, written to checkpoints/_smoke/")
    args = p.parse_args()

    stop = 0 if args.smoke else args.stop_epoch
    run_dir = ROOT / "checkpoints" / ("_smoke" if args.smoke else "") / args.task / args.preset
    ckpt = run_dir / f"epoch_{stop:03d}.pt"
    train_h5, extra_h5 = task_h5(args.task), heldout_h5(args.task)
    smoke = SMOKE if args.smoke else {}

    def stage_args(stage: str) -> list[str]:
        """Task settings first, then smoke settings (argparse keeps the last value)."""
        return [*TASK_ARGS.get(args.task, {}).get(stage, []), *smoke.get(stage, [])]
    print(f"{args.task} | preset {args.preset} | epochs 0..{stop} | {run_dir}")

    if "data" in args.stages:
        if args.task not in MMBENCH_TASKS:
            if not train_h5.exists():
                sys.exit(f"{train_h5} not found; {args.task} is not converted by this repo")
            print(f"\n[data] {args.task}: using {train_h5}")
        else:
            if not skip("data", train_h5, False):
                python("datasets/prepare_mmbench.py", "--task", args.task)
            if not skip("data", extra_h5, False):
                python("datasets/prepare_mmbench.py", "--task", args.task, "--heldout")

    if "train" in args.stages and not skip("train", ckpt, args.force):
        overrides = [f"data.task={args.task}", f"out_dir={run_dir.parent}", f"exp_id={args.preset}",
                     f"optim.stop_epoch={stop}", *smoke.get("train", [])]
        done = sorted(run_dir.glob("epoch_*.pt"))
        resume = ["--resume", "--ckpt", done[-1]] if done and not args.force else []
        python("train.py", "--preset", args.preset, *resume, *overrides)

    if any(s in args.stages for s in ("probe", "decoder", "rollout")) and not ckpt.exists():
        sys.exit(f"{ckpt} does not exist; run the train stage first")

    if "probe" in args.stages:
        if not skip("probe", run_dir / "probe" / "probe.pt", args.force):
            python("ablations/probe.py", "--ckpt", ckpt, *stage_args("probe"))
        if not skip("probe", run_dir / "probe_random" / "probe.pt", args.force):
            python("ablations/probe.py", "--ckpt", ckpt, "--random-init", *stage_args("probe"))

    if "decoder" in args.stages and not skip("decoder", run_dir / "decoder" / "decoder.pt", args.force):
        python("ablations/decoder.py", "--ckpt", ckpt, *stage_args("decoder"))

    if "rollout" in args.stages and not skip("rollout", run_dir / "rollout" / "rollout.json", args.force):
        videos = ["--videos", 1 if args.smoke else args.videos] if args.videos else []
        extra = ["--extra-h5", extra_h5] if extra_h5 else []
        python("ablations/rollout.py", "--ckpt", ckpt, *extra, *videos, *stage_args("rollout"))

    print(f"\ndone: {run_dir}")


if __name__ == "__main__":
    main()
