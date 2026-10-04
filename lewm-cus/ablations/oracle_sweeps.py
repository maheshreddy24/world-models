"""Oracle-state ablation on cube-single: how much does each part of the state matter?

Every experiment hides one group of the 28-d oracle state from the world model
and runs the whole pipeline on what is left:

    stage 1  train.py         MLP world model                     --epochs (7)
    stage 2  train_policy.py  diffusion policy on that frozen model  --policy-epochs (7)
    eval     eval.py          diffusion policy and CEM-MPC on the held-out episodes

    python ablations/oracle_sweeps.py                    # train everything, then evaluate everything
    python ablations/oracle_sweeps.py --phase train      # both stages for every experiment, no eval
    python ablations/oracle_sweeps.py --phase eval       # evaluate the latest sweep
    python ablations/oracle_sweeps.py --phase report     # rebuild the table from whatever has finished
    python ablations/oracle_sweeps.py --only full,no_joint_vel --dry-run

    # a second policy on an existing sweep's world models, next to the first one
    python ablations/oracle_sweeps.py --sweep-dir checkpoints/oracle_sweep_<t> --policy-tag xyz

Each step is the command you would type yourself, run as a subprocess, so a
crash in one experiment leaves the others running and every checkpoint works
with the normal scripts afterwards. A step whose output already exists is
skipped, so rerunning after an interruption carries on where it stopped
(delete an experiment's folder to redo it).

`--policy-tag` puts the diffusion policy and its eval in `policy_<tag>/` and
`eval_diffusion_<tag>/` and the table in `summary_<tag>.md`. The world models
and the MPC evals depend on nothing policy-side, so on an existing sweep they
are found and skipped, and the untagged policy's results stay where they were.

The state is cut inside the encoder (`model.drop_obs`, src/models/encoder.py):
data, normaliser and eval tasks are identical to the full run, and every
experiment is scored on the same held-out tasks, so the numbers compare pairwise.

    checkpoints/oracle_sweep_<unix time>/
      summary.md, summary.json          the results table, rewritten after every eval
      <experiment>/
        world_model/epoch_006.pt ...    stage 1; train.log holds the val metrics
        policy/epoch_006.pt ...         stage 2
        eval_diffusion/eval/results.json + videos
        eval_mpc/eval/results.json + videos
        policy_<tag>/, eval_diffusion_<tag>/    with --policy-tag
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.ogbench import OBS_DIM, kept_obs_dims  # noqa: E402

# name -> groups of src.data.ogbench.OBS_GROUPS hidden from the world model.
EXPERIMENTS = {
    "full": (),
    "no_joint_vel": ("joint_vel",),
    "no_gripper_contact": ("gripper_contact",),
    "no_cube_orientation": ("cube_quat", "cube_yaw"),
    "no_arm_state": ("joint_pos", "joint_vel", "ee_pos", "ee_yaw"),
    "no_cube_state": ("cube_pos", "cube_quat", "cube_yaw"),
}
# The diffusion policy also reads proprio straight from the raw state
# (`policy.use_proprio`), around the world model: end-effector position +
# velocity for `policy.proprio=ee`, joint pos + joint vel + gripper opening for
# "arm". Where that is the removed information it goes too, or the policy would
# get it back that way. no_joint_vel is left on deliberately: with the arm
# proprio its policy still reads the joint velocities, so there the experiment
# asks whether the world-model latent needs them when the policy has them.
NO_PROPRIO = {"no_arm_state"}

PLANNERS = ("diffusion", "mpc")  # eval order: the diffusion evals take seconds, MPC minutes
VAL_LINE = re.compile(r"epoch\s+(\d+) \| val (\w+) ([-+.\deE]+)")


# --------------------------------------------------------------------------- #
#  Paths and commands
# --------------------------------------------------------------------------- #
def world_model_ckpt(root: Path, epochs: int) -> Path:
    return root / "world_model" / f"epoch_{epochs - 1:03d}.pt"


def tagged(name: str, tag: str) -> str:
    return f"{name}_{tag}" if tag else name


def policy_dir(root: Path, tag: str = "") -> Path:
    return root / tagged("policy", tag)


def policy_ckpt(root: Path, epochs: int, tag: str = "") -> Path:
    return policy_dir(root, tag) / f"epoch_{epochs - 1:03d}.pt"


def eval_dir(root: Path, planner: str, tag: str = "") -> Path:
    # Only the diffusion eval depends on the policy; MPC is the world model's alone.
    return root / tagged(f"eval_{planner}", tag if planner == "diffusion" else "")


def eval_results(root: Path, planner: str, tag: str = "") -> Path:
    return eval_dir(root, planner, tag) / "eval" / "results.json"


def world_model_cmd(name: str, root: Path, args) -> list[str]:
    cmd = [
        sys.executable, "train.py", "--preset", "cube_oracle",
        f"run_name=oracle_sweep_{name}", f"out_dir={root}", "exp_id=world_model",
        f"optim.epochs={args.epochs}",
        # One pass over the stride-13 windows per epoch, as checkpoints/oracle was trained.
        "optim.steps_per_epoch=none",
    ]
    if EXPERIMENTS[name]:
        cmd.append(f"model.drop_obs={','.join(EXPERIMENTS[name])}")
    return cmd


def policy_cmd(name: str, root: Path, args) -> list[str]:
    cmd = [
        sys.executable, "train_policy.py", "--ckpt", str(world_model_ckpt(root, args.epochs)),
        f"run_name=oracle_sweep_{tagged(name + '_policy', args.policy_tag)}", f"out_dir={root}",
        f"exp_id={policy_dir(root, args.policy_tag).name}", f"optim.epochs={args.policy_epochs}",
    ]
    if name in NO_PROPRIO:
        cmd.append("policy.use_proprio=false")
    return cmd


def eval_cmd(planner: str, root: Path, args) -> list[str]:
    if planner == "diffusion":
        cmd = [sys.executable, "eval.py", "plan.planner=diffusion",
               "--diffusion-ckpt", str(policy_ckpt(root, args.policy_epochs, args.policy_tag))]
    else:
        cmd = [sys.executable, "eval.py", "--ckpt", str(world_model_ckpt(root, args.epochs))]
    out = eval_dir(root, planner, args.policy_tag)
    return cmd + ["--split", "val", "--out", str(out), f"eval.num_eval={args.num_eval}"]


def run(label: str, cmd: list[str], output: Path, dry_run: bool, failures: list[str]) -> bool:
    """Run one step unless `output` already exists. True when `output` is there afterwards."""
    if output.exists():
        print(f"skip  | {label}: {output} exists", flush=True)
        return True
    print(f"\n==> {label}\n    {shlex.join(cmd)}", flush=True)
    if dry_run:
        return True
    started = time.time()
    code = subprocess.run(cmd, cwd=REPO).returncode
    if code != 0 or not output.exists():
        failures.append(label)
        print(f"FAIL  | {label}: exit code {code}, {output} not written", flush=True)
        return False
    print(f"done  | {label} in {(time.time() - started) / 60:.1f} min", flush=True)
    return True


# --------------------------------------------------------------------------- #
#  Report
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def last_val(log: Path) -> dict | None:
    """The final `epoch N | val <metric> <value>` line of a train.py / train_policy.py log."""
    if not log.exists():
        return None
    matches = VAL_LINE.findall(log.read_text())
    if not matches:
        return None
    epoch, metric, value = matches[-1]
    return {"epoch": int(epoch), "metric": metric, "value": float(value)}


def success_cell(res: dict | None) -> str:
    """Success rate with one binomial standard error, e.g. `88.0 ±4.6`."""
    if res is None:
        return "—"
    p, n = res["success_rate"] / 100, res["num_eval"]
    return f"{res['success_rate']:.1f} ±{100 * math.sqrt(p * (1 - p) / n):.1f}"


def delta_cell(res: dict | None, base: dict | None) -> str:
    if res is None or base is None:
        return "—"
    return f"{res['success_rate'] - base['success_rate']:+.1f}"


def policy_cond(policy_cfg: dict | None) -> str:
    """`goal / proprio` of a policy run, e.g. `cube_xyz / arm`."""
    if policy_cfg is None:
        return "—"
    p = policy_cfg["policy"]
    # Configs saved before policy.goal / policy.proprio existed: goal latent, ee proprio.
    proprio = p.get("proprio", "ee") if p["use_proprio"] else "off"
    cond = f"{p.get('goal', 'latent')} / {proprio}"
    return cond + " + contact" if p.get("use_contact") else cond


def report(sweep: Path, tag: str = "") -> None:
    """Collect every experiment of the sweep into summary[_<tag>].json and .md.

    A tagged report covers the experiments that have that policy, and sets the
    untagged policy's success beside it for comparison.
    """
    rows = {}
    for name, drop in EXPERIMENTS.items():
        root = sweep / name
        if not root.is_dir() or (tag and not policy_dir(root, tag).is_dir()):
            continue
        rows[name] = {
            "drop": list(drop),
            "kept_dims": kept_obs_dims(",".join(drop)),
            "policy_cond": policy_cond(read_json(policy_dir(root, tag) / "config.json")),
            "world_model_val": last_val(root / "world_model" / "train.log"),
            "policy_val": last_val(policy_dir(root, tag) / "train.log"),
            **{planner: read_json(eval_results(root, planner, tag)) for planner in PLANNERS},
        }
        if tag:
            rows[name]["untagged_diffusion"] = read_json(eval_results(root, "diffusion"))
        for key in (*PLANNERS, "untagged_diffusion"):
            if rows[name].get(key):
                rows[name][key].pop("episodes", None)
    if not rows:
        return

    base = rows.get("full", {})
    untagged = "| untagged DP " if tag else ""
    lines = [
        f"# Oracle-state ablation — {sweep.name}" + (f", policy `{tag}`" if tag else ""),
        "",
        "Success rate (%) on held-out episodes, ± one binomial standard error. "
        "Δ is against `full`, on the same tasks."
        + (" `untagged DP` is the `policy/` of the same experiment." if tag else ""),
        "",
        "| experiment | removed | dims | DP goal / proprio | WM val pred/static | DP val action_mse "
        f"| DP success | Δ DP {untagged}| MPC success | Δ MPC |",
        "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if tag else ""),
    ]
    for name, r in rows.items():
        wm, dp = (f"{v['value']:.4f}" if v else "—" for v in (r["world_model_val"], r["policy_val"]))
        lines.append(
            f"| {name} | {', '.join(r['drop']) or '—'} | {len(r['kept_dims'])}/{OBS_DIM} "
            f"| {r['policy_cond']} | {wm} | {dp} "
            f"| {success_cell(r['diffusion'])} | {delta_cell(r['diffusion'], base.get('diffusion'))} "
            + (f"| {success_cell(r['untagged_diffusion'])} " if tag else "")
            + f"| {success_cell(r['mpc'])} | {delta_cell(r['mpc'], base.get('mpc'))} |"
        )
    table = "\n".join(lines) + "\n"

    (sweep / f"{tagged('summary', tag)}.md").write_text(table)
    (sweep / f"{tagged('summary', tag)}.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n" + table, flush=True)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def resolve_sweep(args) -> Path:
    """An explicit --sweep-dir, a fresh one when training, otherwise the latest sweep."""
    if args.sweep_dir:
        return Path(args.sweep_dir).resolve()
    if args.phase in ("all", "train"):
        return REPO / "checkpoints" / f"oracle_sweep_{int(time.time())}"
    sweeps = sorted((REPO / "checkpoints").glob("oracle_sweep_*"))
    if not sweeps:
        raise SystemExit(f"no checkpoints/oracle_sweep_* to {args.phase}; pass --sweep-dir")
    return sweeps[-1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="all", choices=("all", "train", "eval", "report"))
    parser.add_argument("--sweep-dir", default=None, help="default: new dir for all/train, latest sweep otherwise")
    parser.add_argument("--only", default=None, help=f"comma-separated subset of {', '.join(EXPERIMENTS)}")
    parser.add_argument("--epochs", type=int, default=7, help="world-model epochs (stage 1)")
    parser.add_argument("--policy-epochs", type=int, default=7, help="diffusion-policy epochs (stage 2)")
    parser.add_argument("--num-eval", type=int, default=50, help="held-out tasks per evaluation")
    parser.add_argument("--policy-tag", default="", help="train/eval the policy in policy_<tag>/, beside policy/")
    parser.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = parser.parse_args(argv)

    names = list(EXPERIMENTS) if not args.only else [n.strip() for n in args.only.split(",")]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments {unknown}; expected some of {list(EXPERIMENTS)}")
    sweep = resolve_sweep(args)

    print(f"sweep | {sweep}" + (f" | policy tag {args.policy_tag}" if args.policy_tag else ""))
    for name in names:
        keep = kept_obs_dims(",".join(EXPERIMENTS[name]))
        print(
            f"  {name:<20} removes {', '.join(EXPERIMENTS[name]) or 'nothing':<42}"
            f" -> {len(keep):2d}/{OBS_DIM} dims, proprio {'off' if name in NO_PROPRIO else 'on'}"
        )

    failures: list[str] = []
    if args.phase in ("all", "train"):
        for name in names:
            root = sweep / name
            if run(f"{name}: world model", world_model_cmd(name, root, args),
                   world_model_ckpt(root, args.epochs), args.dry_run, failures):
                run(f"{name}: diffusion policy", policy_cmd(name, root, args),
                    policy_ckpt(root, args.policy_epochs, args.policy_tag), args.dry_run, failures)

    if args.phase in ("all", "eval"):
        for planner in PLANNERS:
            for name in names:
                root = sweep / name
                needs = (
                    policy_ckpt(root, args.policy_epochs, args.policy_tag)
                    if planner == "diffusion"
                    else world_model_ckpt(root, args.epochs)
                )
                if not args.dry_run and not needs.exists():
                    print(f"skip  | {name}: {planner} eval, {needs} missing", flush=True)
                    continue
                run(f"{name}: {planner} eval", eval_cmd(planner, root, args),
                    eval_results(root, planner, args.policy_tag), args.dry_run, failures)
                if not args.dry_run:
                    report(sweep, args.policy_tag)

    if not args.dry_run:
        report(sweep, args.policy_tag)
        print(f"summary | {sweep / tagged('summary', args.policy_tag)}.md")
    if failures:
        print("failed  | " + "; ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
