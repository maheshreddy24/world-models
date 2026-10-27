"""ConveyorCupGoal-v0 end to end: data -> train -> probe -> decoder -> rollout -> plan.

    python conveyor_goal.py                                   # everything, ViT, up to epoch_006.pt
    python conveyor_goal.py --stop-epoch 9
    python conveyor_goal.py --stages plan --plan-episodes 5   # only (part of) the planner
    python conveyor_goal.py --stages probe rollout --force    # redo some analyses
    python conveyor_goal.py --smoke                           # few-minute end-to-end check

Scenario (dynamic-environments, `scripts/collect_demos.py ConveyorCupGoal-v0`): a Panda
picks a red cup off a moving conveyor and carries it to a random goal position above the
table, then holds it there until the 150-step (5 s, 30 Hz) episode ends. `reached_step` is
the first frame with the cup at the goal; the frames after it are the hold ("padding").

This script adds the task to the repo at runtime (config.TASKS, config.TASK_DATA,
src.tasks.TASKS) and runs the repo's own scripts unchanged; nothing in the base code is
edited. Stages, each skipped when its output exists (unless --force):

    data     the expert-success episodes of --raw, cut to rows 0..reached_step (the hold
             is dropped), written as the HDF5 table train.py reads
    train    train.py, epochs 0..--stop-epoch of 2000 steps (LR schedule of a 100-epoch run)
    probe    ablations/probe.py: latent -> cup xyz, gripper (TCP) xyz, gripper opening
    decoder  ablations/decoder.py: latent -> 112px frame (for the videos)
    rollout  ablations/rollout.py on the held-out episodes, a decoded video for every one
    plan     goal-conditioned MPC in the simulator, on the same held-out episodes

Held-out episodes: the world model's own split (data.val_episodes = 10% of the episodes,
seed 3072). They are never trained on; probe, rollout and planner are scored on them.

Data columns (one row = one 30 Hz control step):

    pixels       (N, 224, 224, 3)  agentview camera, before the action
    observation  (N, 24)           TCP xyz (0:3), quat (3:7), gripper width (7), 7 joints (8:15),
                                   cup xyz (15:18), cup velocity (18:21), goal xyz (21:24)
    action       (N, 4)            target TCP - current TCP (xyz, metres), grip (-1 open / +1 closed);
                                   NaN on each episode's last row (the goal frame)

The expert commands absolute targets (`ee_abs`) with one fixed orientation, so the action
is that target relative to where the TCP is now. Executed as `ee_abs` with target =
TCP + offset it replays the expert exactly (the env is deterministic per seed); `ee_delta`
cannot, it rate-limits the target to 3 cm / 0.15 rad a step and the grasp of the moving cup
misses.

Planner (`plan`): for each held-out episode the env is reset with the episode's seed (same
cup motion, goal, lighting, camera; the first frame matches the recording bit for bit) and
the goal image is the recorded frame at `reached_step`. Every latent step (frameskip = 5
control steps) CEM searches --plan-horizon action blocks in the normalised action space:
the predictor rolls out from the last `history` encoded frames (fewer at the start) and the
executed blocks between them, and the cost is the latent distance of the last predicted
frame to the encoded goal (--plan-cost all: averaged over the horizon). The first block is
executed, the rest warm-starts the next search. Samples are clipped to the range of the
recorded actions.

Success: the cup within --success-tol (default 2 cm, the env's goal_tolerance) of the goal
at any step; also reported at 5 and 10 cm, at the final step, and the env's own flag (held in
the gripper and within 2 cm). As a check of the action interface, each episode's recorded
actions are also replayed open loop (`replay_success`, should be 100%).

Outputs in checkpoints/conveyor-cup-goal/<preset>/ (as run.py): epoch_NNN.pt, train.log,
probe/ probe_random/ decoder/ rollout/ and

    planner/planner.json             success rates and per-episode results
    planner/episodes/ep<i>.json      per step: cup-goal distance, latent distance, plan cost
    planner/episodes/ep<i>.mp4       camera | goal image | decoded predicted end of the current plan
    planner/episodes/ep<i>.png       12 frames of the same
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import tasks as task_specs  # noqa: E402

TASK = "conveyor-cup-goal"
SCENARIO = "ConveyorCupGoal-v0"
ENV_ROOT = Path("/home/mahesh_tapeti/dynamic-environments")
RAW = ENV_ROOT / "data" / "ConveyorCupGoal"
H5 = config.HOME / "datasets" / "dynamic" / f"{TASK}.h5"
QUAT = np.array([0.0, 0.70710677, 0.70710677, 0.0])  # the expert's (only) TCP orientation
STAGES = ("data", "train", "probe", "decoder", "rollout", "plan")
TOLERANCES = (0.02, 0.05, 0.10)  # m, success reported at each


# --------------------------------------------------------------------------- #
#  The task, added to the repo at runtime
# --------------------------------------------------------------------------- #
class ConveyorCupGoalTask(task_specs.Task):
    """Where the cup and the gripper are (3-D distances, cm), and how far the gripper is open (cm)."""

    name = TASK
    quantities = (task_specs.Quantity("cup", "position"), task_specs.Quantity("effector", "position"),
                  task_specs.Quantity("gripper", "position"))

    def targets(self, obs):
        return obs[..., [15, 16, 17, 0, 1, 2, 7]]

    def read(self, t):
        return t

    def errors(self, pred_targets, true_targets):
        import torch

        diff = self.read(pred_targets) - self.read(true_targets)
        cup, eff = diff[..., 0:3].norm(dim=-1), diff[..., 3:6].norm(dim=-1)
        return torch.stack([cup, eff, diff[..., 6].abs()], -1) * 100.0


def register(h5: Path = H5) -> None:
    """Make TASK known to config (data.task, its data settings) and to src.tasks (the probe spec)."""
    if TASK not in config.TASKS:
        config.TASKS = (*config.TASKS, TASK)
    config.TASK_DATA[TASK] = {
        "data.h5_path": str(h5),
        "data.val_episodes": 1_000_000,  # split_episodes caps it at len // 10: 10% of the episodes
        "data.train_stride": 1,  # short episodes (57-135 rows): every window start
        "data.val_stride": 1,
    }
    task_specs.TASKS[TASK] = ConveyorCupGoalTask()


def exec_script(script: str, argv: list[str]) -> None:
    """Run one of the repo's scripts in this process, with TASK registered."""
    register(Path(os.environ.get("CONVEYOR_H5", H5)))
    path = ROOT / script
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *argv]
    runpy.run_path(str(path), run_name="__main__")


def python(script: str, *args, h5: Path = H5) -> None:
    """Run a repo script (as run.py does), through this file so the task is registered."""
    cmd = [sys.executable, __file__, "_exec", script, *map(str, args)]
    print(f"\n$ python {script} {' '.join(map(str, args))}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "CONVEYOR_H5": str(h5)})


def skip(name: str, output: Path, force: bool) -> bool:
    if output.exists() and not force:
        print(f"\n[{name}] skipped: {output} exists")
        return True
    return False


# --------------------------------------------------------------------------- #
#  data: expert successes, up to the goal frame -> HDF5
# --------------------------------------------------------------------------- #
def load_raw_episode(path: Path) -> dict:
    """One episode cut to rows 0..reached_step, in the training table's columns."""
    ep = np.load(path)
    r = int(ep["reached_step"])
    rows = slice(0, r + 1)
    states, cup, act = ep["states"][rows], ep["object_states"][rows], ep["actions_abs"][rows]
    goal = np.broadcast_to(ep["goal"], (r + 1, 3))
    action = np.concatenate([act[:, :3] - states[:, :3], act[:, 7:8]], 1).astype(np.float32)
    action[-1] = np.nan  # the goal frame: nothing follows it
    return {
        "pixels": ep["images"][rows],
        "observation": np.concatenate([states, cup, goal], 1).astype(np.float32),
        "action": action,
        "reward": ep["rewards"][rows].astype(np.float32),
        "seed": int(ep["seed"]), "goal": ep["goal"].astype(np.float32), "reached_step": r,
        "grasp_step": int(ep["grasp_step"]),
    }


def prepare(raw: Path, out: Path) -> None:
    import h5py
    import hdf5plugin
    from tqdm import tqdm

    # The episode files themselves, not manifest.json: a collection stopped before its end
    # leaves the manifest stale (and possibly a truncated last file).
    keep, skipped = [], {}
    for path in tqdm(sorted(raw.glob("ep_*.npz")), desc="scan", dynamic_ncols=True):
        try:
            ep = np.load(path)
            if str(ep["scenario"]) != SCENARIO:
                sys.exit(f"{path} holds {ep['scenario']}, not {SCENARIO}")
            ok, r, steps = bool(ep["success"]), int(ep["reached_step"]), len(ep["rewards"])
            size = int(ep["images"].shape[1]) if not keep else size
        except Exception as e:  # truncated / unreadable file
            skipped.setdefault(f"unreadable ({type(e).__name__})", []).append(path.name)
            continue
        if not ok:
            skipped.setdefault("expert failed", []).append(path.name)
        elif not 0 <= r < steps:  # reached only after the last recorded frame: no goal image
            skipped.setdefault("no goal frame", []).append(path.name)
        else:
            keep.append({"file": path.name, "reached_step": r})
    if not keep:
        sys.exit(f"no usable episodes in {raw}")
    lens = [e["reached_step"] + 1 for e in keep]
    n = int(sum(lens))
    print(f"{SCENARIO}: {len(keep)} usable expert successes, {n} rows up to the goal "
          f"(episodes of {min(lens)}-{max(lens)} rows) -> {out}")
    for why, files in skipped.items():
        print(f"  skipped {len(files)} ({why}): {', '.join(files[:5])}{' ...' if len(files) > 5 else ''}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.h5")
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    with h5py.File(tmp, "w") as f:
        cols = {
            "pixels": f.create_dataset("pixels", (n, size, size, 3), np.uint8, chunks=(100, size, size, 3), **blosc),
            "observation": f.create_dataset("observation", (n, 24), np.float32),
            "action": f.create_dataset("action", (n, 4), np.float32),
            "reward": f.create_dataset("reward", (n,), np.float32),
        }
        meta = {k: [] for k in ("seed", "goal", "reached_step", "grasp_step")}
        row = 0
        for e in tqdm(keep, desc="episodes", dynamic_ncols=True):
            ep = load_raw_episode(raw / e["file"])
            m = ep["reached_step"] + 1
            for k, col in cols.items():
                col[row : row + m] = ep[k]
            for k in meta:
                meta[k].append(ep[k])
            row += m
        f.create_dataset("ep_offset", data=np.cumsum([0, *lens[:-1]]).astype(np.int64))
        f.create_dataset("ep_len", data=np.asarray(lens, np.int64))
        for k, v in meta.items():
            f.create_dataset(f"ep_{k}", data=np.asarray(v))
        f.create_dataset("ep_file", data=np.asarray([e["file"] for e in keep], dtype="S"))
        f.attrs["scenario"] = SCENARIO
        f.attrs["source"] = str(raw)
    tmp.replace(out)


# --------------------------------------------------------------------------- #
#  plan: goal-conditioned MPC (CEM in latent space) in the simulator
# --------------------------------------------------------------------------- #
def cem(model, ctx, past, z_goal, init_mean, lo, hi, args, gen):
    """Best action blocks (Hp, block_dim) for reaching z_goal from the context.

    ctx (h, P, D) encoded frames, past (h - 1, block_dim) executed blocks between them,
    init_mean (Hp, block_dim); everything normalised. Returns (mean, cost of the mean, its
    predicted latents (Hp, P, D)).
    """
    import torch
    from src.models import latent_sq_dist

    n, hp = args.cem_samples, init_mean.size(0)
    mean, std = init_mean.clone(), torch.full_like(init_mean, args.cem_std)
    ctx_b = ctx[None].expand(n, *ctx.shape)
    past_b = past[None].expand(n, *past.shape)
    for _ in range(args.cem_iters):
        cand = mean + std * torch.randn(n, *mean.shape, device=mean.device, generator=gen)
        cand[0] = mean
        cand = torch.maximum(torch.minimum(cand, hi), lo)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mean.device.type == "cuda"):
            z = model.rollout(ctx_b, torch.cat([past_b, cand], 1)).float()  # (n, Hp, P, D)
        d = latent_sq_dist(z, z_goal.expand_as(z))  # (n, Hp)
        cost = d[:, -1] if args.plan_cost == "last" else d.mean(1)
        elite = cand[cost.topk(args.cem_topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mean.device.type == "cuda"):
        z = model.rollout(ctx[None], torch.cat([past, mean], 0)[None]).float()[0]
    d = latent_sq_dist(z, z_goal.expand_as(z))
    return mean, float(d[-1] if args.plan_cost == "last" else d.mean()), z


def label(img: np.ndarray, lines: list[str], color=(255, 255, 255)) -> np.ndarray:
    import cv2

    img = np.ascontiguousarray(img)
    for k, text in enumerate(lines):
        cv2.putText(img, text, (4, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (4, 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def to_env_action(a: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    """Recorded action (offset xyz, grip) -> ee_abs command."""
    return np.concatenate([tcp + a[:3], QUAT, [np.clip(a[3], -1.0, 1.0)]])


def replay(make_env, seed: int, actions: np.ndarray) -> bool:
    """The recorded actions, open loop: does the action interface reproduce the expert?"""
    env = make_env("state")
    obs, info = env.reset(seed=seed)
    for a in actions:
        obs, _, term, trunc, info = env.step(to_env_action(a, obs["state"][:3]))
        if term or trunc:
            break
    env.close()
    return bool(info["success"])


def run_episode(model, decoder, make_env, transform, normalizer, ep_meta, first_img, goal_img, lo, hi, cfg, args,
                device, out: Path) -> dict:
    import torch
    from common import encode_frames, resize_frames, to_uint8
    from src.models import latent_sq_dist
    from src.utils import save_json, save_video

    fs, hist, a_dim = cfg.data.frameskip, cfg.data.history, normalizer.dim("action")
    i, seed, goal = ep_meta["index"], ep_meta["seed"], ep_meta["goal"]
    env = make_env("rgb_state")
    obs, info = env.reset(seed=seed)
    # The env must be the one the data came from: same seed, same appearance randomisation.
    frame_err = np.abs(obs["image"].astype(np.int16) - first_img).mean()
    if not np.allclose(info["goal"], goal, atol=1e-4) or frame_err > 1.0:
        raise RuntimeError(f"episode {i}: seed {seed} does not reproduce the recording (goal {info['goal']} vs "
                           f"{goal}, first frame off by {frame_err:.1f}/255); check --randomize-lighting")
    max_steps = args.max_steps or env.max_steps

    def enc(img):
        return encode_frames(model, img[None], transform, device)[0]  # (P, D)

    z_goal = enc(goal_img)
    emb, blocks = [enc(obs["image"])], []
    mean = torch.zeros(args.plan_horizon, fs * a_dim, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed * 100_003 + i)
    res = 112 if decoder is None else decoder.res
    goal_small = to_uint8(resize_frames(goal_img[None], res, device))[0]
    imagined = np.zeros_like(goal_small)

    log = {k: [] for k in ("dist", "latent_dist", "plan_cost", "grip")}
    frames, step, reach, t0 = [], 0, None, time.time()
    done = False

    def record(o, inf):
        d = float(inf["dist_to_goal"])
        log["dist"].append(d)
        log["grip"].append(float(o["state"][7]))
        cam = to_uint8(resize_frames(o["image"][None], res, device))[0]
        ok = d < args.success_tol
        lines = [f"t={step} d={100 * d:.1f}cm", "AT GOAL" if ok else ("grasped" if inf["grasp_step"] is not None
                                                                     else "")]
        frames.append(np.concatenate([label(cam, lines, (120, 255, 120) if ok else (255, 255, 255)),
                                      label(goal_small, ["goal"]),
                                      label(imagined, ["plan end"])], 1))

    record(obs, info)
    while step < max_steps and not done:
        ctx = torch.stack(emb[-hist:])
        past = torch.stack(blocks[len(blocks) - (len(ctx) - 1):]) if len(ctx) > 1 else \
            torch.zeros(0, fs * a_dim, device=device)
        mean, cost, z = cem(model, ctx, past, z_goal, mean, lo, hi, args, gen)
        log["plan_cost"].append(cost)
        log["latent_dist"].append(float(latent_sq_dist(emb[-1], z_goal)))
        if decoder is not None:
            imagined = to_uint8(decoder(z[-1:]))[0]
        for b in range(args.exec_blocks):
            block = mean[b]
            raw = normalizer.denormalize("action", block.view(fs, a_dim).cpu().numpy())
            for a in raw:
                obs, _, term, trunc, info = env.step(to_env_action(a, obs["state"][:3]))
                step += 1
                record(obs, info)
                if info["dist_to_goal"] < args.success_tol and reach is None:
                    reach = step
                if term or trunc or step >= max_steps:
                    done = True
                    break
            if done:
                break
            blocks.append(block)
            emb.append(enc(obs["image"]))
        mean = torch.cat([mean[args.exec_blocks:], torch.zeros(args.exec_blocks, *mean.shape[1:], device=device)])
    env.close()

    dist = np.array(log["dist"])
    result = {
        "index": i, "seed": seed, "file": ep_meta["file"], "goal": goal.tolist(),
        "expert_reached_step": ep_meta["reached_step"], "expert_grasp_step": ep_meta["grasp_step"],
        "steps": step, "failure": bool(info["failure"]), "events": dict(env.events),
        "env_success": bool(info["success"]), "env_reached_step": info["reached_step"],
        "grasp_step": info["grasp_step"], "reach_step": reach,
        "min_dist": float(dist.min()), "final_dist": float(dist[-1]),
        **{f"success_{int(100 * t)}cm": bool(dist.min() < t) for t in TOLERANCES},
        **{f"final_{int(100 * t)}cm": bool(dist[-1] < t) for t in TOLERANCES},
        "seconds": round(time.time() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / f"ep{i}.json", {**result, **log})
    save_video(out / f"ep{i}.mp4", np.stack(frames), fps=30)
    from PIL import Image

    cols = np.linspace(0, len(frames) - 1, 12).round().astype(int)  # repeats if the episode is short
    Image.fromarray(np.concatenate([np.concatenate([frames[c] for c in cols[:6]], 1),
                                    np.concatenate([frames[c] for c in cols[6:]], 1)], 0)).save(out / f"ep{i}.png")
    return result


def plan_stage(ckpt: Path, args) -> None:
    import torch

    sys.path.insert(0, str(ROOT / "ablations"))
    sys.path.insert(0, str(args.env_root))
    import dynamic_envs
    from common import Decoder, analysis_dir, load_world_model
    from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
    from src.utils import save_json, set_seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_world_model(ckpt, device)
    set_seed(cfg.seed)
    dec_path = analysis_dir(ckpt, "decoder") / "decoder.pt"
    decoder = Decoder.load(dec_path, device) if dec_path.exists() else None
    transform = ImageTransform(cfg.data.img_size)
    out = analysis_dir(ckpt, "planner")

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action",))
    _, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    f = reader.file
    action = f["action"][:]
    raw_lo, raw_hi = np.nanmin(action, 0), np.nanmax(action, 0)
    fs = cfg.data.frameskip
    lo = torch.as_tensor(np.tile(normalizer.normalize("action", raw_lo), fs), device=device)
    hi = torch.as_tensor(np.tile(normalizer.normalize("action", raw_hi), fs), device=device)

    def make_env(obs_mode):
        return dynamic_envs.make(SCENARIO, action_mode="ee_abs", obs_mode=obs_mode,
                                 render_size=cfg.data.img_size, randomize_lighting=args.randomize_lighting)

    episodes = np.sort(val_eps)[: args.plan_episodes] if args.plan_episodes else np.sort(val_eps)
    print(f"planner | {len(episodes)} held-out episodes | CEM {args.cem_samples} samples x {args.cem_iters} iters, "
          f"top {args.cem_topk}, horizon {args.plan_horizon} x {fs} steps, cost {args.plan_cost} "
          f"| success: cup within {100 * args.success_tol:g} cm of the goal")
    results = []
    for ep in episodes:
        ep = int(ep)
        off, n = int(reader.ep_offset[ep]), int(reader.ep_len[ep])
        meta = {"index": ep, "seed": int(f["ep_seed"][ep]), "goal": f["ep_goal"][ep],
                "reached_step": int(f["ep_reached_step"][ep]), "grasp_step": int(f["ep_grasp_step"][ep]),
                "file": f["ep_file"][ep].decode()}
        first_img = reader.span("pixels", off, off + 1)[0]
        goal_img = reader.span("pixels", off + n - 1, off + n)[0]
        r = run_episode(model, decoder, make_env, transform, normalizer, meta, first_img, goal_img, lo, hi, cfg, args,
                        device,
                        out / "episodes")
        r["replay_success"] = replay(make_env, meta["seed"], reader.span("action", off, off + n - 1))
        results.append(r)
        print(f"ep {ep:3d} (seed {meta['seed']:3d}) | min {100 * r['min_dist']:5.1f} cm, final "
              f"{100 * r['final_dist']:5.1f} cm | success {r['reach_step'] is not None} | env {r['env_success']} | grasp {r['grasp_step']} | replay {r['replay_success']} | {r['seconds']}s",
              flush=True)
    reader.close()

    def rate(key):
        return float(np.mean([r[key] for r in results]))

    summary = {
        "ckpt": str(ckpt), "episodes": len(results), "success_tol_m": args.success_tol,
        "success_rate": float(np.mean([r["reach_step"] is not None for r in results])),
        **{f"success_rate_{int(100 * t)}cm": rate(f"success_{int(100 * t)}cm") for t in TOLERANCES},
        **{f"final_rate_{int(100 * t)}cm": rate(f"final_{int(100 * t)}cm") for t in TOLERANCES},
        "env_success_rate": rate("env_success"),
        "grasp_rate": float(np.mean([r["grasp_step"] is not None for r in results])),
        "replay_success_rate": rate("replay_success"),
        "mean_min_dist_cm": 100 * rate("min_dist"), "mean_final_dist_cm": 100 * rate("final_dist"),
        "planner": {k: getattr(args, k) for k in ("plan_horizon", "cem_samples", "cem_iters", "cem_topk", "cem_std",
                                                  "plan_cost", "exec_blocks", "max_steps", "seed")},
        "per_episode": results,
    }
    save_json(out / "planner.json", summary)
    print(f"\nplanner on {len(results)} held-out episodes: success (cup within {100 * args.success_tol:g} cm) "
          f"{summary['success_rate']:.1%} | at 2/5/10 cm "
          + "/".join(f"{summary[f'success_rate_{int(100 * t)}cm']:.1%}" for t in TOLERANCES)
          + f" | env success {summary['env_success_rate']:.1%} | grasped {summary['grasp_rate']:.1%}"
          f" | expert replay {summary['replay_success_rate']:.1%}"
          f" | mean min / final distance {summary['mean_min_dist_cm']:.1f} / {summary['mean_final_dist_cm']:.1f} cm")
    print(f"saved {out / 'planner.json'} and {out / 'episodes'}/ep<i>.{{mp4,png,json}}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
SMOKE = {
    "train": ["data.max_episodes=40", "data.val_episodes=8", "optim.steps_per_epoch=10", "optim.val_steps=5",
              "optim.num_workers=2"],
    "probe": ["--row-stride", "10", "--workers", "2"],
    "decoder": ["--steps", "200", "--row-stride", "10"],
    "rollout": ["--horizon", "5", "--start-stride", "5"],
}


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "_exec":
        exec_script(sys.argv[2], sys.argv[3:])
        return

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="vit", choices=sorted(config.PRESETS))
    p.add_argument("--stop-epoch", type=int, default=6,
                   help="train epochs 0..N (LR schedule of a 100-epoch run); the analyses use epoch_N")
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--force", action="store_true", help="redo stages whose output already exists")
    p.add_argument("--smoke", action="store_true", help="tiny settings, written to checkpoints/_smoke/")
    p.add_argument("--raw", type=Path, default=RAW, help="collect_demos.py output of ConveyorCupGoal-v0")
    p.add_argument("--h5", type=Path, default=H5, help="where the training table is written")
    p.add_argument("--env-root", type=Path, default=ENV_ROOT, help="the dynamic-environments repo")
    p.add_argument("--run-name", default=None,
                   help="run folder checkpoints/conveyor-cup-goal/<run-name> (default: the preset)")
    p.add_argument("--rollout-horizon", type=int, default=10,
                   help="latent steps (x5 control steps); episodes are 12-27 latent frames up to the goal")
    g = p.add_argument_group("planner")
    g.add_argument("--plan-episodes", type=int, default=0, help="first N held-out episodes (0: all)")
    g.add_argument("--plan-horizon", type=int, default=5, help="latent steps (action blocks) per plan")
    g.add_argument("--plan-cost", choices=("last", "all"), default="last",
                   help="latent distance to the goal at the last predicted step, or averaged over the horizon")
    g.add_argument("--cem-samples", type=int, default=300)
    g.add_argument("--cem-iters", type=int, default=30)
    g.add_argument("--cem-topk", type=int, default=30)
    g.add_argument("--cem-std", type=float, default=1.0, help="initial sampling std (normalised actions)")
    g.add_argument("--exec-blocks", type=int, default=1, help="blocks executed before replanning")
    g.add_argument("--max-steps", type=int, default=0, help="control steps per episode (0: the env's 150)")
    g.add_argument("--success-tol", type=float, default=0.02, help="m, cup-to-goal distance that counts")
    g.add_argument("--seed", type=int, default=0, help="planner sampling seed")
    g.add_argument("--randomize-lighting", action="store_true",
                   help="the data was collected with randomised lighting (collect_demos.py --randomize-lighting)")
    args = p.parse_args()

    stop = 0 if args.smoke else args.stop_epoch
    run_name = args.run_name or args.preset
    run_dir = ROOT / "checkpoints" / ("_smoke" if args.smoke else "") / TASK / run_name
    ckpt = run_dir / f"epoch_{stop:03d}.pt"
    smoke = SMOKE if args.smoke else {}
    if args.smoke:
        args.plan_episodes = args.plan_episodes or 2
        args.cem_samples, args.cem_iters, args.cem_topk = 64, 3, 8
        args.max_steps = args.max_steps or 30
    print(f"{TASK} | preset {args.preset} | epochs 0..{stop} | {run_dir}")

    def run(script, *a):
        python(script, *a, h5=args.h5)

    if "data" in args.stages and not skip("data", args.h5, False):
        prepare(args.raw, args.h5)

    if "train" in args.stages and not skip("train", ckpt, args.force):
        overrides = [f"data.task={TASK}", f"out_dir={run_dir.parent}", f"exp_id={run_name}",
                     f"optim.stop_epoch={stop}", "wandb.enabled=false", *smoke.get("train", [])]
        done = sorted(run_dir.glob("epoch_*.pt"))
        resume = ["--resume", "--ckpt", done[-1]] if done and not args.force else []
        run("train.py", "--preset", args.preset, *resume, *overrides)

    if any(s in args.stages for s in STAGES[2:]) and not ckpt.exists():
        sys.exit(f"{ckpt} does not exist; run the train stage first")

    if "probe" in args.stages:
        if not skip("probe", run_dir / "probe" / "probe.pt", args.force):
            run("ablations/probe.py", "--ckpt", ckpt, "--workers", 4, *smoke.get("probe", []))
        if not skip("probe", run_dir / "probe_random" / "probe.pt", args.force):
            run("ablations/probe.py", "--ckpt", ckpt, "--random-init", "--workers", 4, *smoke.get("probe", []))

    if "decoder" in args.stages and not skip("decoder", run_dir / "decoder" / "decoder.pt", args.force):
        # a patch grid is 196 x 384 numbers a frame: every 10th frame keeps the cached latents in RAM
        patches = ["--row-stride", 10] if args.preset == "dino" else []
        run("ablations/decoder.py", "--ckpt", ckpt, *patches, *smoke.get("decoder", []))

    if "rollout" in args.stages and not skip("rollout", run_dir / "rollout" / "rollout.json", args.force):
        # a decoded video for every held-out episode
        run("ablations/rollout.py", "--ckpt", ckpt, "--horizon", args.rollout_horizon, "--videos", 10_000,
            *smoke.get("rollout", []))

    if "plan" in args.stages and not skip("plan", run_dir / "planner" / "planner.json", args.force):
        register(args.h5)
        plan_stage(ckpt, args)

    print(f"\ndone: {run_dir}")


if __name__ == "__main__":
    main()
