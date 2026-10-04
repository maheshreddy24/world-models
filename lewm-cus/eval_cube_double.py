"""Closed-loop planning evaluation on OGBench cube-double.

    python eval_cube_double.py --ckpt <run>/best.pt                  # MPC with a trained model
    python eval_cube_double.py --ckpt <run>/best.pt eval.num_eval=200
    python eval_cube_double.py --policy random                       # the floor
    python eval_cube_double.py --policy zero                         # hold still

Each task replays a recorded start state from the held-out val recording and
asks the planner to reach the frame the recording reaches `eval.goal_offset`
steps later, given only that goal *image*.  Success is OGBench's criterion:
both cubes within `eval.cube_tol` (4 cm) of where they are in the goal, arm
ignored.  Both the start and the goal have every cube at rest (on the table or
stacked), and some cube moves at least `eval.min_goal_distance` between them, so
no task is solved by dropping a held cube or by standing still.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from eval_scene import build_planner
from src.data import get_normalizer, open_store
from src.data.scene_store import SceneStore
from src.envs import CubeDoubleVecEnv
from src.envs.cube_double import cube_positions
from src.planner import RandomPlanner, ZeroPlanner
from src.utils import load_model, make_panel, save_json, save_video, set_seed


# Heights a cube rests at: on the table, or stacked on the other cube. A cube
# anywhere else is in the gripper, so a start holding one is solved by letting go
# and a goal holding one is a snapshot of transport, not a place to put it.
REST_Z = (0.02, 0.06)
REST_TOL = 0.005


def resting(cubes: np.ndarray) -> np.ndarray:
    """(..., num_cubes, 3) -> (...,): every cube on the table or on the other one."""
    z = cubes[..., 2:3]
    return (np.abs(z - np.array(REST_Z)) < REST_TOL).any(axis=-1).all(axis=-1)


def sample_tasks(store: SceneStore, n: int, offset: int, budget: int, min_dist: float, qpos_adr, seed: int) -> dict:
    """`n` start/goal row pairs from the store, each moving some cube >= `min_dist`
    between two states where every cube is at rest.

    The start must leave room for the goal and for `budget` steps of expert
    reference after it, all inside one episode.
    """
    qpos = store.column("qpos")
    cubes = cube_positions(np.asarray(qpos), qpos_adr)  # (rows, num_cubes, 3), small
    rng = np.random.default_rng(seed)
    span = max(offset, budget)
    valid = store.ep_len > span
    if not valid.any():
        raise SystemExit(f"no episode is longer than {span} steps")

    starts: list[int] = []
    for _ in range(100):
        ep = rng.choice(np.flatnonzero(valid), 20 * n)
        t = rng.integers(0, store.ep_len[ep] - span)
        rows = store.ep_offset[ep] + t
        moved = np.linalg.norm(cubes[rows + offset] - cubes[rows], axis=-1).max(axis=1)
        ok = (moved >= min_dist) & resting(cubes[rows]) & resting(cubes[rows + offset])
        starts.extend(int(r) for r in rows[ok] if r not in starts)
        if len(starts) >= n:
            break
    if len(starts) < n:
        raise SystemExit(f"found only {len(starts)} tasks moving a cube >= {min_dist} m; lower eval.min_goal_distance")

    start = np.array(sorted(starts[:n]))
    goal = start + offset
    return {
        "start": start,
        "qpos": np.asarray(qpos[start]),
        "qvel": store.rows("qvel", start),
        "start_obs": store.rows("pixels", start),
        "goal_qpos": np.asarray(qpos[goal]),
        "goal_obs": store.rows("pixels", goal),
        "cube_moves": np.linalg.norm(cubes[goal] - cubes[start], axis=-1),  # (n, num_cubes)
    }


def run_batch(env, planner, tasks: dict, idx: np.ndarray, n: int, budget: int) -> dict:
    """Run one batch; `idx` fills the whole pool, only the first `n` are real."""
    obs = env.reset_to(tasks["qpos"][idx], tasks["qvel"][idx])
    env.set_goals(tasks["goal_qpos"][idx])
    env.freeze(np.arange(env.num_envs) >= n)
    planner.reset(env.num_envs)
    planner.set_goal(tasks["goal_obs"][idx])

    render_gap = np.abs(obs["pixels"][:n].astype(np.int16) - tasks["start_obs"][idx[:n]].astype(np.int16))
    frames = [obs["pixels"][:n].copy()]
    steps_to_success = np.full(n, -1, dtype=np.int64)
    for step in range(budget):
        actions = planner.act(obs, active=~env.done)
        obs = env.step(actions)
        frames.append(obs["pixels"][:n].copy())
        newly = (steps_to_success < 0) & obs["success"][:n]
        steps_to_success[newly] = step + 1
        if env.done.all():
            break

    dist = env.cube_distances()[:n]
    return {
        "success": env.success[:n].copy(),
        "steps_to_success": steps_to_success,
        "cube_distance": dist,  # (n, num_cubes) at the end
        "frames": np.stack(frames, axis=1),
        "render_gap_sum": float(render_gap.mean(axis=(1, 2, 3)).sum()),
        "steps_run": step + 1,
    }


def main(argv=None) -> None:
    cfg, args = get_config(argv, default_preset="cube_double_pixels", ckpt=None, policy="model", out=None)
    set_seed(cfg.seed)
    if args.policy not in ("model", "random", "zero"):
        raise SystemExit(f"--policy must be model, random or zero, got {args.policy!r}")
    baseline = args.policy in ("random", "zero")
    if not baseline and not args.ckpt:
        raise SystemExit("pass --ckpt <path> (or --policy random / zero for a baseline)")
    if cfg.plan.planner == "diffusion":
        raise SystemExit("cube-double has no diffusion policy yet; use the default MPC planner")

    # -- load first: the checkpoint decides what an observation is ---------- #
    model = None
    if not baseline:
        model, train_cfg = load_model(args.ckpt, device=cfg.device)
        cfg.data, cfg.model = train_cfg.data, train_cfg.model
        cfg.validate()

    # Action statistics come from the training store: the solver plans in the
    # normalised space the world model was trained in.
    train_store = open_store(cfg)
    normalizer = get_normalizer(cfg, train_store, (cfg.data.obs_key, "action"))
    train_store.close()

    ev = cfg.eval
    val_store = SceneStore(cfg.scene.val_store_dir)
    render_size = int(val_store.meta["columns"]["pixels"]["shape"][1])  # the recording's, not data.img_size
    pool = min(ev.num_envs, ev.num_eval)
    env = CubeDoubleVecEnv(
        num_envs=pool,
        env_id=cfg.scene.env_id,
        img_size=render_size,
        max_episode_steps=2 * ev.eval_budget,
        cube_tol=ev.cube_tol,
        seed=cfg.seed,
    )
    tasks = sample_tasks(val_store, ev.num_eval, ev.goal_offset, ev.eval_budget, ev.min_goal_distance,
                         env.qpos_adr, cfg.seed)
    n = len(tasks["start"])

    if args.out:
        out_dir = Path(args.out) / "eval_cube_double"
    elif args.ckpt:
        out_dir = Path(args.ckpt).parent / "eval_cube_double" / Path(args.ckpt).stem
    else:
        out_dir = Path(f"eval_{args.policy}") / "eval_cube_double"

    try:
        if model is None:
            cls = ZeroPlanner if args.policy == "zero" else RandomPlanner
            planner = cls(env.bounds, env.action_dim, seed=cfg.seed)
        else:
            planner = build_planner(cfg, model, normalizer, env.bounds, env.action_dim)

        success = np.zeros(n, dtype=bool)
        steps_to_success = np.full(n, -1, dtype=np.int64)
        cube_distance = np.zeros_like(tasks["cube_moves"])
        render_gap, env_steps = 0.0, 0
        started = time.time()
        batches = [np.arange(i, min(i + pool, n)) for i in range(0, n, pool)]
        for b, positions in enumerate(tqdm(batches, desc=f"batches of {pool}", dynamic_ncols=True,
                                           disable=not sys.stderr.isatty())):
            real = len(positions)
            idx = np.r_[positions, np.full(pool - real, positions[-1])]
            out = run_batch(env, planner, tasks, idx, real, ev.eval_budget)
            success[positions] = out["success"]
            steps_to_success[positions] = out["steps_to_success"]
            cube_distance[positions] = out["cube_distance"]
            render_gap += out["render_gap_sum"]
            env_steps += out["steps_run"] * real
            print(f"batch {b + 1}/{len(batches)} | {int(out['success'].sum())}/{real} solved"
                  f" | running {success[: positions[-1] + 1].mean() * 100:.1f}%", flush=True)

            # Written per batch, so memory holds one batch of frames at a time.
            if ev.save_video:
                for j, i in enumerate(positions):
                    if i >= ev.num_videos:
                        break
                    T = out["frames"].shape[1]
                    s = tasks["start"][i]
                    reference = val_store.span("pixels", s, s + T)
                    goal = np.repeat(tasks["goal_obs"][i][None], T, axis=0)
                    tag = "solved" if success[i] else "failed"
                    save_video(out_dir / f"episode_{i:03d}_{tag}.mp4",
                               make_panel(out["frames"][j], reference, goal), fps=ev.video_fps)
        elapsed = time.time() - started
        render_gap /= n

        solved = steps_to_success[steps_to_success > 0]
        # Partial credit: of the cubes the task actually moves, how many end in place.
        # A cube the goal leaves where it was would score for free, so it is left out.
        needed = tasks["cube_moves"] > ev.cube_tol
        placed = (cube_distance <= ev.cube_tol) & needed
        results = {
            "policy": args.policy if baseline else str(args.ckpt),
            "planner": args.policy if baseline else cfg.plan.planner,
            "solver": None if baseline else cfg.plan.solver,
            "num_eval": n,
            "success_rate": float(success.mean() * 100),
            "moved_cubes_placed_rate": float(placed.sum() / max(needed.sum(), 1) * 100),
            "two_cube_tasks": int((needed.sum(axis=1) == 2).sum()),
            "mean_final_worst_cube_distance": float(cube_distance.max(axis=1).mean()),
            "mean_steps_to_success": float(solved.mean()) if len(solved) else None,
            "eval_budget": ev.eval_budget,
            "goal_offset": ev.goal_offset,
            "min_goal_distance": ev.min_goal_distance,
            "cube_tol": ev.cube_tol,
            "mean_goal_distance": float(tasks["cube_moves"].max(axis=1).mean()),
            "start_frame_mean_abs_diff": round(render_gap, 3),
            "wall_time_s": round(elapsed, 1),
            "seconds_per_env_step": round(elapsed / max(env_steps, 1), 4),
            "start_rows": tasks["start"].tolist(),
            "success_per_task": success.tolist(),
            "steps_to_success": steps_to_success.tolist(),
            "goal_distance": tasks["cube_moves"].max(axis=1).round(4).tolist(),
        }
        print(f"\nsuccess {results['success_rate']:.1f}%  ({int(success.sum())}/{n})"
              f"  |  moved cubes placed {results['moved_cubes_placed_rate']:.1f}%"
              f"  |  {results['wall_time_s']}s over {len(batches)} batches of {pool}")
        if render_gap > 8:
            print(f"\nwarning | rendered start frames differ from the recorded ones by "
                  f"{render_gap:.1f}/255 on average; check scene.env_id and the recording")
        save_json(out_dir / "results.json", results)
        if ev.save_video:
            print(f"videos  | {out_dir}  (agent | expert | goal)")
        print(f"results | {out_dir / 'results.json'}")
    finally:
        env.close()
        val_store.close()


if __name__ == "__main__":
    main()
