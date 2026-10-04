"""Closed-loop evaluation on the seven OGBench Scene v1 tasks.

    python eval_scene.py --diffusion-ckpt <policy run>/epoch_019.pt plan.planner=diffusion
    python eval_scene.py --ckpt <world model>/epoch_007.pt            # MPC baseline
    python eval_scene.py --policy random                              # the floor
    python eval_scene.py --policy zero                                # hold still
    python eval_scene.py --diffusion-ckpt <...> scene.num_eval=70     # a quick pass

Every task is one row of `v1_pairs.npz`: a start state to reset the simulator
to and a goal state a few dozen steps later, in which exactly one thing about
the scene has changed.  The planner is given the goal *image* and nothing else,
and succeeds when the final scene matches the goal state — cube pose, drawer,
window and buttons, with the arm ignored (`scene_tasks.is_success`).

`v1_pairs.npz` is self-contained: it carries the simulator state and both frames,
so evaluation never reads a recording.  Its `start_idx`/`goal_idx` columns are
provenance from whichever recording it was mined from and are not used here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from src.data import STATE, ImageTransform, SceneState, get_normalizer, load_pairs, open_store
from src.envs import SceneVecEnv
from src.planner import DiffusionPlanner, MPCPlanner, RandomPlanner, ZeroPlanner, build_solver
from src.utils import load_model, load_policy, make_panel, save_json, save_video, set_seed


def select_tasks(pairs: dict, num_eval: int | None, seed: int) -> np.ndarray:
    """Row indices to evaluate, spread evenly over the seven tasks.

    Taking the first `num_eval` rows would evaluate two tasks and call it a
    score, since the file is stored task by task.
    """
    order = np.arange(len(pairs["task_id"]))
    if num_eval is None or num_eval >= len(order):
        return order

    rng = np.random.default_rng(seed)
    by_task = [rng.permutation(order[pairs["task_id"] == k]) for k in range(len(pairs["task_names"]))]
    picked: list[int] = []
    for round_ in range(max(len(t) for t in by_task)):
        for rows in by_task:
            if round_ < len(rows) and len(picked) < num_eval:
                picked.append(int(rows[round_]))
    return np.array(sorted(picked))


def build_planner(cfg, model, normalizer, bounds, action_dim, policy=None):
    """The diffusion policy, or MPC around the world model in normalised action space."""
    if cfg.plan.planner == "diffusion":
        return DiffusionPlanner(
            policy=policy, model=model, cfg=cfg, normalizer=normalizer,
            image_transform=ImageTransform(cfg.data.img_size), action_space_bounds=bounds,
        )
    low = normalizer.normalize("action", np.broadcast_to(bounds[0], (action_dim,)).astype(np.float32))
    high = normalizer.normalize("action", np.broadcast_to(bounds[1], (action_dim,)).astype(np.float32))
    solver = build_solver(
        cfg,
        action_dim=action_dim * cfg.plan.action_block,
        bounds=(np.tile(low, cfg.plan.action_block), np.tile(high, cfg.plan.action_block)),
        device=cfg.device,
    )
    return MPCPlanner(
        model=model, solver=solver, cfg=cfg, normalizer=normalizer,
        image_transform=ImageTransform(cfg.data.img_size), action_space_bounds=bounds,
    )


def run_batch(env, planner, pairs: dict, tasks: np.ndarray, budget: int, goal_key: str = "goal_obs") -> dict:
    """Run one batch of tasks on the env pool; return per-task outcomes.

    `tasks` may be shorter than the pool. The spare envs are reset to a copy of
    the last task, so every array stays rectangular, and frozen before the first
    step, so they are never stepped and never counted.
    """
    n, pool = len(tasks), env.num_envs
    idx = np.r_[tasks, np.full(pool - n, tasks[-1])]

    obs = env.reset_to(pairs["start_qpos"][idx], pairs["start_qvel"][idx], pairs["start_btn"][idx])
    env.set_goals(pairs["goal_qpos"][idx], pairs["goal_btn"][idx])
    env.freeze(np.arange(pool) >= n)
    planner.reset(pool)
    planner.set_goal(pairs[goal_key][idx])

    # The recorded start frame and the one this renderer produces for the same
    # state should agree; a large gap means the encoder is being shown something
    # the recording never contained.
    render_gap = np.abs(obs["pixels"][:n].astype(np.int16) - pairs["start_obs"][tasks].astype(np.int16))

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

    return {
        "success": env.success[:n].copy(),
        "steps_to_success": steps_to_success,
        "frames": np.stack(frames, axis=1),  # (n, T, H, W, 3)
        "render_gap_sum": float(render_gap.mean(axis=(1, 2, 3)).sum()),
        "steps_run": step + 1,
    }


def main(argv=None) -> None:
    cfg, args = get_config(
        argv, default_preset="scene_pixels", ckpt=None, policy="model", out=None, diffusion_ckpt=None
    )
    set_seed(cfg.seed)
    if args.policy not in ("model", "random", "zero"):
        raise SystemExit(f"--policy must be model, random or zero, got {args.policy!r}")
    random_policy = args.policy in ("random", "zero")  # a baseline: no checkpoint
    diffusion = not random_policy and cfg.plan.planner == "diffusion"
    if diffusion and not args.diffusion_ckpt:
        raise SystemExit("plan.planner=diffusion needs --diffusion-ckpt <path>")
    if not random_policy and not diffusion and not args.ckpt:
        raise SystemExit("pass --ckpt <path> (or --policy random / zero for a baseline)")

    # -- load first: the checkpoint decides what an observation is ---------- #
    policy = None
    if diffusion:
        policy, policy_cfg, lewm_ckpt = load_policy(args.diffusion_ckpt, device=cfg.device)
        if args.ckpt and Path(args.ckpt).resolve() != Path(lewm_ckpt).resolve():
            raise SystemExit(f"--diffusion-ckpt was trained on {lewm_ckpt}, not --ckpt {args.ckpt}")
        args.ckpt = lewm_ckpt
        cfg.policy = policy_cfg.policy

    model = None
    if not random_policy:
        model, train_cfg = load_model(args.ckpt, device=cfg.device)
        cfg.data, cfg.model = train_cfg.data, train_cfg.model
        cfg.validate()

    # -- tasks -------------------------------------------------------------- #
    pairs = load_pairs(cfg.scene.v1_pairs_path)
    rows = select_tasks(pairs, cfg.scene.num_eval, cfg.seed)
    n = len(rows)
    task_id = pairs["task_id"][rows]
    names = pairs["task_names"]

    # Action statistics come from the training recording: the solver plans in
    # the normalised space the world model was trained in.
    store = open_store(cfg)
    normalizer = get_normalizer(cfg, store, (cfg.data.obs_key, "action"))
    store.close()

    # The oracle model reads states, so its goals are states too. The pairs file
    # stores goal positions but not goal velocities: the goal is built with zero
    # velocity, which is how the policy's training goals were built as well.
    oracle = cfg.data.obs_key == STATE
    goal_key = "goal_obs"
    if oracle:
        state_fn = SceneState()
        pairs["goal_state"] = state_fn.batch(
            pairs["goal_qpos"], np.zeros_like(pairs["start_qvel"]), pairs["goal_btn"]
        )
        state_fn.close()
        goal_key = "goal_state"

    pool = min(cfg.scene.eval_batch, n)
    env = SceneVecEnv(
        num_envs=pool,
        env_id=cfg.scene.env_id,
        max_episode_steps=2 * cfg.scene.budget,
        cube_tol=cfg.scene.cube_tol,
        slide_tol=cfg.scene.slide_tol,
        with_state=oracle,
        seed=cfg.seed,
    )
    if model is None:
        baseline = ZeroPlanner if args.policy == "zero" else RandomPlanner
        planner = baseline(env.bounds, env.action_dim, seed=cfg.seed)
    else:
        planner = build_planner(cfg, model, normalizer, env.bounds, env.action_dim, policy)

    run = args.diffusion_ckpt if diffusion else args.ckpt
    # One folder per evaluated checkpoint: evaluating epoch_007 and epoch_015 of
    # the same run must not overwrite each other's results and videos.
    if args.out:
        out_dir = Path(args.out) / "eval_scene"
    elif run:
        out_dir = Path(run).parent / "eval_scene" / Path(run).stem
    else:
        out_dir = Path(f"eval_{args.policy}") / "eval_scene"

    # -- run, one batch of tasks at a time ---------------------------------- #
    batches = [np.arange(i, min(i + pool, n)) for i in range(0, n, pool)]
    success = np.zeros(n, dtype=bool)
    steps_to_success = np.full(n, -1, dtype=np.int64)
    render_gap, env_steps, steps_run = 0.0, 0, 0
    videos: set[tuple[int, bool]] = set()  # (task, solved) already written
    started = time.time()

    for b, positions in enumerate(tqdm(batches, desc=f"batches of {pool}", dynamic_ncols=True,
                                       disable=not sys.stderr.isatty())):
        out = run_batch(env, planner, pairs, rows[positions], cfg.scene.budget, goal_key)
        success[positions] = out["success"]
        steps_to_success[positions] = out["steps_to_success"]
        render_gap += out["render_gap_sum"]
        steps_run = max(steps_run, out["steps_run"])
        env_steps += out["steps_run"] * len(positions)
        print(f"batch {b + 1}/{len(batches)} | {int(out['success'].sum())}/{len(positions)} solved"
              f" | running {success[:positions[-1] + 1].mean() * 100:.1f}%", flush=True)

        # Videos are written as each batch finishes, so frames never pile up:
        # at most one solved and one failed example per task.
        if cfg.eval.save_video:
            for j, i in enumerate(positions):
                key = (int(task_id[i]), bool(out["success"][j]))
                if key in videos:
                    continue
                videos.add(key)
                goal = np.repeat(pairs["goal_obs"][rows[i]][None], out["frames"].shape[1], axis=0)
                tag = "solved" if key[1] else "failed"
                save_video(
                    out_dir / f"{names[key[0]]}_{i:03d}_{tag}.mp4",
                    make_panel(out["frames"][j], goal),
                    fps=cfg.eval.video_fps,
                )

    elapsed = time.time() - started
    render_gap /= n

    # -- report -------------------------------------------------------------- #
    per_task = {}
    for k, name in enumerate(names):
        mine = task_id == k
        if mine.any():
            per_task[name] = {
                "n": int(mine.sum()),
                "success_rate": float(success[mine].mean() * 100),
            }

    solved = steps_to_success[steps_to_success > 0]
    results = {
        "policy": args.policy if random_policy else str(args.ckpt),
        "planner": args.policy if random_policy else cfg.plan.planner,
        "solver": None if random_policy or diffusion else cfg.plan.solver,
        "diffusion_ckpt": str(args.diffusion_ckpt) if diffusion else None,
        "diffusion_samples": cfg.plan.diffusion_samples if diffusion else None,
        "chunk": cfg.policy.chunk if diffusion else None,
        "v1_pairs": str(cfg.scene.v1_pairs_path),
        "observation": cfg.data.obs_key,
        "num_eval": n,
        "success_rate": float(success.mean() * 100),
        "per_task": per_task,
        "mean_steps_to_success": float(solved.mean()) if len(solved) else None,
        "budget": cfg.scene.budget,
        "cube_tol": cfg.scene.cube_tol,
        "slide_tol": cfg.scene.slide_tol,
        "eval_batch": pool,
        "start_frame_mean_abs_diff": round(render_gap, 3),
        "steps_run": steps_run,
        "wall_time_s": round(elapsed, 1),
        "seconds_per_env_step": round(elapsed / max(env_steps, 1), 4),
    }

    print(f"\nsuccess {results['success_rate']:.1f}%  ({int(success.sum())}/{n})"
          f"  |  {results['wall_time_s']}s over {len(batches)} batches of {pool}")
    for name, row in per_task.items():
        print(f"  {name:18s} {row['success_rate']:5.1f}%  ({row['n']} tasks)")
    if render_gap > 8:
        print(f"\nwarning | rendered start frames differ from the recorded ones by "
              f"{render_gap:.1f}/255 on average; check scene.env_id and the recording")

    save_json(out_dir / "results.json", results)
    if cfg.eval.save_video:
        print(f"videos  | {out_dir}  (agent | goal), one solved + one failed per task where available")
    print(f"results | {out_dir / 'results.json'}")
    env.close()


if __name__ == "__main__":
    main()
