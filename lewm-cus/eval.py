"""Closed-loop planning evaluation on the OGBench cube task.

    python eval.py --ckpt <run>/best.pt                 # plan with a trained model
    python eval.py --ckpt <run>/best.pt plan.solver=mppi
    python eval.py --policy random                      # the baseline to beat

Each evaluation task replays a recorded start state and asks the planner to
reach the pose the expert reached `eval.goal_offset` steps later, given only the
goal *image*.  Success is the simulator's own check (cube within 4cm of the
target), so the number is directly comparable to the OGBench literature.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from src.data import H5Reader, ImageTransform, get_normalizer, sample_eval_episodes, split_episodes
from src.envs import CubeVecEnv
from src.planner import MPCPlanner, RandomPlanner, build_solver
from src.utils import load_model, make_panel, save_json, save_video, set_seed

def load_trained(cfg, args):
    """Load the model and adopt the data/model config it was trained with.

    This has to happen before anything reads `cfg.data`: the checkpoint decides
    what an observation *is* (pixels vs. oracle state) and at what resolution,
    while the CLI keeps ownership of the planning knobs.
    """
    model, train_cfg = load_model(args.ckpt, device=cfg.device)
    cfg.data = train_cfg.data
    cfg.model = train_cfg.model
    cfg.validate()
    return model

def build_planner(cfg, model, normalizer, bounds, action_dim):
    """MPC around a trained model, in normalised action space."""
    # Clip candidates to the env's action box, expressed in normalised space.
    low = normalizer.normalize("action", np.broadcast_to(bounds[0], (action_dim,)).astype(np.float32))
    high = normalizer.normalize("action", np.broadcast_to(bounds[1], (action_dim,)).astype(np.float32))
    solver = build_solver(
        cfg,
        action_dim=action_dim * cfg.plan.action_block,
        bounds=(np.tile(low, cfg.plan.action_block), np.tile(high, cfg.plan.action_block)),
        device=cfg.device,
    )
    return MPCPlanner(
        model=model,
        solver=solver,
        cfg=cfg,
        normalizer=normalizer,
        image_transform=ImageTransform(cfg.data.img_size),
        action_space_bounds=bounds,
    )

def main(argv=None) -> None:
    cfg, args = get_config(argv, ckpt=None, policy="model", split="val", out=None)
    set_seed(cfg.seed)
    if args.policy != "random" and not args.ckpt:
        raise SystemExit("pass --ckpt <path> (or --policy random for the baseline)")

    # Load first: the checkpoint decides what an observation is.
    model = None if args.policy == "random" else load_trained(cfg, args)

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action", "observation"))

    # Evaluate on episodes the model never trained on, by default.
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    pool = {"val": val_eps, "train": train_eps, "all": None}[args.split]

    n = cfg.eval.num_eval
    tasks = sample_eval_episodes(
        reader, n, cfg.eval.goal_offset, seed=cfg.seed, episodes=pool,
        obs_key=cfg.data.obs_key, min_goal_distance=cfg.eval.min_goal_distance,
    )

    env = CubeVecEnv(
        num_envs=n,
        env_id=cfg.eval.env_id,
        env_type=cfg.eval.env_type,
        img_size=cfg.data.img_size,
        max_episode_steps=2 * cfg.eval.eval_budget,
        terminate_at_goal=cfg.eval.terminate_at_goal,
        seed=cfg.seed,
    )
    planner = (
        RandomPlanner(env.bounds, env.action_dim, seed=cfg.seed)
        if model is None
        else build_planner(cfg, model, normalizer, env.bounds, env.action_dim)
    )

    # -- run --------------------------------------------------------------- #
    obs = env.reset_to(tasks["qpos"], tasks["qvel"])
    env.set_targets(tasks["privileged_block_0_pos"], tasks["privileged_block_0_quat"])
    planner.reset(n)
    planner.set_goal(tasks["goal_obs"])

    frames = [obs["pixels"].copy()]
    steps_to_success = np.full(n, -1, dtype=np.int64)
    started = time.time()

    for step in tqdm(range(cfg.eval.eval_budget), desc="planning", dynamic_ncols=True,
                     disable=not sys.stderr.isatty()):
        actions = planner.act(obs[cfg.data.obs_key], active=~env.done)
        obs = env.step(actions)
        frames.append(obs["pixels"].copy())
        newly_solved = (steps_to_success < 0) & obs["success"]
        steps_to_success[newly_solved] = step + 1
        if env.done.all():
            break

    elapsed = time.time() - started
    success = env.success.copy()

    # -- report ------------------------------------------------------------ #
    solved = steps_to_success[steps_to_success > 0]
    results = {
        "policy": args.policy if args.policy == "random" else str(args.ckpt),
        "solver": "random" if args.policy == "random" else cfg.plan.solver,
        "split": args.split,
        "num_eval": n,
        "success_rate": float(success.mean() * 100),
        "mean_steps_to_success": float(solved.mean()) if len(solved) else None,
        "eval_budget": cfg.eval.eval_budget,
        "goal_offset": cfg.eval.goal_offset,
        "min_goal_distance": cfg.eval.min_goal_distance,
        "mean_goal_distance": float(tasks["goal_distance"].mean()),
        "steps_run": step + 1,
        "wall_time_s": round(elapsed, 1),
        "seconds_per_env_step": round(elapsed / (step + 1), 3),
        "episodes": tasks["episode"].tolist(),
    }
    print(
        f"\nsuccess {results['success_rate']:.1f}%  ({int(success.sum())}/{n})"
        f"  |  {results['wall_time_s']}s for {step + 1} steps"
    )

    out_dir = Path(args.out) if args.out else (Path(args.ckpt).parent if args.ckpt else Path("eval_random"))
    out_dir = out_dir / "eval"
    save_json(out_dir / "results.json", results)

    # -- videos ------------------------------------------------------------ #
    if cfg.eval.save_video:
        agent = np.stack(frames, axis=1)  # (n, T, H, W, 3)
        for i in range(min(n, 8)):
            goal = np.repeat(tasks["goal_pixels"][i][None], agent.shape[1], axis=0)
            panel = make_panel(agent[i], tasks["reference"][i], goal)
            tag = "solved" if success[i] else "failed"
            save_video(out_dir / f"episode_{i:02d}_{tag}.mp4", panel, fps=cfg.eval.video_fps)
        print(f"videos  | {out_dir}  (agent | expert | goal)")

    print(f"results | {out_dir / 'results.json'}")
    env.close()
    reader.close()

if __name__ == "__main__":
    main()
