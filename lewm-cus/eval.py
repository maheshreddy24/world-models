"""Closed-loop planning evaluation on the OGBench cube task.

    python eval.py --ckpt <run>/best.pt                 # plan with a trained model
    python eval.py --ckpt <run>/best.pt plan.solver=mppi
    python eval.py --policy random                      # the baseline to beat
    python eval.py --policy zero                        # hold still: the "do nothing" floor
    python eval.py plan.planner=diffusion --diffusion-ckpt <policy run>/epoch_019.pt

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
from src.data.ogbench import CONTACT
from src.envs import CubeVecEnv
from src.planner import DiffusionPlanner, MPCPlanner, RandomPlanner, ZeroPlanner, build_solver
from src.utils import load_model, load_policy, make_panel, save_json, save_video, set_seed

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

def load_diffusion(cfg, args):
    """Load the policy and point `--ckpt` at the world model it was trained on.

    The policy only understands that model's latents, so the pairing comes from
    the policy checkpoint rather than the command line.
    """
    policy, policy_cfg, lewm_ckpt = load_policy(args.diffusion_ckpt, device=cfg.device)
    if args.ckpt and Path(args.ckpt).resolve() != Path(lewm_ckpt).resolve():
        raise SystemExit(f"--diffusion-ckpt was trained on {lewm_ckpt}, not --ckpt {args.ckpt}")
    args.ckpt = lewm_ckpt
    cfg.policy = policy_cfg.policy
    return policy


def build_planner(cfg, model, normalizer, bounds, action_dim, policy=None):
    """The diffusion policy, or MPC around the world model in normalised action space."""
    if cfg.plan.planner == "diffusion":
        return DiffusionPlanner(
            policy=policy,
            model=model,
            cfg=cfg,
            normalizer=normalizer,
            image_transform=ImageTransform(cfg.data.img_size),
            action_space_bounds=bounds,
        )
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
    cfg, args = get_config(argv, ckpt=None, policy="model", split="val", out=None, diffusion_ckpt=None)
    set_seed(cfg.seed)
    # "random" and "zero" are the baselines every planning number is judged against.
    baseline = args.policy in ("random", "zero")
    diffusion = not baseline and cfg.plan.planner == "diffusion"
    if diffusion and not args.diffusion_ckpt:
        raise SystemExit("plan.planner=diffusion needs --diffusion-ckpt <path>")
    if not baseline and not diffusion and not args.ckpt:
        raise SystemExit("pass --ckpt <path> (or --policy random|zero for a baseline)")

    # Load first: the checkpoint decides what an observation is.
    policy = load_diffusion(cfg, args) if diffusion else None
    model = None if baseline else load_trained(cfg, args)

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

    # MuJoCo holds one EGL context per env, ~120 MB of GPU memory each, so a few
    # hundred parallel envs do not fit next to a training job. The tasks run in
    # batches of `eval.num_envs` on the same envs; a short last batch is padded
    # with repeats of its final task, and those results are dropped.
    batch = min(cfg.eval.num_envs, n)
    num_batches = -(-n // batch)
    env = CubeVecEnv(
        num_envs=batch,
        env_id=cfg.eval.env_id,
        env_type=cfg.eval.env_type,
        img_size=cfg.data.img_size,
        max_episode_steps=2 * cfg.eval.eval_budget,
        terminate_at_goal=cfg.eval.terminate_at_goal,
        seed=cfg.seed,
        proprio=cfg.policy.proprio,
    )
    run = args.diffusion_ckpt if diffusion else args.ckpt
    out_dir = Path(args.out) if args.out else (Path(run).parent if run else Path(f"eval_{args.policy}"))
    out_dir = out_dir / "eval"

    # Always close the envs: a crash that skips this leaves MuJoCo's EGL contexts
    # to fail noisily at exit, burying the actual traceback under them.
    try:
        if model is None:
            cls = ZeroPlanner if args.policy == "zero" else RandomPlanner
            planner = cls(env.bounds, env.action_dim, seed=cfg.seed)
        else:
            planner = build_planner(cfg, model, normalizer, env.bounds, env.action_dim, policy)

        # -- run --------------------------------------------------------------- #
        success = np.zeros(n, dtype=bool)
        steps_to_success = np.full(n, -1, dtype=np.int64)
        grasp = {k: np.zeros(n, dtype=bool) for k in ("touched", "grasped", "lifted")}
        steps_run = 0
        started = time.time()
        for b, lo in enumerate(range(0, n, batch)):
            real = min(batch, n - lo)
            part = {k: v[np.minimum(np.arange(lo, lo + batch), n - 1)] for k, v in tasks.items()}
            frames, solved_at, steps, batch_grasp = run_batch(
                env, planner, part, cfg, diffusion, f"planning {b + 1}/{num_batches}"
            )
            success[lo : lo + real] = env.success[:real]
            steps_to_success[lo : lo + real] = solved_at[:real]
            for k, v in batch_grasp.items():
                grasp[k][lo : lo + real] = v[:real]
            steps_run += steps

            # Written per batch, so memory holds one batch of frames, not all of them.
            if cfg.eval.save_video:
                for j in range(min(real, cfg.eval.num_videos - lo)):
                    goal = np.repeat(part["goal_pixels"][j][None], frames.shape[1], axis=0)
                    panel = make_panel(frames[j], part["reference"][j], goal)
                    tag = "solved" if success[lo + j] else "failed"
                    save_video(out_dir / f"episode_{lo + j:02d}_{tag}.mp4", panel, fps=cfg.eval.video_fps)

        elapsed = time.time() - started

        # -- report ------------------------------------------------------------ #
        solved = steps_to_success[steps_to_success > 0]
        results = {
            "policy": args.policy if baseline else str(args.ckpt),
            "planner": args.policy if baseline else cfg.plan.planner,
            "solver": args.policy if baseline else (cfg.plan.solver if not diffusion else None),
            "diffusion_ckpt": str(args.diffusion_ckpt) if diffusion else None,
            "diffusion_samples": cfg.plan.diffusion_samples if diffusion else None,
            "split": args.split,
            "num_eval": n,
            "num_envs": batch,
            "success_rate": float(success.mean() * 100),
            # Did the planner ever engage the cube at all? A task solved without
            # touching it was solved by the cube's own momentum at the start.
            "touch_rate": float(grasp["touched"].mean() * 100),
            "grasp_rate": float(grasp["grasped"].mean() * 100),
            "lift_rate": float(grasp["lifted"].mean() * 100),
            "grasp_and_success_rate": float((grasp["grasped"] & success).mean() * 100),
            "mean_steps_to_success": float(solved.mean()) if len(solved) else None,
            "eval_budget": cfg.eval.eval_budget,
            "goal_offset": cfg.eval.goal_offset,
            "min_goal_distance": cfg.eval.min_goal_distance,
            "mean_goal_distance": float(tasks["goal_distance"].mean()),
            "steps_run": steps_run,
            "wall_time_s": round(elapsed, 1),
            "seconds_per_env_step": round(elapsed / steps_run, 3),
            "episodes": tasks["episode"].tolist(),
            # Per task, in the order of `episodes`, so two runs on the same tasks
            # can be compared pairwise instead of only on the headline rate.
            "success_per_task": success.tolist(),
            "grasped_per_task": grasp["grasped"].tolist(),
            "lifted_per_task": grasp["lifted"].tolist(),
            "steps_to_success": steps_to_success.tolist(),
            "goal_distance": tasks["goal_distance"].round(4).tolist(),
        }
        print(
            f"\nsuccess {results['success_rate']:.1f}%  ({int(success.sum())}/{n})"
            f"  |  {results['wall_time_s']}s for {steps_run} steps in {num_batches} batch(es) of {batch}"
        )
        print(
            f"grasp   touched {results['touch_rate']:.1f}%  |  grasped {results['grasp_rate']:.1f}%"
            f"  |  lifted {results['lift_rate']:.1f}%  |  grasped & solved {results['grasp_and_success_rate']:.1f}%"
        )
        save_json(out_dir / "results.json", results)
        if cfg.eval.save_video:
            print(f"videos  | {out_dir}  (agent | expert | goal)")
        print(f"results | {out_dir / 'results.json'}")
    finally:
        env.close()
        reader.close()


def run_batch(env, planner, tasks, cfg, diffusion, desc="planning"):
    """Run one batch of tasks (one per env) until they finish or the budget runs out.

    Returns:
        frames (num_envs, T, H, W, 3), steps_to_success (num_envs,) with -1 where
        unsolved, and the number of env steps run. Success is left in `env.success`.
    """
    obs = env.reset_to(tasks["qpos"], tasks["qvel"], tasks["prev_observation"])
    env.set_targets(tasks["privileged_block_0_pos"], tasks["privileged_block_0_quat"])
    planner.reset(env.num_envs)
    # The simulator's target doubles as the goal of a policy.goal=cube_xyz policy;
    # MPC gets the goal's oracle state, which a model with model.oracle_obs reads.
    goal = {"goal_pos": tasks["privileged_block_0_pos"]} if diffusion else {"goal_state": tasks["goal_state"]}
    planner.set_goal(tasks["goal_obs"], **goal)

    frames = [obs["pixels"].copy()]
    steps_to_success = np.full(env.num_envs, -1, dtype=np.int64)
    # Latched like success: touching or lifting the cube once counts.
    grasp = {k: np.zeros(env.num_envs, dtype=bool) for k in ("touched", "grasped", "lifted")}
    start_z = obs["cube_pos"][:, 2].copy()
    for step in tqdm(range(cfg.eval.eval_budget), desc=desc, dynamic_ncols=True,
                     disable=not sys.stderr.isatty()):
        actions = planner.act(obs, active=~env.done)
        obs = env.step(actions)
        frames.append(obs["pixels"].copy())
        newly_solved = (steps_to_success < 0) & obs["success"]
        steps_to_success[newly_solved] = step + 1

        touching = obs["observation"][:, CONTACT].ravel() > cfg.eval.grasp_contact
        near = np.linalg.norm(obs["cube_pos"] - obs["effector_pos"], axis=1) < cfg.eval.grasp_dist_cm / 100
        grasp["touched"] |= touching
        grasp["grasped"] |= touching & near
        grasp["lifted"] |= (obs["cube_pos"][:, 2] - start_z) > cfg.eval.lift_cm / 100
        if env.done.all():
            break
    return np.stack(frames, axis=1), steps_to_success, step + 1, grasp

if __name__ == "__main__":
    main()
