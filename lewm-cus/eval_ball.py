"""Closed-loop MPC evaluation on BallCatch (src/envs/ball.py).

    python eval_ball.py --preset ball_pixels --ckpt <run>/epoch_099.pt
    python eval_ball.py --preset ball_pixels --ckpt <run>/epoch_099.pt plan.solver=mppi
    python eval_ball.py --preset ball_pixels --policy random   # baselines to beat
    python eval_ball.py --preset ball_pixels --policy zero     # basket never moves
    python eval_ball.py --preset ball_pixels --policy replay   # recorded actions: checks the sim reproduces the data

Same protocol as eval.py on the cube, and the same planner: a task is a held-out
*expert* episode that caught the ball. The env is reset to its launch frame
(step 0) and the planner gets only the goal *image*: the recorded frame
`eval.goal_offset` steps after the catch, with the ball lying in the basket.
Tasks where the basket moves less than `eval.min_goal_distance` metres between
start and goal are dropped, since standing still would solve them.

Success is the simulator's own check: the ball's first landing is on the basket
floor. The planner is `MPCPlanner` with the solver from `plan.*` (CEM, 300
samples, 30 iterations, top 30, horizon 5, receding 5, blocks of 5 steps) —
the LeWM eval defaults.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_config
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
from src.envs import BallVecEnv
from src.envs.ball import MAX_SPEED
from src.planner import MPCPlanner, RandomPlanner, ZeroPlanner, build_solver
from src.utils import load_model, make_panel, save_json, save_video, set_seed

BASKET_X = 6  # column of sim_state


def sample_ball_tasks(reader: H5Reader, episodes: np.ndarray, num: int, goal_offset: int,
                      min_goal_distance: float, seed: int, budget: int) -> dict:
    """Held-out expert catches, turned into "start at launch, reach this frame" tasks.

    Only catches the expert made within `budget` steps count, so replaying its
    actions solves every task.
    """
    f = reader.file
    offset, length = reader.ep_offset, reader.ep_len
    catch, policy = f["ep_catch_step"][:], f["ep_policy"][:]

    starts, goals = [], []
    for e in episodes:
        c = int(catch[e])
        if policy[e] != 0 or c < 0 or c > budget or c + goal_offset > length[e] - 1:
            continue
        lo = int(offset[e])
        start_x, goal_x = f["sim_state"][lo, BASKET_X], f["sim_state"][lo + c + goal_offset, BASKET_X]
        if abs(goal_x - start_x) >= min_goal_distance:
            starts.append(lo)
            goals.append(lo + c + goal_offset)
    if len(starts) < num:
        raise SystemExit(f"only {len(starts)} eligible tasks for num_eval={num}; lower eval.num_eval or eval.min_goal_distance")

    pick = np.sort(np.random.default_rng(seed).choice(len(starts), num, replace=False))
    starts, goals = np.asarray(starts)[pick], np.asarray(goals)[pick]
    span = int((goals - starts).max()) + 1
    return {
        "episode": f["ep_idx"][:][starts],
        "start": starts,
        "goal": goals,
        "start_state": reader.rows("sim_state", starts),
        "goal_obs": reader.rows("pixels", goals),
        "goal_state": reader.rows("observation", goals),
        "goal_distance": np.abs(reader.rows("sim_state", goals)[:, BASKET_X] - reader.rows("sim_state", starts)[:, BASKET_X]),
        "catch_step": goals - starts - goal_offset,
        # The expert's own frames and actions from the start, for videos and `--policy replay`.
        "reference": np.stack([reader.span("pixels", s, s + span) for s in starts]),
        "recorded_actions": np.stack([reader.span("action", s, s + span) for s in starts]),
    }


def load_trained(cfg, args):
    """Load the model and adopt the data/model config it was trained with (as eval.py)."""
    model, train_cfg = load_model(args.ckpt, device=cfg.device)
    cfg.data = train_cfg.data
    cfg.model = train_cfg.model
    cfg.validate()
    return model


def build_planner(cfg, model, normalizer, bounds, action_dim):
    """MPC around the world model in normalised action space — eval.py's MPC branch."""
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


class ReplayPlanner(RandomPlanner):
    """Plays back the recorded expert actions — a check that resets reproduce the data."""

    def __init__(self, actions: np.ndarray):
        self.actions = np.nan_to_num(actions, nan=0.0)
        self.t = 0
        self.last_cost = None

    def reset(self, num_envs: int) -> None:
        self.num_envs, self.t = num_envs, 0

    def act(self, obs, active=None) -> np.ndarray:
        t = min(self.t, self.actions.shape[1] - 1)
        self.t += 1
        return self.actions[:, t].astype(np.float32)


def main(argv=None) -> None:
    cfg, args = get_config(argv, default_preset="ball_pixels", ckpt=None, policy="model", split="val", out=None)
    set_seed(cfg.seed)
    baseline = args.policy in ("random", "zero", "replay")
    if not baseline and not args.ckpt:
        raise SystemExit("pass --ckpt <path> (or --policy random|zero|replay)")
    if cfg.plan.planner != "mpc":
        raise SystemExit("eval_ball.py runs the MPC planner only (plan.planner=mpc)")

    # Load first: the checkpoint decides what an observation is.
    model = None if baseline else load_trained(cfg, args)

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action", "observation"))
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    pool = {"val": val_eps, "train": train_eps, "all": np.arange(reader.num_episodes)}[args.split]

    n = cfg.eval.num_eval
    tasks = sample_ball_tasks(reader, pool, n, cfg.eval.goal_offset, cfg.eval.min_goal_distance, cfg.seed,
                              cfg.eval.eval_budget)
    print(f"{n} tasks from {len(pool)} {args.split} episodes | mean basket travel {tasks['goal_distance'].mean():.2f} m"
          f" | expert caught at step {tasks['catch_step'].mean():.1f} on average")

    batch = min(cfg.eval.num_envs, n)
    num_batches = -(-n // batch)
    env = BallVecEnv(batch, img_size=cfg.data.img_size, max_episode_steps=cfg.eval.eval_budget,
                     terminate_at_goal=cfg.eval.terminate_at_goal)
    out_dir = Path(args.out) if args.out else (Path(args.ckpt).parent if args.ckpt else Path(f"eval_ball_{args.policy}"))
    out_dir = out_dir / "eval_ball"

    success = np.zeros(n, dtype=bool)
    landed = np.zeros(n, dtype=bool)
    steps_to_success = np.full(n, -1, dtype=np.int64)
    steps_run = 0
    started = time.time()
    for b, lo in enumerate(range(0, n, batch)):
        real = min(batch, n - lo)
        part = {k: v[np.minimum(np.arange(lo, lo + batch), n - 1)] for k, v in tasks.items()}
        if args.policy == "replay":
            planner = ReplayPlanner(part["recorded_actions"][..., 0:1])
        elif args.policy in ("random", "zero"):
            planner = (ZeroPlanner if args.policy == "zero" else RandomPlanner)(env.bounds, env.action_dim, seed=cfg.seed)
        else:
            planner = build_planner(cfg, model, normalizer, env.bounds, env.action_dim)

        frames, solved_at, steps = run_batch(env, planner, part, cfg, f"planning {b + 1}/{num_batches}")
        success[lo : lo + real] = env.success[:real]
        landed[lo : lo + real] = env.landed[:real]
        steps_to_success[lo : lo + real] = solved_at[:real]
        steps_run += steps

        if cfg.eval.save_video:
            for j in range(max(0, min(real, cfg.eval.num_videos - lo))):
                goal = np.repeat(part["goal_obs"][j][None], frames.shape[1], axis=0)
                panel = make_panel(frames[j], part["reference"][j], goal)
                panel = np.repeat(np.repeat(panel, 2, axis=1), 2, axis=2)
                tag = "caught" if success[lo + j] else "missed"
                save_video(out_dir / f"episode_{lo + j:03d}_{tag}.mp4", panel, fps=cfg.eval.video_fps)

    elapsed = time.time() - started
    solved = steps_to_success[steps_to_success > 0]
    results = {
        "policy": args.policy if baseline else str(args.ckpt),
        "solver": args.policy if baseline else cfg.plan.solver,
        "plan": None if baseline else {k: getattr(cfg.plan, k) for k in (
            "horizon", "receding_horizon", "action_block", "history_len", "num_samples", "n_iters", "topk",
            "var_scale", "momentum", "warm_start")},
        "split": args.split,
        "num_eval": n,
        "success_rate": float(success.mean() * 100),
        "landed_rate": float(landed.mean() * 100),
        "mean_steps_to_success": float(solved.mean()) if len(solved) else None,
        "expert_mean_catch_step": float(tasks["catch_step"].mean()),
        "eval_budget": cfg.eval.eval_budget,
        "goal_offset_after_catch": cfg.eval.goal_offset,
        "min_goal_distance": cfg.eval.min_goal_distance,
        "mean_goal_distance": float(tasks["goal_distance"].mean()),
        "max_basket_speed": MAX_SPEED,
        "steps_run": steps_run,
        "wall_time_s": round(elapsed, 1),
        "seconds_per_env_step": round(elapsed / max(steps_run, 1), 3),
        "episodes": tasks["episode"].tolist(),
        "success_per_task": success.tolist(),
        "steps_to_success": steps_to_success.tolist(),
        "goal_distance": tasks["goal_distance"].round(3).tolist(),
    }
    print(f"\nsuccess {results['success_rate']:.1f}%  ({int(success.sum())}/{n})  |  "
          f"{results['wall_time_s']}s for {steps_run} steps in {num_batches} batch(es) of {batch}")
    save_json(out_dir / "results.json", results)
    if cfg.eval.save_video:
        print(f"videos  | {out_dir}  (agent | expert | goal)")
    print(f"results | {out_dir / 'results.json'}")
    reader.close()


def run_batch(env, planner, tasks, cfg, desc="planning"):
    """Run one batch of tasks until every env finishes or the budget runs out."""
    obs = env.reset_to(tasks["start_state"])
    planner.reset(env.num_envs)
    planner.set_goal(tasks["goal_obs"], goal_state=tasks["goal_state"])

    frames = [obs["pixels"].copy()]
    steps_to_success = np.full(env.num_envs, -1, dtype=np.int64)
    step = 0
    for step in tqdm(range(cfg.eval.eval_budget), desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        obs = env.step(planner.act(obs, active=~env.done))
        frames.append(obs["pixels"].copy())
        steps_to_success[(steps_to_success < 0) & obs["success"]] = step + 1
        if env.done.all():
            break
    return np.stack(frames, axis=1), steps_to_success, step + 1


if __name__ == "__main__":
    main()
