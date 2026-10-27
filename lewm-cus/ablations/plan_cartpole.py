"""Goal-conditioned MPC on cartpole-swingup with a trained LeWM: can it swing the pole up?

    python ablations/plan_cartpole.py                                   # vit (SIGReg) epoch_006, every heldout episode
    python ablations/plan_cartpole.py --episodes 2 --max-steps 100      # quick check
    python ablations/plan_cartpole.py --ckpt checkpoints/cartpole-swingup/vit_ema/epoch_006.pt

Episodes: the heldout table (`<task>-valtest.h5`, MMBench val + test, 40 episodes never
trained on). Each one starts DMControl cartpole from the episode's first recorded state
(pole hanging down); rendering and dynamics match the recording (0.2/255 per pixel, replayed
returns identical).

Goal: one rendered frame of the target state, pole upright, cart at x = 0, at rest. Every
latent step (frameskip = 5 rows) CEM (conveyor_goal.cem) searches --plan-horizon action
blocks in the normalised action space: the predictor rolls out from the last `history`
encoded frames and the executed blocks between them, the cost is the latent distance to the
encoded goal (last predicted step, or --plan-cost all). The first block is executed, the rest
warm-starts the next search. One row = 2 physics steps (TD-MPC2's action repeat), as recorded.

Scored with the task's own reward (per row, summed over its 2 physics steps, so <= 2; an
episode is 500 rows) next to the recorded episode's return, plus how long the pole is upright
(|angle| < --upright-deg).

Outputs in <run dir>/planner/:

    planner.json             summary and per-episode results
    episodes/ep<i>.json      per row: pole angle, cart x, reward, action; per plan: cost, latent distance
    episodes/ep<i>.mp4       planner camera | the recorded episode at the same row | goal
    episodes/ep<i>.png       12 frames of the same
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ablations"))

from common import encode_frames  # noqa: E402
from config import heldout_h5, task_h5  # noqa: E402
from conveyor_goal import cem, label  # noqa: E402
from src.data import H5Reader, ImageTransform, build_datasets, get_normalizer  # noqa: E402
from src.models import latent_sq_dist  # noqa: E402
from src.utils import load_model, recalibrate_bn, save_json, save_video, set_seed  # noqa: E402

TASK = "cartpole-swingup"
SUBSTEPS = 2  # physics steps per recorded row


def load(ckpt: Path, device):
    """The world model with recalibrated BatchNorm, and the action normalizer, from the local training table.

    The checkpoint's config holds the h5 path of the machine it was trained on.
    """
    model, cfg = load_model(ckpt, device=str(device))
    cfg.data.h5_path = str(task_h5(cfg.data.task))
    train_set, _, reader, _ = build_datasets(cfg)
    recalibrate_bn(model, train_set, device)
    normalizer = get_normalizer(cfg, reader, ("action",))
    reader.close()
    return model, cfg, normalizer


class Cartpole:
    """DMControl cartpole-swingup, set to a recorded observation and stepped one row (2 physics steps) at a time."""

    def __init__(self, size: int):
        from dm_control import suite

        self.env = suite.load("cartpole", "swingup")
        self.physics, self.size = self.env.physics, size

    def reset(self, obs: np.ndarray) -> None:
        import mujoco

        m, d = self.physics.model.ptr, self.physics.data.ptr
        mujoco.mj_resetData(m, d)
        d.qpos[:] = [obs[0], math.atan2(obs[2], obs[1])]
        d.qvel[:] = obs[3:5]
        mujoco.mj_forward(m, d)

    def step(self, a: float) -> float:
        import mujoco

        m, d = self.physics.model.ptr, self.physics.data.ptr
        d.ctrl[:] = a
        reward = 0.0
        for _ in range(SUBSTEPS):
            mujoco.mj_step(m, d)
            reward += float(self.env.task.get_reward(self.physics))
        return reward

    @property
    def x(self) -> float:
        return float(self.physics.data.qpos[0])

    @property
    def angle(self) -> float:
        """Pole angle from upright, degrees in (-180, 180]."""
        a = float(self.physics.data.qpos[1])
        return math.degrees(math.atan2(math.sin(a), math.cos(a)))

    def render(self) -> np.ndarray:
        return self.physics.render(self.size, self.size, camera_id=0)


def run_episode(model, sim, transform, normalizer, rec, goal_img, lo, hi, cfg, args, device, i, out: Path) -> dict:
    fs, hist, a_dim = cfg.data.frameskip, cfg.data.history, normalizer.dim("action")
    obs, pixels, rec_reward = rec["observation"], rec["pixels"], rec["reward"]
    rows = min(args.max_steps or len(obs) - 1, len(obs) - 1)

    def enc(img):
        return encode_frames(model, img[None], transform, device)[0]  # (P, D)

    sim.reset(obs[0])
    z_goal = enc(goal_img)
    img = sim.render()
    emb, blocks = [enc(img)], []
    mean = torch.zeros(args.plan_horizon, fs * a_dim, device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed * 100_003 + i)
    goal_panel = label(goal_img, ["goal"])

    log = {k: [] for k in ("angle", "x", "reward", "action", "plan_cost", "latent_dist")}
    frames, step, t0 = [], 0, time.time()

    def record(img):
        ang = sim.angle
        log["angle"].append(ang)
        log["x"].append(sim.x)
        up = abs(ang) < args.upright_deg
        ret, rec_ret = sum(log["reward"]), float(np.nansum(rec_reward[: step]))
        frames.append(np.concatenate([
            label(img, [f"planner t={step}", f"angle {ang:+.0f} deg", f"return {ret:.0f}", "UPRIGHT" if up else ""],
                  (120, 255, 120) if up else (255, 255, 255)),
            label(pixels[step], ["recorded", f"return {rec_ret:.0f}"]),
            goal_panel], 1))

    record(img)
    while step < rows:
        ctx = torch.stack(emb[-hist:])
        past = torch.stack(blocks[len(blocks) - (len(ctx) - 1):]) if len(ctx) > 1 else \
            torch.zeros(0, fs * a_dim, device=device)
        mean, cost, _ = cem(model, ctx, past, z_goal, mean, lo, hi, args, gen)
        log["plan_cost"].append(cost)
        log["latent_dist"].append(float(latent_sq_dist(emb[-1], z_goal)))
        for b in range(args.exec_blocks):
            block = mean[b]
            raw = normalizer.denormalize("action", block.view(fs, a_dim).cpu().numpy())
            for a in raw:
                a = float(np.clip(a[0], -1.0, 1.0))
                log["reward"].append(sim.step(a))
                log["action"].append(a)
                step += 1
                img = sim.render()
                record(img)
                if step >= rows:
                    break
            if step >= rows:
                break
            blocks.append(block)
            emb.append(enc(img))
        mean = torch.cat([mean[args.exec_blocks:], torch.zeros(args.exec_blocks, *mean.shape[1:], device=device)])

    angle = np.abs(np.array(log["angle"][1:]))
    up = angle < args.upright_deg
    result = {
        "index": i, "rows": step, "return": float(sum(log["reward"])),
        "recorded_return": float(np.nansum(rec_reward[:step])),
        "upright_fraction": float(up.mean()), "first_upright_row": int(up.argmax()) + 1 if up.any() else None,
        "upright_last_100": float(up[-100:].mean()), "min_abs_angle": float(angle.min()),
        "final_abs_angle": float(angle[-1]), "final_x": log["x"][-1],
        "seconds": round(time.time() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / f"ep{i}.json", {**result, **log})
    save_video(out / f"ep{i}.mp4", np.stack(frames), fps=args.fps)
    from PIL import Image

    cols = np.unique(np.linspace(0, len(frames) - 1, 12).round().astype(int))
    Image.fromarray(np.concatenate([np.concatenate([frames[c] for c in cols[:6]], 1),
                                    np.concatenate([frames[c] for c in cols[6:]], 1)], 0)).save(out / f"ep{i}.png")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints" / TASK / "vit" / "epoch_006.pt")
    p.add_argument("--h5", type=Path, default=heldout_h5(TASK), help="episodes to plan from (default: heldout)")
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="episode indices (default: all)")
    p.add_argument("--max-steps", type=int, default=0, help="rows per episode (0: the recording's 500)")
    p.add_argument("--upright-deg", type=float, default=20.0, help="|pole angle| that counts as upright")
    p.add_argument("--plan-horizon", type=int, default=5, help="latent steps (action blocks) per plan")
    p.add_argument("--plan-cost", choices=("last", "all"), default="last")
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-iters", type=int, default=30)
    p.add_argument("--cem-topk", type=int, default=30)
    p.add_argument("--cem-std", type=float, default=1.0, help="initial sampling std (normalised actions)")
    p.add_argument("--exec-blocks", type=int, default=1, help="blocks executed before replanning")
    p.add_argument("--fps", type=int, default=50, help="video fps (one row = 0.02 s, so 50 is real time)")
    p.add_argument("--out", type=Path, default=None, help="default: <run dir>/planner")
    p.add_argument("--seed", type=int, default=0, help="planner sampling seed")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, normalizer = load(args.ckpt, device)
    set_seed(cfg.seed)
    transform = ImageTransform(cfg.data.img_size)
    out = args.out or args.ckpt.parent / "planner"
    fs = cfg.data.frameskip
    # the env's action range, [-1, 1]
    lo = torch.as_tensor(np.tile(normalizer.normalize("action", np.array([-1.0], np.float32)), fs), device=device)
    hi = torch.as_tensor(np.tile(normalizer.normalize("action", np.array([1.0], np.float32)), fs), device=device)

    sim = Cartpole(cfg.data.img_size)
    sim.reset(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))  # upright, centred, at rest
    goal_img = sim.render()

    reader = H5Reader(args.h5)
    episodes = args.episodes if args.episodes is not None else list(range(reader.num_episodes))
    print(f"planner | {args.ckpt} | {len(episodes)} episodes of {args.h5.name} | CEM {args.cem_samples} samples x "
          f"{args.cem_iters} iters, top {args.cem_topk}, horizon {args.plan_horizon} x {fs} rows, cost {args.plan_cost}",
          flush=True)
    results = []
    for ep in episodes:
        off, n = int(reader.ep_offset[ep]), int(reader.ep_len[ep])
        rec = {k: reader.span(k, off, off + n) for k in ("observation", "pixels", "reward")}
        r = run_episode(model, sim, transform, normalizer, rec, goal_img, lo, hi, cfg, args, device, ep,
                        out / "episodes")
        results.append(r)
        print(f"ep {ep:2d} | return {r['return']:6.1f} (recorded {r['recorded_return']:6.1f}) | upright "
              f"{r['upright_fraction']:.0%}, last 100 rows {r['upright_last_100']:.0%}, first at row "
              f"{r['first_upright_row']} | final angle {r['final_abs_angle']:5.1f} deg | {r['seconds']}s", flush=True)
    reader.close()

    def mean(key):
        return float(np.mean([r[key] for r in results]))

    summary = {
        "ckpt": str(args.ckpt), "h5": str(args.h5), "episodes": len(results), "upright_deg": args.upright_deg,
        "mean_return": mean("return"), "mean_recorded_return": mean("recorded_return"),
        "swung_up_rate": float(np.mean([r["first_upright_row"] is not None for r in results])),
        "mean_upright_fraction": mean("upright_fraction"), "mean_upright_last_100": mean("upright_last_100"),
        "planner": {k: getattr(args, k) for k in ("plan_horizon", "plan_cost", "cem_samples", "cem_iters", "cem_topk",
                                                  "cem_std", "exec_blocks", "max_steps", "seed")},
        "per_episode": results,
    }
    save_json(out / "planner.json", summary)
    print(f"\n{len(results)} episodes: return {summary['mean_return']:.1f} (recorded {summary['mean_recorded_return']:.1f})"
          f" | reached upright {summary['swung_up_rate']:.0%} | upright {summary['mean_upright_fraction']:.0%} of rows,"
          f" {summary['mean_upright_last_100']:.0%} of the last 100")
    print(f"saved {out / 'planner.json'} and {out / 'episodes'}/ep<i>.{{mp4,png,json}}")


if __name__ == "__main__":
    main()
