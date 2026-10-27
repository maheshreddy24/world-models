"""CEM planning success rate, with LeWM's evaluation protocol (le-wm/eval.py, config/eval/*.yaml).

    python compare/plan.py --ckpt checkpoints/compare/pusht/subjepa/epoch_009.pt --seeds 42 100 2026

stable-worldmodel runs it unchanged: `swm.World` with 50 parallel envs,
`swm.policy.WorldModelPolicy` + `swm.solver.CEMSolver` (300 samples, 30
iterations, top 30, var scale 1, one env per solve), planning 5 action blocks of
5 env steps and executing all 5 before replanning, warm-started. Start states:
50 random rows of the task's recording (seeded, the same sampling as eval.py),
the goal is the recorded frame 25 steps later, the budget 50 env steps;
success is the env's own check (TwoRoom: agent within 16 px of the goal;
PushT: positions within 20 and block angle within pi/9).

The only piece written here is the cost model: our LeWM encodes the current frame
and the goal frame, rolls the predictor forward on each candidate (normalised
action blocks, the training statistics), and scores the squared L2 between the
last predicted latent and the goal latent, as LeWM's JEPA.get_cost does.

Writes <ckpt dir>/cem/seed<s>.json and cem.json (mean and std over the requested seeds). A seed
whose seed<s>.json exists with the same protocol is not re-run, so adding seeds costs only the
new ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "ablations")]

from common import load_world_model  # noqa: E402
from src.data import H5Reader, get_normalizer  # noqa: E402
from src.utils import save_json  # noqa: E402

SEEDS = (42, 100, 2026, 3407, 1234, 4444)  # Sub-JEPA's evaluation seeds
DEFAULT_PROTOCOL = {"num_eval": 50, "goal_offset": 25, "eval_budget": 50, "horizon": 5, "receding_horizon": 5,
                    "samples": 300, "iters": 30, "topk": 30, "var_scale": 1.0}

ENVS = {
    "tworoom": {
        "env_name": "swm/TwoRoom-v1",
        "state_key": "proprio",
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    },
    "pusht": {
        "env_name": "swm/PushT-v1",
        "state_key": "state",
        "callables": [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    },
}


class ActionScaler:
    """The z-score of our training statistics, with sklearn's StandardScaler interface (what the policy calls)."""

    def __init__(self, normalizer):
        self.norm = normalizer

    def transform(self, x):
        return self.norm.normalize("action", np.asarray(x, dtype=np.float32))

    def inverse_transform(self, x):
        return self.norm.denormalize("action", np.asarray(x, dtype=np.float32))


class CostModel(nn.Module):
    """Our LeWM behind stable-worldmodel's Costable protocol: get_cost(info, candidates) -> (B, S)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def get_cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        dev = candidates.device
        b, s, h, a = candidates.shape
        pixels = info["pixels"][:, 0, -1:].to(dev)  # (B, 1, H, W, 3) uint8: the current frame
        goal = info["goal"][:, 0, -1:].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z0 = self.model.encode(pixels)  # (B, 1, P, D)
            zg = self.model.encode(goal)
            ctx = z0[:, None].expand(b, s, *z0.shape[1:]).reshape(b * s, *z0.shape[1:])
            pred = self.model.rollout(ctx, candidates.reshape(b * s, h, a))  # (B*S, h, P, D)
        last = pred[:, -1].float().view(b, s, *pred.shape[2:])
        return (last - zg[:, None, 0].float()).pow(2).sum((-1, -2))


def render_check(spec: dict, state: np.ndarray, goal_state: np.ndarray, recorded: np.ndarray) -> float:
    """Mean absolute pixel difference (0-255) between the env rendering a recorded state and the recording.

    A large value means the simulator (stable-worldmodel version) draws the scene differently from
    the one the data was recorded with, so the planner would see frames unlike the training ones.
    """
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers the swm/* envs)

    env = gym.make(spec["env_name"], render_mode="rgb_array")
    env.reset(seed=0)
    env.unwrapped._set_state(state)
    env.unwrapped._set_goal_state(goal_state)
    img = np.asarray(env.render())
    env.close()
    a = torch.as_tensor(img).permute(2, 0, 1)[None].float()
    b = torch.as_tensor(recorded).permute(2, 0, 1)[None].float()
    a = torch.nn.functional.interpolate(a, size=b.shape[-2:], mode="bilinear", antialias=True)
    return float((a - b).abs().mean())


def sample_starts(reader: H5Reader, num_eval: int, goal_offset: int, seed: int):
    """eval.py's sampling: `num_eval` distinct rows that leave room for the goal, sorted; -> (episodes, start steps)."""
    max_start = reader.ep_len - goal_offset - 1
    valid = np.concatenate([off + np.arange(m + 1) for off, m in zip(reader.ep_offset, max_start) if m >= 0])
    g = np.random.default_rng(seed)
    rows = np.sort(valid[g.choice(len(valid) - 1, size=num_eval, replace=False)])
    eps = np.searchsorted(reader.ep_offset, rows, side="right") - 1
    return eps, rows - reader.ep_offset[eps]


def evaluate(ckpt: Path, seeds, num_eval=50, goal_offset=25, eval_budget=50, horizon=5, receding=5,
             samples=300, iters=30, topk=30, var_scale=1.0, video=False, out: Path | None = None,
             redo: bool = False) -> dict:
    import stable_worldmodel as swm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load_world_model(ckpt, device)
    spec = ENVS[cfg.data.task]
    fs = cfg.data.frameskip
    out = out or ckpt.parent / "cem"
    name = ckpt.parent.name if ckpt.parent.name != "keep" else ckpt.parent.parent.name
    name = f"{name} {ckpt.stem}"
    protocol = {"num_eval": num_eval, "goal_offset": goal_offset, "eval_budget": eval_budget, "horizon": horizon,
                "receding_horizon": receding, "action_block": fs, "samples": samples, "iters": iters, "topk": topk,
                "var_scale": var_scale}

    def done(seed):
        path = out / f"seed{seed}.json"
        if redo or not path.exists():
            return None
        r = json.loads(path.read_text())
        # files written before the protocol was stored ran the default protocol
        return r if r.get("protocol", DEFAULT_PROTOCOL | {"action_block": fs}) == protocol else None
    # Everything we read ourselves is read up front: stable-worldmodel opens the file in SWMR
    # mode, and h5py refuses a second handle on it with a different mode in the same process.
    reader = H5Reader(cfg.data.h5_path)
    normalizer = get_normalizer(cfg, reader, ("action",))
    # render at the recording's resolution: the goal frames come from it, and so did the training frames
    image_shape = tuple(int(v) for v in reader.file["pixels"].shape[1:3])
    plans = {}
    for seed in seeds:
        eps, starts = sample_starts(reader, num_eval, goal_offset, seed)
        plans[seed] = (eps, starts)
    # env 0 of the first seed: its recorded start frame, state and goal state
    row = int(reader.ep_offset[plans[seeds[0]][0][0]] + plans[seeds[0]][1][0])
    key = spec["state_key"]
    check = (reader.span(key, row, row + 1)[0], reader.span(key, row + goal_offset, row + goal_offset + 1)[0],
             reader.span("pixels", row, row + 1)[0])
    reader.close()
    try:
        check_mae = render_check(spec, *check)
    except Exception as e:  # the check is a diagnostic; never let it stop the evaluation
        print(f"render check failed: {e!r}")
        check_mae = None
    print(f"{cfg.data.task}: env render vs recorded frame, mean abs diff {check_mae}/255", flush=True)
    if check_mae is not None and check_mae > 10:
        print(f"WARNING: the env draws the recorded state differently (mean abs diff {check_mae:.0f}/255); "
              "check the stable-worldmodel version the data was recorded with", flush=True)
    dataset = swm.data.HDF5Dataset(path=cfg.data.h5_path)

    per_seed = []
    for seed in seeds:
        if (r := done(seed)) is not None:
            per_seed.append(r)
            print(f"{cfg.data.task} | {name} | seed {seed}: success {r['success_rate']:.1f}% (already done)", flush=True)
            continue
        t0 = time.time()
        eps, starts = plans[seed]
        world = swm.World(spec["env_name"], num_envs=num_eval, image_shape=image_shape,
                          max_episode_steps=2 * eval_budget)
        cost = CostModel(model).to(device)
        solver = swm.solver.CEMSolver(model=cost, batch_size=1, num_samples=samples, var_scale=var_scale,
                                      n_steps=iters, topk=topk, device=str(device), seed=seed)
        plan = swm.PlanConfig(horizon=horizon, receding_horizon=receding, action_block=fs, warm_start=True)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=plan, process={"action": ActionScaler(normalizer)})
        world.set_policy(policy)
        metrics = world.evaluate(dataset=dataset, episodes_idx=eps.tolist(), start_steps=starts.tolist(),
                                 goal_offset=goal_offset, eval_budget=eval_budget, callables=spec["callables"],
                                 video=(out / f"videos_seed{seed}") if video else None)
        world.close()

        result = {"seed": seed, "success_rate": float(metrics["success_rate"]),
                  "episode_successes": np.asarray(metrics["episode_successes"]).astype(bool).tolist(),
                  "episodes": eps.tolist(), "start_steps": starts.tolist(),
                  "render_check_mae": check_mae, "seconds": round(time.time() - t0, 1), "protocol": protocol}
        save_json(out / f"seed{seed}.json", result)
        per_seed.append(result)
        print(f"{cfg.data.task} | {name} | seed {seed}: success {result['success_rate']:.1f}% "
              f"| {result['seconds']}s", flush=True)

    rates = np.array([r["success_rate"] for r in per_seed])
    summary = {"ckpt": str(ckpt), "task": cfg.data.task, "seeds": list(seeds), "success_rate_mean": float(rates.mean()),
               "success_rate_std": float(rates.std()), "success_rates": rates.tolist(), "render_check_mae": check_mae,
               "protocol": protocol}
    save_json(out / "cem.json", summary)
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--video", action="store_true")
    p.add_argument("--redo", action="store_true", help="re-run seeds that already have a seed<s>.json")
    p.add_argument("--out", type=Path, default=None, help="default: <ckpt dir>/cem")
    args = p.parse_args(argv)
    evaluate(args.ckpt, args.seeds, num_eval=args.num_eval, samples=args.samples, iters=args.iters,
             topk=args.topk, video=args.video, out=args.out, redo=args.redo)


if __name__ == "__main__":
    main()
