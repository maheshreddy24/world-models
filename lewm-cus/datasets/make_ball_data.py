"""Generate the BallCatch recording (src/envs/ball.py) as one HDF5 table.

    python datasets/make_ball_data.py                                  # 20k episodes + samples
    python datasets/make_ball_data.py --episodes 200 --out /tmp/b.h5   # quick look
    python datasets/make_ball_data.py --samples-only                   # redo samples from an existing file

The layout is the one `src.data.ogbench.H5Reader` reads, so training needs only
`data.h5_path` (see `--preset ball_pixels`):

    per row (N = episodes x (steps + 1))
        pixels       (N, 128, 128, 3) uint8   the whole scene, fixed camera
        action       (N, 1)  float32          basket velocity in [-1, 1]; NaN on each episode's last row
        observation  (N, 6)  float32          ball x, y, vx, vy, basket x, basket vx
        sim_state    (N, 12) float64          everything `BallCatchSim.set_state` needs to resume
        landed, caught (N,)  bool             latched: the ball has landed / landed in the basket
        ep_idx, step_idx (N,) int32
    per episode (E)
        ep_offset, ep_len                     row span of each episode
        ep_success, ep_catch_step, ep_land_step, ep_policy (0 expert, 1 random), ep_launch (h, speed, angle)

Half the episodes come from a noisy expert that chases the landing point, half
from smoothed random velocities; see `NoisyExpert` / `OUPolicy`.

After writing, a few episodes are read back from the file into `--samples`:
mp4s, a contact sheet of the frames the model sees (every `frameskip`-th), a
few training windows exactly as `SequenceDataset` returns them, and stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HOME
from src.envs import ball

DEFAULT_OUT = HOME / "datasets/ballcatch/ballcatch_v2_20k.h5"
DEFAULT_SAMPLES = Path(__file__).resolve().parent.parent / "ballcatch_samples"


def _collect(args):
    seed, steps, policy = args
    return ball.collect_episode(seed, steps, policy)


def generate(out: Path, episodes: int, steps: int, seed: int, workers: int) -> None:
    rows = steps + 1
    n = episodes * rows
    # Alternate expert / random so every prefix of the file is balanced.
    jobs = [(seed + e, steps, "expert" if e % 2 == 0 else "random") for e in range(episodes)]

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    started = time.time()
    with h5py.File(tmp, "w") as f:
        s = ball.IMG_SIZE
        # One chunk per episode: a training window never straddles two.
        pixels = f.create_dataset("pixels", (n, s, s, 3), np.uint8, chunks=(rows, s, s, 3), **blosc)
        cols = {
            "action": f.create_dataset("action", (n, ball.ACTION_DIM), np.float32),
            "observation": f.create_dataset("observation", (n, ball.OBS_DIM), np.float32),
            "sim_state": f.create_dataset("sim_state", (n, ball.STATE_DIM), np.float64),
            "landed": f.create_dataset("landed", (n,), bool),
            "caught": f.create_dataset("caught", (n,), bool),
        }
        f["ep_idx"] = np.repeat(np.arange(episodes, dtype=np.int32), rows)
        f["step_idx"] = np.tile(np.arange(rows, dtype=np.int32), episodes)
        f["ep_offset"] = np.arange(episodes, dtype=np.int64) * rows
        f["ep_len"] = np.full(episodes, rows, dtype=np.int64)
        f["ep_policy"] = np.array([0 if p == "expert" else 1 for _, _, p in jobs], np.int8)
        success = np.zeros(episodes, bool)
        catch_step = np.full(episodes, -1, np.int32)
        land_step = np.full(episodes, -1, np.int32)
        launch = np.zeros((episodes, 3), np.float32)

        with Pool(workers) as pool:
            for e, ep in enumerate(pool.imap(_collect, jobs, chunksize=8)):
                lo = e * rows
                pixels[lo : lo + rows] = ep["pixels"]
                for k, d in cols.items():
                    d[lo : lo + rows] = ep[k]
                success[e], catch_step[e], land_step[e] = ep["success"], ep["catch_step"], ep["land_step"]
                launch[e] = ep["launch"]
                if (e + 1) % 1000 == 0 or e + 1 == episodes:
                    rate = (e + 1) / (time.time() - started)
                    print(f"  {e + 1:>6}/{episodes} episodes  ({rate:.0f} ep/s)", flush=True)

        f["ep_success"], f["ep_catch_step"], f["ep_land_step"], f["ep_launch"] = success, catch_step, land_step, launch
        f.attrs["env"] = json.dumps({
            "name": "BallCatch", "control_hz": ball.CONTROL_HZ, "substeps": ball.SUBSTEPS,
            "steps_per_episode": steps, "img_size": ball.IMG_SIZE, "max_speed": ball.MAX_SPEED,
            "world_w": ball.WORLD_W, "ball_r": ball.BALL_R, "basket_half": ball.BASKET_HALF,
            "seed": seed,
        })
    tmp.rename(out)
    print(f"wrote {out}  ({out.stat().st_size / 1e9:.2f} GB, {time.time() - started:.0f}s)")


# --------------------------------------------------------------------------- #
#  Samples, read back from the file
# --------------------------------------------------------------------------- #
def write_samples(path: Path, out_dir: Path, frameskip: int = 5, num: int = 8) -> None:
    import cv2

    from config import get_config
    from src.data import build_datasets
    from src.utils import save_video

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as f:
        offset, length = f["ep_offset"][:], f["ep_len"][:]
        success, policy = f["ep_success"][:], f["ep_policy"][:]
        catch_step, land_step = f["ep_catch_step"][:], f["ep_land_step"][:]
        launch = f["ep_launch"][:]
        env = json.loads(f.attrs["env"])

        # A mix: caught and missed, expert and random.
        picks = []
        for pol in (0, 1):
            for ok in (True, False):
                picks += list(np.flatnonzero((policy == pol) & (success == ok))[: num // 4])

        sheet = []
        for e in picks:
            lo, hi = int(offset[e]), int(offset[e] + length[e])
            frames = f["pixels"][lo:hi]
            actions = f["action"][lo:hi, 0]
            tag = f"ep{e:05d}_{'expert' if policy[e] == 0 else 'random'}_{'caught' if success[e] else 'missed'}"
            big = np.repeat(np.repeat(frames, 3, axis=1), 3, axis=2)
            for t in range(len(big)):  # step counter, and the action about to be applied
                a = actions[t]
                txt = f"t={t:02d}" + ("" if np.isnan(a) else f" a={a:+.2f}")
                cv2.putText(big[t], txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            save_video(out_dir / f"{tag}.mp4", big, fps=env["control_hz"])
            sheet.append(np.concatenate(list(np.repeat(np.repeat(frames[::frameskip], 2, 1), 2, 2)), axis=1))
        cv2.imwrite(str(out_dir / "contact_sheet.png"), cv2.cvtColor(np.concatenate(sheet, 0), cv2.COLOR_RGB2BGR))

        rows = int(length[0])
        stats = {
            "file": str(path),
            "episodes": int(len(offset)),
            "rows": int(offset[-1] + length[-1]),
            "steps_per_episode": rows - 1,
            "seconds_per_episode": (rows - 1) / env["control_hz"],
            "success_rate": {"all": float(success.mean()),
                             "expert": float(success[policy == 0].mean()),
                             "random": float(success[policy == 1].mean())},
            "land_step_percentiles_0_10_50_90_100": np.percentile(land_step[land_step >= 0], [0, 10, 50, 90, 100]).tolist(),
            "launch_height_m": [float(launch[:, 0].min()), float(launch[:, 0].max())],
            "launch_speed_mps": [float(launch[:, 1].min()), float(launch[:, 1].max())],
            "launch_angle_deg": [float(np.degrees(launch[:, 2]).min()), float(np.degrees(launch[:, 2]).max())],
            "sample_episodes": {int(e): {"policy": "expert" if policy[e] == 0 else "random",
                                         "caught": bool(success[e]), "catch_step": int(catch_step[e]),
                                         "land_step": int(land_step[e])} for e in picks},
            "env": env,
        }

    # A few training windows exactly as the training loader builds them.
    cfg, _ = get_config(["--preset", "ball_pixels", f"data.h5_path={path}", f"data.frameskip={frameskip}"])
    train_set, _, reader, normalizer = build_datasets(cfg)
    from src.data import ImageTransform

    tf = ImageTransform(cfg.data.img_size)
    windows = []
    rng = np.random.default_rng(0)
    for i in rng.choice(len(train_set), 6, replace=False):
        item = train_set[int(i)]
        imgs = tf.denormalize(item["pixels"]).numpy()  # (T, H, W, 3)
        imgs = np.repeat(np.repeat(imgs, 2, 1), 2, 2)
        act = normalizer.denormalize("action", item["action"].numpy())  # (T, frameskip)
        for t in range(len(imgs)):
            label = " ".join(f"{a:+.1f}" for a in act[t]) if t < len(imgs) - 1 else "(target)"
            cv2.putText(imgs[t], label, (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        windows.append(np.concatenate(list(imgs), axis=1))
    cv2.imwrite(str(out_dir / "training_windows.png"), cv2.cvtColor(np.concatenate(windows, 0), cv2.COLOR_RGB2BGR))
    stats["training_windows"] = {
        "train_windows": len(train_set),
        "seq_len": cfg.data.seq_len,
        "frameskip": cfg.data.frameskip,
        "pixels_shape": list(item["pixels"].shape),
        "action_shape": list(item["action"].shape),
        "action_mean_std": [normalizer.stats["action"][0].tolist(), normalizer.stats["action"][1].tolist()],
    }
    reader.close()
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"samples -> {out_dir}")
    print(json.dumps({k: stats[k] for k in ("episodes", "rows", "seconds_per_episode", "success_rate")}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--steps", type=int, default=60, help="control steps per episode (20 Hz)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--samples-only", action="store_true")
    args = p.parse_args()

    if not args.samples_only:
        if args.out.exists():
            raise SystemExit(f"{args.out} exists; delete it or pass --out")
        generate(args.out, args.episodes, args.steps, args.seed, args.workers)
    write_samples(args.out, args.samples)


if __name__ == "__main__":
    main()
