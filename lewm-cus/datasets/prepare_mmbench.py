"""Convert an MMBench2 task download (TD-MPC2 / Newt format) into the HDF5 table train.py reads.

    python datasets/prepare_mmbench.py                                    # acrobot-swingup, 4 training splits
    python datasets/prepare_mmbench.py --splits expert mixed-large        # a subset
    python datasets/prepare_mmbench.py --size 128                         # downscale frames (smaller file)
    python datasets/prepare_mmbench.py --task cheetah-run --action-dim 6  # another task

then
    python train.py --preset acrobot_pixels

Download first (into --root):
    hf download nicklashansen/mmbench2 --repo-type dataset --include "*/acrobot-swingup*" --local-dir ./data

What changes on the way (see datasets/inspect_mmbench.py for the raw layout):

  * Padding is cut. MMBench pads every task to 128 obs / 16 action columns. obs
    padding is zeros, so the real width is detected; action padding is zeros in
    `expert`/`zeros` but random noise in `mixed-*`/`val`/`test`, so it is detected
    on `expert` (or given with --action-dim).
  * Actions are shifted by one row. MMBench stores the action that *led to* row t
    (row 0 of an episode is the reset, NaN); the HDF5 convention here is the
    action *taken at* row t (the episode's last row is NaN). Same for reward.
  * The frames move from the horizontally tiled PNG strips into one `pixels`
    column (uint8, 100-frame lz4 chunks).

Columns written: pixels (N,S,S,3) uint8, observation (N,obs_dim), action (N,action_dim),
reward (N,), source_split (N,) int8 (index into attrs["splits"]), ep_offset, ep_len.
`val`/`test` are left out by default: train.py holds out its own validation
episodes (`data.val_episodes`), and these two stay untouched for later evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import HOME  # noqa: E402
from inspect_mmbench import Frames, load_td, used_cols  # noqa: E402  (same folder)

TRAIN_SPLITS = ["expert", "mixed-small", "mixed-large", "zeros"]
CHUNK = 100  # frames per pixel chunk, as in the cube recording


def shift_to_taken(x: torch.Tensor, ep: torch.Tensor) -> np.ndarray:
    """Row t's value moves to row t-1 within an episode; each episode's last row becomes NaN."""
    x = x.float().numpy()
    out = np.full_like(x, np.nan)
    same = (ep[1:] == ep[:-1]).numpy()
    out[:-1][same] = x[1:][same]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/home/world-models/data"))
    p.add_argument("--task", default="acrobot-swingup")
    p.add_argument("--splits", nargs="+", default=TRAIN_SPLITS)
    p.add_argument("--out", type=Path, default=None, help="default: $HOME/datasets/mmbench/<task>.h5")
    p.add_argument("--size", type=int, default=224, help="frame side to store (native 224)")
    p.add_argument("--action-dim", type=int, default=None, help="default: detected on the expert split")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = args.out or HOME / "datasets" / "mmbench" / f"{args.task}.h5"
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} exists; pass --overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)

    tds = {s: load_td(args.root / s / f"{args.task}.pt") for s in args.splits}

    obs_dim = max(used_cols(td["obs"]) for td in tds.values())
    action_dim = args.action_dim
    if action_dim is None:
        # expert's padding columns are exactly zero (mixed/val/test fill them with noise),
        # so detect on it even when it is not one of the splits being converted
        ref = args.root / "expert" / f"{args.task}.pt"
        if not ref.exists():
            sys.exit(f"cannot detect the action width without {ref}; pass --action-dim")
        action_dim = used_cols((tds["expert"] if "expert" in tds else load_td(ref))["action"])
    n = sum(len(td["episode"]) for td in tds.values())
    print(f"{args.task}: splits={args.splits} rows={n} obs_dim={obs_dim} action_dim={action_dim} "
          f"frames={args.size}x{args.size} -> {out}")

    tmp = out.with_suffix(".tmp.h5")
    blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    s = args.size
    ep_offset, ep_len = [], []
    with h5py.File(tmp, "w") as f:
        pixels = f.create_dataset("pixels", (n, s, s, 3), np.uint8, chunks=(CHUNK, s, s, 3), **blosc)
        obs_col = f.create_dataset("observation", (n, obs_dim), np.float32)
        act_col = f.create_dataset("action", (n, action_dim), np.float32)
        rew_col = f.create_dataset("reward", (n,), np.float32)
        src_col = f.create_dataset("source_split", (n,), np.int8)

        row = 0
        for k, (split, td) in enumerate(tds.items()):
            ep = td["episode"].long()
            m = len(ep)
            frames = Frames(args.root / split, args.task)
            if len(frames) != m:
                sys.exit(f"{split}: {len(frames)} frames but {m} rows")

            obs_col[row : row + m] = td["obs"][:, :obs_dim].float().numpy()
            act_col[row : row + m] = shift_to_taken(td["action"][:, :action_dim], ep)
            rew_col[row : row + m] = shift_to_taken(td["reward"], ep)
            src_col[row : row + m] = k
            _, lens = torch.unique_consecutive(ep, return_counts=True)
            ep_offset.append(row + np.concatenate([[0], lens.cumsum(0)[:-1].numpy()]))
            ep_len.append(lens.numpy())

            # one decoded PNG strip at a time, written in CHUNK-aligned blocks
            for i in range(0, m, CHUNK):
                block = np.stack([frames[t] for t in range(i, min(i + CHUNK, m))])
                if s != block.shape[1]:
                    block = np.stack([np.asarray(Image.fromarray(x).resize((s, s), Image.BILINEAR)) for x in block])
                pixels[row + i : row + i + len(block)] = block
            row += m
            print(f"  {split:12s} {len(lens):4d} episodes  {m:7d} rows  (rows {row - m}..{row - 1})")

        f["ep_offset"] = np.concatenate(ep_offset).astype(np.int64)
        f["ep_len"] = np.concatenate(ep_len).astype(np.int64)
        f.attrs["splits"] = json.dumps(list(tds))
        f.attrs["source"] = json.dumps({"dataset": "nicklashansen/mmbench2", "task": args.task,
                                        "action": "row t = action taken at t (shifted from MMBench)"})
    tmp.replace(out)
    stats = out.with_suffix(".stats.npz")
    if stats.exists():  # cached normaliser of an older conversion
        stats.unlink()
    print(f"done: {len(np.concatenate(ep_len))} episodes, {out.stat().st_size / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
