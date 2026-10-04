"""Inspect an MMBench2 (TD-MPC2 / Newt) task download: stats + sample frames.

    python datasets/inspect_mmbench.py                                   # acrobot-swingup, every split
    python datasets/inspect_mmbench.py --split expert --episodes 4 --steps 10
    python datasets/inspect_mmbench.py --task cheetah-run --root /home/world-models/data

Layout on disk (one folder per split: expert, mixed-small, mixed-large, val, test, zeros):

    <split>/<task>.pt       TensorDict, flat over steps: obs, action, reward, terminated, episode
    <split>/<task>-<k>.png  the frames, 224x224 RGB tiles laid side by side, chunk k holds
                            frames [k*F, (k+1)*F) of the same flat index (F = 4008 here)

obs/action are padded to a shared multitask width (128 / 16). obs padding is zeros, so
"dims used" is exact for obs; action padding is NOT zero (acrobot has 1 real action dim,
yet all 16 vary), so look at the per-dim std instead.  Writes, per split, into --out:
    <split>_grid.png   rows = episodes, cols = evenly spaced steps
    <split>_ep<i>.gif  one full episode
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the frame strips are ~900k x 224 px

SPLITS = ["expert", "mixed-small", "mixed-large", "val", "test", "zeros"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _stub_make_td(cls, state, *args, **kwargs):
    return dict(state["_tensordict"])


class _Unpickler(pickle.Unpickler):
    """Read a pickled TensorDict as a plain dict, so `tensordict` need not be installed."""

    def find_class(self, module, name):
        if module.startswith("tensordict"):
            return _stub_make_td if name == "_make_td" else dict
        return super().find_class(module, name)


_pickle_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def load_td(path: Path) -> dict[str, torch.Tensor]:
    try:
        import tensordict  # noqa: F401

        td = torch.load(path, map_location="cpu", weights_only=False)
        return {k: td[k] for k in td.keys()}
    except ModuleNotFoundError:
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle_module)


class Frames:
    """Random access into the horizontally tiled PNG chunks (decodes a chunk on first use)."""

    def __init__(self, split_dir: Path, task: str):
        self.files = sorted(split_dir.glob(f"{task}-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        if not self.files:
            raise FileNotFoundError(f"no {task}-*.png in {split_dir}")
        with Image.open(self.files[0]) as im:
            self.h = im.size[1]
            self.per_chunk = im.size[0] // self.h
        self.counts = []
        for f in self.files:
            with Image.open(f) as im:
                self.counts.append(im.size[0] // self.h)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self):
        return sum(self.counts)

    def __getitem__(self, i: int) -> np.ndarray:
        k, j = divmod(int(i), self.per_chunk)
        if k not in self._cache:
            self._cache.clear()  # one decoded chunk (~600 MB) at a time
            self._cache[k] = np.asarray(Image.open(self.files[k]).convert("RGB"))
        return self._cache[k][:, j * self.h : (j + 1) * self.h]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def used_cols(x: torch.Tensor) -> int:
    """Number of leading columns that are not identically zero (the rest is multitask padding)."""
    x = torch.nan_to_num(x.float())
    nz = (x != 0).any(0).nonzero()
    return int(nz.max()) + 1 if len(nz) else 0


def describe(name: str, x: torch.Tensor):
    nan = torch.isnan(x.float()).any(-1) if x.ndim > 1 else torch.isnan(x.float())
    v = torch.nan_to_num(x.float())
    print(f"  {name:11s} shape={tuple(x.shape)} dtype={x.dtype}  NaN rows={int(nan.sum())}"
          f"  min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f}")


def report(split: str, td: dict, frames: Frames):
    ep = td["episode"].long()
    ids, lens = torch.unique_consecutive(ep, return_counts=True)
    print(f"\n=== {split} " + "=" * (60 - len(split)))
    print(f"  steps (rows)        {len(ep)}")
    print(f"  episodes            {len(ids)}  (ids {int(ids.min())}..{int(ids.max())})")
    print(f"  episode length      min={int(lens.min())} max={int(lens.max())} mean={lens.float().mean():.1f}"
          f"  (rows incl. the reset frame)")
    for k in ["obs", "action", "reward"]:
        describe(k, td[k])
    print(f"  obs dims used       {used_cols(td['obs'])} of {td['obs'].shape[-1]}")
    print(f"  action dims used    {used_cols(td['action'])} of {td['action'].shape[-1]}")
    a = td["action"][:, : used_cols(td["action"])]
    a = a[~torch.isnan(a).any(-1)]
    if len(a):
        print(f"  action range used   [{a.min():.3f}, {a.max():.3f}]  per-dim std={a.std(0).numpy().round(3)}")
    first = torch.cat([torch.tensor([True]), ep[1:] != ep[:-1]])
    print(f"  NaN action at ep start: {int(torch.isnan(td['action'][first]).any(-1).sum())}/{len(ids)} episodes"
          "  (TD-MPC2 convention: row 0 is the reset obs, no action/reward yet)")
    r = torch.nan_to_num(td["reward"].float())
    returns = torch.zeros(len(ids)).index_add_(0, torch.repeat_interleave(torch.arange(len(ids)), lens), r)
    print(f"  episode return      min={returns.min():.1f} max={returns.max():.1f} mean={returns.mean():.1f}")
    print(f"  terminated=True     {int(td['terminated'].sum())}")
    print(f"  frames              {len(frames)} in {len(frames.files)} png(s), {frames.per_chunk}/png,"
          f" each {frames.h}x{frames.h} RGB  -> {'OK' if len(frames) == len(ep) else 'MISMATCH with rows!'}")
    return ids, lens


def visualize(split: str, td: dict, frames: Frames, lens: torch.Tensor, n_ep: int, n_steps: int,
              out: Path, size: int, seed: int):
    rng = np.random.default_rng(seed)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)[:-1]])
    # sorting keeps the chunk decodes sequential
    picks = np.sort(rng.choice(len(lens), size=min(n_ep, len(lens)), replace=False))
    rows = []
    for e in picks:
        s, n = int(starts[e]), int(lens[e])
        ts = np.linspace(0, n - 1, n_steps).round().astype(int)
        rows.append(np.concatenate([np.asarray(Image.fromarray(frames[s + t]).resize((size, size))) for t in ts], 1))
    grid = Image.fromarray(np.concatenate(rows, 0))
    grid.save(out / f"{split}_grid.png")

    e = int(picks[0])
    s, n = int(starts[e]), int(lens[e])
    gif = [Image.fromarray(frames[s + t]).resize((size, size)) for t in range(0, n, 2)]
    gif[0].save(out / f"{split}_ep{e}.gif", save_all=True, append_images=gif[1:], duration=40, loop=0)
    print(f"  saved {out / f'{split}_grid.png'} (episodes {picks.tolist()}) and {split}_ep{e}.gif")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/home/world-models/data"))
    p.add_argument("--task", default="acrobot-swingup")
    p.add_argument("--split", nargs="*", default=None, help=f"default: every one of {SPLITS} present")
    p.add_argument("--episodes", type=int, default=4, help="episodes in the grid")
    p.add_argument("--steps", type=int, default=8, help="frames per episode in the grid")
    p.add_argument("--size", type=int, default=112, help="frame side in the saved images")
    p.add_argument("--no-viz", action="store_true", help="stats only (skips the slow PNG decode)")
    p.add_argument("--out", type=Path, default=Path("mmbench_viz"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    splits = args.split or [s for s in SPLITS if (args.root / s / f"{args.task}.pt").exists()]
    if not splits:
        sys.exit(f"no {args.task}.pt under {args.root}/<split>/")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"task={args.task}  root={args.root}  splits={splits}")

    for split in splits:
        d = args.root / split
        td = load_td(d / f"{args.task}.pt")
        frames = Frames(d, args.task)
        _, lens = report(split, td, frames)
        if not args.no_viz:
            visualize(split, td, frames, lens, args.episodes, args.steps, args.out, args.size, args.seed)


if __name__ == "__main__":
    main()
