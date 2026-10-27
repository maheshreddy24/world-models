"""Per-model analysis for the regulariser comparison: probe, folding, rollouts, off-manifold.

    python compare/analysis.py --ckpt checkpoints/compare/tworoom/sigreg_0.01/epoch_009.pt

Everything is measured on the world model's own held-out episodes (its seeded
split), and every model of a task sees exactly the same frames and windows
(fixed seeds, same split), so per-window numbers line up across models.

Frame sets
    bank      every `frameskip`-th row of `--bank-episodes` random *training*
              episodes: fits the probe, is the "real latent" reference for
              off-manifold distance, and the neighbour pool for local stretch
    heldout   every `frameskip`-th row of the held-out episodes, `--heldout-frames`
              of them: the folding metrics
    windows   `--windows` random held-out start points; each has `history` real
              context frames and a `--horizon`-step autoregressive rollout on the
              recorded actions

State
    The visible state, as the probe regresses it (src/tasks.py): TwoRoom agent
    x, y; PushT agent x, y, block x, y, block angle as (cos, sin). Velocities
    are left out: one frame cannot show them. Each dimension is z-scored with
    the bank's statistics, for every distance in state space below.

Metrics (analysis.json)
    probe       ridge probe latent -> state, fitted on the bank; on predicted latents
                at every horizon step h = 1..H, error per quantity (px / deg) and
                `std_err` = RMS over target dimensions of (pred - true) / std,
                unit-free (the number the correlations use); `floor` is the probe
                on the real frame's latent
    folding     saturation distance (primary): random held-out pairs, latent distance
                vs state distance, binned by state-distance quantile and divided by
                the curve's own plateau (its maximum); `d80` is the state distance
                where it first reaches 80% of the plateau (smaller = more folded),
                `d80_rel` = d80 / median pair state distance. Also participation
                ratio of the latent covariance (effective dimension), and
                trustworthiness / continuity at k = `--tc-k` (self-intersection
                and tearing)
    offmanifold per h: median distance of the predicted latent to its nearest bank
                latent, over the same median for the real held-out frames' latents
                (the windows' true targets). 1 = looks like an encoded real frame.
                The bank holds training episodes only, so no query ever finds a
                frame of its own episode.
    stretch     per window: the local Jacobian of state -> latent around its last
                context frame, a ridge fit of latent offsets on state offsets over
                its `--stretch-k` nearest bank states; its RMS singular value, over
                the model's median. `within` is the Spearman correlation of that
                with std_err at every h.

Writes <run dir>/analysis/{analysis.json, windows.npz}.
"""

from __future__ import annotations

import argparse
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "ablations")]

from common import LinearProbe, load_world_model, probe_features  # noqa: E402
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402

TTY = sys.stderr.isatty()
_T0 = time.time()


def log(stage: str) -> None:
    """Progress line with elapsed time and peak memory: the run log is otherwise silent (no tqdm off a TTY)."""
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[analysis {time.time() - _T0:6.0f}s, peak RSS {peak_gb:.1f} GB] {stage}", flush=True)


# --------------------------------------------------------------------------- #
#  Statistics without scipy
# --------------------------------------------------------------------------- #
def rankdata(x) -> np.ndarray:
    """Ranks 1..n, ties averaged."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / counts)[inv] + 1


def spearman(a, b) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_p_normal(rho: float, n: int) -> float:
    """Two-sided p of a Spearman rho under H0, normal approximation (fine for n in the hundreds)."""
    if not np.isfinite(rho) or n < 4:
        return float("nan")
    return float(math.erfc(abs(rho) * math.sqrt(n - 1) / math.sqrt(2)))


# --------------------------------------------------------------------------- #
#  Reading frames
# --------------------------------------------------------------------------- #
class EpisodeFrames(Dataset):
    """Every `stride`-th row of each episode: uint8 frames and the probe targets of their state."""

    def __init__(self, reader: H5Reader, episodes, task, stride: int):
        self.reader, self.episodes, self.task, self.stride = reader, np.asarray(episodes), task, stride

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        start = int(self.reader.ep_offset[ep])
        stop = start + int(self.reader.ep_len[ep])
        # copies, not strided views: a view would ship the whole episode to the main process
        frames = np.ascontiguousarray(self.reader.span("pixels", start, stop)[:: self.stride])
        obs = torch.from_numpy(self.reader.span(self.task.state_key, start, stop)[:: self.stride].astype(np.float32))
        return frames, self.task.targets(obs).clone(), torch.full((len(frames),), ep, dtype=torch.long)


class Windows(Dataset):
    """Rollout windows: `n_frames` frames `frameskip` rows apart and the action blocks between them."""

    def __init__(self, reader: H5Reader, starts: np.ndarray, task, normalizer, frameskip: int, n_frames: int):
        self.reader, self.starts, self.task, self.norm = reader, starts, task, normalizer
        self.fs, self.n = frameskip, n_frames

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        ep, start = (int(v) for v in self.starts[i])
        lo = int(self.reader.ep_offset[ep]) + start
        span = (self.n - 1) * self.fs
        frames = np.ascontiguousarray(self.reader.span("pixels", lo, lo + span + 1)[:: self.fs])
        obs = torch.from_numpy(self.reader.span(self.task.state_key, lo, lo + span + 1)[:: self.fs].astype(np.float32))
        act = np.nan_to_num(self.reader.span("action", lo, lo + span), nan=0.0)
        act = self.norm.normalize("action", act).astype(np.float32).reshape(self.n - 1, -1)
        return torch.from_numpy(frames), self.task.targets(obs).clone(), torch.from_numpy(act)


@torch.no_grad()
def encode_frames(model, frames: torch.Tensor, transform: ImageTransform, device, chunk: int = 256) -> torch.Tensor:
    """uint8 frames (N, H, W, 3) -> latents (N, P, D) on `device`, normalised on the GPU.

    common.encode_frames normalises on the CPU: a float copy of every batch, of a different size
    per episode, which glibc's allocator fragments and never returns (+0.2 GB per 1000 frames,
    enough to be OOM-killed on a 3000-episode bank). Here only uint8 frames touch the CPU.
    """
    out = []
    for i in range(0, len(frames), chunk):
        x = transform(frames[i : i + chunk].to(device, non_blocking=True))
        out.append(model.encode(x[:, None])[:, 0].float())
    return torch.cat(out)


@torch.no_grad()
def encode_episodes(model, transform, reader, episodes, task, stride, device, workers, desc):
    """Latents (N, F), targets (N, K) and episode ids (N,) of every `stride`-th row."""
    loader = DataLoader(EpisodeFrames(reader, episodes, task, stride), batch_size=None, num_workers=workers)
    zs, ys, es = [], [], []
    for frames, target, ep in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not TTY):
        zs.append(encode_frames(model, torch.as_tensor(frames), transform, device).flatten(1).cpu())
        ys.append(target)
        es.append(ep)
    return torch.cat(zs), torch.cat(ys), torch.cat(es)


def window_starts(reader: H5Reader, episodes, frameskip: int, n_frames: int, stride: int) -> np.ndarray:
    """(M, 2) [episode, start row] of every window that fits inside its episode (last row has no action)."""
    span = (n_frames - 1) * frameskip
    out = []
    for ep in episodes:
        last = int(reader.ep_len[ep]) - 1 - span
        if last >= 0:
            s = np.arange(0, last + 1, stride)
            out.append(np.stack([np.full_like(s, ep), s], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.int64)


def choose_horizon(reader, episodes, frameskip, history, wanted: int, min_windows: int, stride: int) -> int:
    """The largest horizon <= `wanted` with at least `min_windows` windows (TwoRoom episodes are short)."""
    for h in range(wanted, 0, -1):
        if len(window_starts(reader, episodes, frameskip, history + h, stride)) >= min_windows:
            return h
    raise SystemExit("no horizon leaves enough rollout windows in the held-out episodes")


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nn_dist(query: torch.Tensor, bank: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Distance of every query row to its nearest bank row (both on the same device)."""
    out = [torch.cdist(query[i : i + chunk], bank).min(1).values for i in range(0, len(query), chunk)]
    return torch.cat(out)


@torch.no_grad()
def saturation(S: torch.Tensor, Z: torch.Tensor, pairs: int, bins: int, frac: float, gen: torch.Generator) -> dict:
    """Latent vs state distance over random pairs, normalised by its plateau; d80 where it reaches `frac`."""
    n = len(S)
    i = torch.randint(n, (pairs,), generator=gen)
    j = torch.randint(n - 1, (pairs,), generator=gen)
    j = j + (j >= i).long()  # never a frame with itself
    ds = (S[i] - S[j]).norm(dim=1)
    dz = (Z[i] - Z[j]).norm(dim=1)
    edges = torch.quantile(ds, torch.linspace(0, 1, bins + 1))
    idx = torch.bucketize(ds, edges[1:-1])
    count = torch.bincount(idx, minlength=bins)
    keep = count > 0  # repeated quantiles (many identical states) leave bins empty
    x = (torch.zeros(bins).index_add_(0, idx, ds) / count.clamp_min(1))[keep]
    y = (torch.zeros(bins).index_add_(0, idx, dz) / count.clamp_min(1))[keep]
    plateau = float(y.max())
    g = (y / plateau).numpy()
    xs, gs = np.concatenate([[0.0], x.numpy()]), np.concatenate([[0.0], g])  # the curve starts at (0, 0)
    k = int(np.argmax(gs >= frac))
    d80 = float(xs[k - 1] + (frac - gs[k - 1]) * (xs[k] - xs[k - 1]) / max(gs[k] - gs[k - 1], 1e-12))
    median = float(ds.median())
    return {"d80": d80, "d80_rel": d80 / median, "median_state_dist": median, "plateau": plateau,
            "curve_state_dist": x.tolist(), "curve_latent_frac": g.tolist()}


def participation_ratio(Z: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(torch.cov(Z.double().T)).clamp_min(0)
    return float(ev.sum() ** 2 / ev.square().sum())


@torch.no_grad()
def trust_continuity(S: torch.Tensor, Z: torch.Tensor, k: int) -> tuple[float, float]:
    """Venna & Kaski trustworthiness (latent neighbours that are not state neighbours) and continuity."""
    n = len(S)

    def ranks(X):
        d = torch.cdist(X, X)
        d.fill_diagonal_(float("inf"))
        order = d.argsort(dim=1)
        r = torch.empty_like(order)
        r.scatter_(1, order, torch.arange(n, device=X.device).expand(n, n).contiguous())
        return r + 1  # 1 = nearest; self is n

    rs, rz = ranks(S), ranks(Z)
    ns, nz = rs <= k, rz <= k
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    trust = 1 - norm * float(((rs - k) * (nz & ~ns)).sum())
    cont = 1 - norm * float(((rz - k) * (ns & ~nz)).sum())
    return trust, cont


@torch.no_grad()
def local_stretch(s_a, z_a, S_bank, Z_bank, k: int, chunk: int = 512) -> torch.Tensor:
    """RMS singular value of the ridge-fitted local Jacobian d latent / d state at each anchor."""
    out = []
    dim = s_a.size(1)
    for i in range(0, len(s_a), chunk):
        sa, za = s_a[i : i + chunk], z_a[i : i + chunk]
        idx = torch.cdist(sa, S_bank).topk(k, largest=False).indices  # (c, k)
        dS = S_bank[idx] - sa[:, None]  # (c, k, ds)
        dZ = Z_bank[idx] - za[:, None]  # (c, k, F)
        gram = dS.transpose(1, 2) @ dS
        ridge = 1e-3 * gram.diagonal(dim1=1, dim2=2).mean(1).clamp_min(1e-8)
        gram = gram + ridge[:, None, None] * torch.eye(dim, device=sa.device)
        J = torch.linalg.solve(gram, dS.transpose(1, 2) @ dZ)  # (c, ds, F)
        out.append(J.flatten(1).norm(dim=1) / math.sqrt(dim))
    return torch.cat(out)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=15, help="rollout steps (lowered if too few windows fit)")
    p.add_argument("--windows", type=int, default=3000, help="rollout windows (random held-out start points)")
    p.add_argument("--min-windows", type=int, default=500)
    p.add_argument("--bank-episodes", type=int, default=3000, help="training episodes in the reference bank")
    p.add_argument("--heldout-frames", type=int, default=10000, help="held-out frames for the folding metrics")
    p.add_argument("--pairs", type=int, default=200_000, help="random pairs for the saturation curve")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--plateau-frac", type=float, default=0.8)
    p.add_argument("--tc-frames", type=int, default=5000, help="frames for trustworthiness / continuity")
    p.add_argument("--tc-k", type=int, default=10)
    p.add_argument("--stretch-k", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1e-3, help="probe ridge, relative to the sample count")
    p.add_argument("--batch", type=int, default=64, help="windows per rollout batch")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0, help="frame / window sampling (keep equal across models)")
    p.add_argument("--out", type=Path, default=None, help="default: <run dir>/analysis")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, task = load_world_model(args.ckpt, device)
    log(f"loaded {args.ckpt}")
    if model.encoder.num_patches != 1:
        sys.exit("this analysis expects one latent vector per frame (ViT CLS / DINO CLS), not a patch grid")
    set_seed(cfg.seed)
    transform = ImageTransform(cfg.data.img_size)
    fs, hist = cfg.data.frameskip, cfg.data.history
    out = args.out or args.ckpt.parent / "analysis"
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, ("action",))
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    bank_eps = rng.choice(np.sort(train_eps), size=min(args.bank_episodes, len(train_eps)), replace=False)

    # -- bank (training episodes): probe fit, off-manifold reference, stretch neighbours
    Zb, Yb, _ = encode_episodes(model, transform, reader, bank_eps, task, fs, device, args.workers, "encode bank")
    probe = LinearProbe.fit(Zb, Yb, args.ridge, grid=4, device=device)
    s_mean, s_std = Yb.mean(0), Yb.std(0).clamp_min(1e-6)
    Zb_d, Sb_d = Zb.to(device), ((Yb - s_mean) / s_std).to(device)
    log(f"bank: {len(Zb)} frames of {len(bank_eps)} training episodes, probe fitted")

    # -- held-out frames: folding
    Zh, Yh, _ = encode_episodes(model, transform, reader, np.sort(val_eps), task, fs, device, args.workers,
                                "encode held-out")
    pick = torch.from_numpy(rng.choice(len(Zh), size=min(args.heldout_frames, len(Zh)), replace=False))
    Zh, Sh = Zh[pick], (Yh[pick] - s_mean) / s_std
    log(f"held-out: {len(Zh)} frames")
    fold = saturation(Sh, Zh, args.pairs, args.bins, args.plateau_frac, gen)
    fold["participation_ratio"] = participation_ratio(Zh)
    fold["participation_ratio_frac"] = fold["participation_ratio"] / Zh.size(1)
    tc = torch.from_numpy(rng.choice(len(Zh), size=min(args.tc_frames, len(Zh)), replace=False))
    fold["trustworthiness"], fold["continuity"] = trust_continuity(Sh[tc].to(device), Zh[tc].to(device), args.tc_k)
    fold["folding_1_minus_trust"] = 1 - fold["trustworthiness"]
    log("folding metrics done")

    # -- rollout windows on held-out episodes
    H = choose_horizon(reader, val_eps, fs, hist, args.horizon, args.min_windows, fs)
    starts = window_starts(reader, np.sort(val_eps), fs, hist + H, fs)
    starts = starts[np.sort(rng.choice(len(starts), size=min(args.windows, len(starts)), replace=False))]
    loader = DataLoader(Windows(reader, starts, task, normalizer, fs, hist + H), batch_size=args.batch,
                        num_workers=args.workers)
    q_names = [q.name for q in task.quantities]
    rec = {k: [] for k in ("err", "err_floor", "std_err", "std_err_floor", "mse", "copy_mse", "nn_pred", "nn_true",
                           "anchor_s", "anchor_z")}
    with torch.no_grad():
        for frames, targets, act in tqdm(loader, desc="rollouts", dynamic_ncols=True, disable=not TTY):
            b, n = frames.shape[:2]
            emb = encode_frames(model, frames.flatten(0, 1), transform, device).view(b, n, 1, -1)
            pred = model.rollout(emb[:, :hist], act.to(device)).float()  # (b, H, P, D): frames hist .. hist+H-1
            true = emb[:, hist:]
            y = targets[:, hist:].to(device)
            yp, yt = probe(pred), probe(true)
            rec["err"].append(task.errors(yp, y).cpu())
            rec["err_floor"].append(task.errors(yt, y).cpu())
            rec["std_err"].append((((yp - y) / probe.y_std) ** 2).mean(-1).sqrt().cpu())
            rec["std_err_floor"].append((((yt - y) / probe.y_std) ** 2).mean(-1).sqrt().cpu())
            rec["mse"].append((pred - true).pow(2).sum(-1).mean(-1).cpu())
            rec["copy_mse"].append((emb[:, hist - 1 : hist] - true).pow(2).sum(-1).mean(-1).cpu())
            rec["nn_pred"].append(nn_dist(pred.flatten(2).flatten(0, 1), Zb_d).view(b, H).cpu())
            rec["nn_true"].append(nn_dist(true.flatten(2).flatten(0, 1), Zb_d).view(b, H).cpu())
            rec["anchor_s"].append(((targets[:, hist - 1] - s_mean) / s_std))
            rec["anchor_z"].append(emb[:, hist - 1].flatten(1).cpu())
    reader.close()
    rec = {k: torch.cat(v) for k, v in rec.items()}
    log(f"rollouts: {len(starts)} windows, horizon {H}")

    stretch = local_stretch(rec["anchor_s"].to(device), rec["anchor_z"].to(device), Sb_d, Zb_d, args.stretch_k).cpu()
    stretch_norm = (stretch / stretch.median()).numpy()

    base = float(rec["nn_true"].median())
    per_h = {
        "h": list(range(1, H + 1)),
        "std_err": rec["std_err"].mean(0).tolist(),
        "std_err_floor": rec["std_err_floor"].mean(0).tolist(),
        "latent_mse": rec["mse"].mean(0).tolist(),
        "copy_mse": rec["copy_mse"].mean(0).tolist(),
        "offmanifold": (rec["nn_pred"].median(0).values / base).tolist(),
        "offmanifold_vs_same_h": (rec["nn_pred"].median(0).values / rec["nn_true"].median(0).values).tolist(),
        "nn_pred_median": rec["nn_pred"].median(0).values.tolist(),
        "nn_true_median": rec["nn_true"].median(0).values.tolist(),
    }
    for k, name in enumerate(q_names):
        unit = task.quantities[k].unit
        per_h[f"{name}_{unit}"] = rec["err"][..., k].mean(0).tolist()
        per_h[f"{name}_{unit}_floor"] = rec["err_floor"][..., k].mean(0).tolist()

    within = []
    for h in range(H):
        rho = spearman(stretch_norm, rec["std_err"][:, h].numpy())
        within.append({"h": h + 1, "rho": rho, "p": spearman_p_normal(rho, len(stretch_norm))})

    results = {
        "ckpt": str(args.ckpt), "task": task.name, "horizon": H, "requested_horizon": args.horizon,
        "windows": len(starts), "bank_frames": len(Zb), "bank_episodes": len(bank_eps),
        "heldout_frames": len(Zh), "heldout_episodes": len(val_eps), "latent_dim": Zh.size(1),
        "quantities": [f"{q.name}_{q.unit}" for q in task.quantities],
        "offmanifold_baseline_nn": base,
        "folding": fold, "per_h": per_h,
        "stretch": {"median_raw": float(stretch.median()), "within_rho_vs_std_err": within},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    save_json(out / "analysis.json", results)
    np.savez_compressed(out / "windows.npz", starts=starts, std_err=rec["std_err"].numpy(), err=rec["err"].numpy(),
                        nn_pred=rec["nn_pred"].numpy(), nn_true=rec["nn_true"].numpy(), stretch=stretch_norm)
    print(f"{task.name} | {args.ckpt.parent.name}: d80_rel {fold['d80_rel']:.3f}, PR {fold['participation_ratio']:.1f}, "
          f"trust {fold['trustworthiness']:.3f} | std_err h=1 {per_h['std_err'][0]:.3f} h={H} {per_h['std_err'][-1]:.3f}"
          f" | off-manifold h={H} {per_h['offmanifold'][-1]:.2f} -> {out}")



if __name__ == "__main__":
    main()
