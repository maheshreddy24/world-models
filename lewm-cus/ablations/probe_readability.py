"""How *easily* is the cube read off a frozen latent?  Probe capacity and sample efficiency.

    python ablations/probe_readability.py                          # every checkpoints/*/epoch_007.pt
    python ablations/probe_readability.py --target pos --sizes 1000 10000
    python ablations/probe_readability.py --ckpts checkpoints/vit_cube/epoch_007.pt

`probe_cube.py` answers "is the cube's position in there at all" with one probe
on one training-set size.  That conflates two different things: information the
encoder stores explicitly, and information a probe can dig out given enough
capacity and enough examples.  A planner's L2 cost can only use the first kind.

Three knobs are varied against each other, all scored on the same held-out
frames from episodes the world model never trained on:

    probe      linear | mlp        capacity: entangled information needs the mlp
    train size 1k | 10k | 100k     sample efficiency: easy features need few examples
    target     pos | motion        the cube's position, and its displacement over
                                   one latent step (frameskip env steps apart)

Two readings of the result:

*Capacity gap* (mlp - linear at the same size).  Small means the quantity sits
in the latent in a directly usable form.  Large means it is present but
entangled, which a linear planning cost cannot exploit.

*Sample efficiency*, reported both as the error curve over train size and as a
prequential (online) code length, the MDL probing idea of Voita & Titov: walk
through the training pool in blocks, train on everything seen so far, and pay
for the next block with the current probe.  A feature that is easy to read is
cheap to transmit early, so its code is short.  Regression has no uniform code,
so a block costs the Gaussian surprisal of its residuals at a fixed resolution
(`--sigma-cm`, 1 cm by default: the scale the task is scored at).  The first
block is charged to the marginal predictor, identically for every model, so
differences between models are differences in the probe's own coding.  The
baseline column is the same pool coded by the training-set mean, so
`compression = baseline / model`; higher is easier to read.

An oracle model's own state token is dropped automatically: probing it would
measure that model's state MLP, not what its pixels encode.  Such runs are
marked with `*` in the table.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probe_cube import CubeProbe, EpisodeFrames, features, metrics
from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
from src.utils import load_model, save_json, set_seed

TARGETS = ("pos", "motion")
FRACTIONS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1.0)


def fmt_size(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


# --------------------------------------------------------------------------- #
#  Features and targets
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_split(model, dataset, device, workers: int, max_frames: int, desc: str) -> dict:
    """Encode episodes until `max_frames` frames are collected.

    Frames are `frameskip` env steps apart, so consecutive ones are one latent
    step apart and can be paired for the motion target.

    Returns:
        {"pos": (X, Y), "motion": (X, Y)} with X on the CPU and Y in metres.
    """
    loader = DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda")
    out = {t: ([], []) for t in TARGETS}
    total = 0

    progress = tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty())
    for obs, state, cube in progress:
        state = state.to(device, non_blocking=True)[None] if state.numel() else None
        emb = model.encode(obs.to(device, non_blocking=True)[None], state)[0]  # (n, P, D)
        if model.oracle_encoder is not None:
            emb = emb[:, :-1]  # the oracle token holds the answer; probe the pixels
        feat = features(emb).cpu()

        out["pos"][0].append(feat)
        out["pos"][1].append(cube)
        # one latent step: [e_t, e_t+1] -> where the cube went
        out["motion"][0].append(torch.cat([feat[:-1], feat[1:]], dim=-1))
        out["motion"][1].append(cube[1:] - cube[:-1])

        total += len(feat)
        progress.set_postfix(frames=total)
        # The motion set loses a frame per episode, so keep going until *it* is full too.
        if total - len(out["motion"][0]) >= max_frames:
            break
    progress.close()

    return {t: (torch.cat(X)[:max_frames], torch.cat(Y)[:max_frames]) for t, (X, Y) in out.items()}


# --------------------------------------------------------------------------- #
#  Probes
# --------------------------------------------------------------------------- #
def fit(kind: str, X: torch.Tensor, Y: torch.Tensor, device, steps: int, lr: float, wd: float, batch_size: int):
    """AdamW on standardised features and targets; statistics from this train split only."""
    probe = CubeProbe(X.size(1), kind).to(device)
    X, Y = X.to(device), Y.to(device)
    probe.set_stats(X, Y)
    Xn, Yn = (X - probe.x_mean) / probe.x_std, (Y - probe.y_mean) / probe.y_std

    probe.train()
    opt = torch.optim.AdamW(probe.net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    batch = min(batch_size, len(Xn))
    for _ in range(steps):
        idx = torch.randint(len(Xn), (batch,), device=device)
        loss = F.mse_loss(probe.net(Xn[idx]), Yn[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return probe.eval()


@torch.no_grad()
def predict(probe, X: torch.Tensor, device, chunk: int = 100_000) -> torch.Tensor:
    return torch.cat([probe(X[i : i + chunk].to(device)).cpu() for i in range(0, len(X), chunk)])


# --------------------------------------------------------------------------- #
#  Minimum description length
# --------------------------------------------------------------------------- #
def code_bits(pred: torch.Tensor, true: torch.Tensor, sigma_cm: float) -> float:
    """Gaussian surprisal of the residuals at a fixed resolution, in bits.

    Constant per value, so it only ranks models; the ranking is the point.
    """
    err = (pred - true) * 100  # cm
    const = 0.5 * math.log2(2 * math.pi * sigma_cm**2)
    return float((const + err.pow(2) / (2 * sigma_cm**2 * math.log(2))).sum())


def online_codelength(kind, X, Y, device, args) -> dict:
    """Prequential code: train on the prefix, pay for the next block, repeat."""
    n = len(X)
    bounds = sorted({max(args.mdl_min_block, int(f * n)) for f in FRACTIONS} | {n})
    bounds = [b for b in bounds if b <= n]

    # The first block has no trained probe yet, so the marginal pays for it —
    # the same charge for every model, so comparisons are unaffected.
    mean = Y[: bounds[0]].mean(0, keepdim=True).expand(bounds[0], -1)
    total = code_bits(mean, Y[: bounds[0]], args.sigma_cm)

    for lo, hi in zip(bounds, bounds[1:]):
        probe = fit(kind, X[:lo], Y[:lo], device, args.mdl_steps, args.lr, args.weight_decay, args.batch_size)
        total += code_bits(predict(probe, X[lo:hi], device), Y[lo:hi], args.sigma_cm)

    baseline = code_bits(Y.mean(0, keepdim=True).expand(n, -1), Y, args.sigma_cm)
    return {
        "bits": total,
        "baseline_bits": baseline,
        "compression": baseline / total,
        "blocks": len(bounds),
        "num_frames": n,
    }


# --------------------------------------------------------------------------- #
#  One checkpoint
# --------------------------------------------------------------------------- #
def analyse(ckpt: Path, args, device) -> dict:
    model, cfg = load_model(ckpt, device=str(device))
    if cfg.data.backend != "h5":
        raise SystemExit(f"{ckpt} was trained on the scene recording; this reads the cube one")

    state_key = model.oracle_key if model.oracle_encoder is not None else None
    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    normalizer = get_normalizer(cfg, reader, (cfg.data.obs_key, "action") + ((state_key,) if state_key else ()))
    transform = ImageTransform(cfg.data.img_size)

    # The world model's own split: scored on episodes its encoder never saw.
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    rng = np.random.default_rng(args.seed)
    train_eps = rng.permutation(train_eps)
    val_eps = rng.permutation(val_eps)

    def split(episodes):
        return EpisodeFrames(reader, episodes, cfg.data.obs_key, cfg.data.frameskip, transform, normalizer, state_key)

    train = encode_split(model, split(train_eps), device, args.workers, max(args.sizes), "encode train")
    val = encode_split(model, split(val_eps), device, args.workers, args.val_frames, "encode val")

    results = {}
    for target in args.targets:
        Xtr, Ytr = train[target]
        Xva, Yva = val[target]
        per_target = {"num_train_frames": len(Xtr), "num_val_frames": len(Xva), "feature_dim": Xtr.size(1), "sizes": {}}

        for kind in args.probes:
            for size in args.sizes:
                if size > len(Xtr):
                    continue
                probe = fit(kind, Xtr[:size], Ytr[:size], device, args.steps, args.lr, args.weight_decay, args.batch_size)
                m = metrics(predict(probe, Xva, device), Yva, Ytr[:size].mean(0))
                per_target["sizes"].setdefault(str(size), {})[kind] = m
            if not args.skip_mdl:
                per_target.setdefault("mdl", {})[kind] = online_codelength(kind, Xtr, Ytr, device, args)

        results[target] = per_target

    reader.close()
    return {"run": ckpt.parent.name, "ckpt": str(ckpt), "encoder": cfg.model.encoder,
            "oracle_obs": cfg.model.oracle_obs, "image_token_only": state_key is not None, **{"targets": results}}


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def report(all_results: list[dict], args) -> None:
    for target in args.targets:
        label = "cube position" if target == "pos" else "cube displacement over one latent step"
        print(f"\n{'=' * 110}\nheld-out error (cm) — {label}\n{'=' * 110}")
        head = "".join(f"{kind[:3]} {fmt_size(size)}".rjust(11) for kind in args.probes for size in args.sizes)
        print(f"{'run':26}{head}{'  mlp-lin':>10}{'  MDL kbits':>12}{'  compr':>8}")
        for r in all_results:
            t = r["targets"][target]
            cells, last = "", {}
            for kind in args.probes:
                for size in args.sizes:
                    m = t["sizes"].get(str(size), {}).get(kind)
                    cells += ("-" if m is None else f"{m['mean_err_cm']:.2f}").rjust(11)
                    if m is not None:
                        last[kind] = m["mean_err_cm"]
            gap = last.get("linear", float("nan")) - last.get("mlp", float("nan"))
            mdl = t.get("mdl", {}).get("linear")
            name = r["run"] + ("*" if r["image_token_only"] else "")
            print(f"{name:26}{cells}{gap:10.2f}"
                  + (f"{mdl['bits'] / 1000:12.1f}{mdl['compression']:8.2f}" if mdl else f"{'-':>12}{'-':>8}"))
    print("\n* image token only (the model's own oracle token was dropped)")
    print("mlp-lin: how much capacity buys at the largest size — big means entangled, not absent")
    print(f"MDL: prequential code of the linear probe over the training pool at {args.sigma_cm:g} cm resolution; "
          "compression is against the training-mean code, higher is easier to read")


def plot(all_results: list[dict], args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(args.targets), figsize=(6 * len(args.targets), 4.2), squeeze=False)
    for ax, target in zip(axes[0], args.targets):
        for r in all_results:
            sizes = r["targets"][target]["sizes"]
            for kind, style in (("linear", "-o"), ("mlp", "--s")):
                xs = sorted(int(s) for s in sizes if kind in sizes[s])
                if not xs:
                    continue
                ys = [sizes[str(x)][kind]["mean_err_cm"] for x in xs]
                ax.plot(xs, ys, style, label=f"{r['run']} ({kind})", alpha=0.85, ms=4)
        ax.set(xscale="log", xlabel="probe training frames", ylabel="held-out error (cm)",
               title="cube position" if target == "pos" else "cube displacement")
        ax.axhline(4.0, color="gray", ls=":", lw=1)  # the simulator's success tolerance
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Probe capacity and sample efficiency across checkpoints.")
    parser.add_argument("--ckpts", nargs="*", default=None, help="default: every checkpoints/*/epoch_<epoch>.pt")
    parser.add_argument("--epoch", type=int, default=7, help="which epoch to collect when --ckpts is not given")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--probes", nargs="+", choices=("linear", "mlp"), default=["linear", "mlp"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--val-frames", type=int, default=8000, help="held-out frames every probe is scored on")
    parser.add_argument("--steps", type=int, default=5000, help="optimiser steps per probe")
    parser.add_argument("--mdl-steps", type=int, default=2000, help="steps per prequential block (there are ~11)")
    parser.add_argument("--mdl-min-block", type=int, default=100)
    parser.add_argument("--sigma-cm", type=float, default=1.0, help="resolution the code is written at")
    parser.add_argument("--skip-mdl", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="default: ablations/results/readability")
    args = parser.parse_args(argv)
    args.sizes = sorted(args.sizes)
    set_seed(args.seed)

    if args.ckpts:
        ckpts = [Path(c) for c in args.ckpts]
    else:
        ckpts = sorted(ROOT.glob(f"checkpoints/*/epoch_{args.epoch:03d}.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoints found; pass --ckpts or check for epoch_{args.epoch:03d}.pt files")
    missing = [c for c in ckpts if not c.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"probing {len(ckpts)} checkpoints: {', '.join(c.parent.name for c in ckpts)}")

    out_dir = Path(args.out) if args.out else ROOT / "ablations" / "results" / "readability"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ckpt in ckpts:
        print(f"\n--- {ckpt.parent.name} / {ckpt.name}")
        all_results.append(analyse(ckpt, args, device))
        save_json(out_dir / "readability.json", {"results": all_results, "args": vars(args)})

    report(all_results, args)
    plot(all_results, args, out_dir / "readability.png")
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
