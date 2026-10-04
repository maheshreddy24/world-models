"""Does the latent encode motion?  Probe [e_t, e_t+1] -> the displacement between the two frames.

    python ablations/latent_analysis.py                    # observation_model, epoch 7
    python ablations/latent_analysis.py --ckpt checkpoints/observation_model/epoch_015.pt
    python ablations/latent_analysis.py --probe mlp
    python ablations/latent_analysis.py --moving-cm 4   # stricter "the cube really moved"

A pair is two consecutive latent steps, `frameskip` (5) env steps apart.  A
linear layer maps the concatenated embeddings to how far the cube and the
end-effector moved over that step (cm).  It is fitted on training episodes and
scored on the world model's own held-out episodes.

Every variant is scored on the same held-out pairs:

    static (Δ=0)            "nothing moves"                    the floor to beat
    current frame only      probe on e_t alone                 the expert's next move is partly
                                                               predictable from the state, so this
                                                               is what e_t+1 has to add to
    pair, real e_t+1        probe on [e_t, e_t+1]              is the displacement in the latent?
    pair, predicted ê_t+1   the same probe on [e_t, ê_t+1]     does the world model's step move
                            ê from the predictor, given the    things the right way?
                            H frames up to t + the true action

The cube sits still for most steps, so every metric is also reported on the
steps where the body really moves (> `--moving-cm`, 2 cm by default); `mov cos`
is the cosine between predicted and true displacement there, i.e. whether the
direction is right.  The number of pairs that pass is printed, so the subset
can be checked.  Cube motion is close to all-or-nothing in this recording: in
a moving step the median displacement is ~7 cm, so any threshold from 0.5 to
4 cm keeps roughly the same third of the pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import H5Reader, ImageTransform, get_normalizer, split_episodes
from src.utils import load_model, save_json, set_seed

POSITIONS = {"cube": "privileged_block_0_pos", "ee": "proprio_effector_pos"}  # metres in the recording
AXES = ("x", "y", "z")
MOVING_CM = 2.0  # default: a body "moves" in a step when it travels further than this


class EpisodeSteps(Dataset):
    """One item per episode at latent-step resolution: n+1 observations, n action blocks, positions in cm."""

    def __init__(self, reader, episodes, cfg, transform, normalizer, state_key=None):
        self.reader = reader
        self.episodes = np.asarray(episodes)
        self.obs_key = cfg.data.obs_key
        self.fs = cfg.data.frameskip
        self.transform = transform
        self.normalizer = normalizer
        # Set for a model that reads the oracle state next to the pixels.
        self.state_key = state_key

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        ep = int(self.episodes[i])
        start = int(self.reader.ep_offset[ep])
        n = (int(self.reader.ep_len[ep]) - 1) // self.fs  # the last row's action is NaN
        stop = start + n * self.fs + 1

        obs = self.reader.span(self.obs_key, start, stop)[:: self.fs]
        if self.obs_key == "pixels":
            obs = self.transform(obs)
        else:
            obs = torch.from_numpy(self.normalizer.normalize(self.obs_key, obs).astype(np.float32))

        # action block t is the `frameskip` raw actions carrying step t to t+1, normalised as in training
        act = np.nan_to_num(self.reader.span("action", start, start + n * self.fs), nan=0.0)
        act = self.normalizer.normalize("action", act).astype(np.float32).reshape(n, -1)

        pos = {
            name: torch.from_numpy(self.reader.span(col, start, stop)[:: self.fs].astype(np.float32) * 100)
            for name, col in POSITIONS.items()
        }
        state = torch.zeros(0)
        if self.state_key is not None:
            raw = self.reader.span(self.state_key, start, stop)[:: self.fs]
            state = torch.from_numpy(self.normalizer.normalize(self.state_key, raw).astype(np.float32))
        return obs, torch.from_numpy(act), pos, state


def features(emb: torch.Tensor) -> torch.Tensor:
    """(N, P, D) latents -> (N, F) probe inputs; a patch grid is pooled to 4x4."""
    if emb.size(1) == 1:
        return emb[:, 0]
    g = int(round(emb.size(1) ** 0.5))
    grid = emb.transpose(1, 2).reshape(emb.size(0), emb.size(2), g, g)
    return F.adaptive_avg_pool2d(grid, 4).flatten(1)


@torch.no_grad()
def collect(model, dataset, device, workers: int, desc: str):
    """Every step t >= H-1 of every episode: features of e_t, e_t+1 and ê_t+1, and the displacement t -> t+1.

    Steps before H-1 are skipped because the predictor needs H frames of
    context, and every variant must be scored on the same pairs.
    """
    H = model.context_len
    loader = DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda")
    cur, nxt, pred = [], [], []
    delta = {name: [] for name in POSITIONS}

    for obs, act, pos, state in tqdm(loader, desc=desc, dynamic_ncols=True, disable=not sys.stderr.isatty()):
        state = state.to(device, non_blocking=True)[None] if state.numel() else None
        emb = model.encode(obs.to(device, non_blocking=True)[None], state)[0]  # (n+1, P, D)
        act = act.to(device)

        s = torch.arange(len(act) - H + 1, device=device)  # window s observes frames s .. s+H-1
        ctx = s[:, None] + torch.arange(H, device=device)  # (B, H)
        e_hat = model.rollout(emb[ctx], act[ctx])[:, 0]  # (B, P, D): one step, the true action
        t = s + H - 1  # each pair is (t, t+1)
        if model.oracle_encoder is not None:
            # Probe the image tokens only: the oracle token holds the state the
            # displacement is computed from, so including it answers nothing.
            emb, e_hat = emb[:, :-1], e_hat[:, :-1]

        cur.append(features(emb[t]).cpu())
        nxt.append(features(emb[t + 1]).cpu())
        pred.append(features(e_hat).cpu())
        t = t.cpu()
        for name, p in pos.items():
            delta[name].append(p[t + 1] - p[t])

    targets = torch.cat([torch.cat(delta[name]) for name in POSITIONS], dim=-1)  # (N, 3 * bodies) cm
    return torch.cat(cur), torch.cat(nxt), torch.cat(pred), targets


class Probe(nn.Module):
    """Features -> displacement (cm), with input/output standardisation built in."""

    def __init__(self, dim_in: int, dim_out: int, kind: str = "linear", hidden: int = 512):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim_in, dim_out)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim_out)
            )
        self.register_buffer("x_mean", torch.zeros(dim_in))
        self.register_buffer("x_std", torch.ones(dim_in))
        self.register_buffer("y_mean", torch.zeros(dim_out))
        self.register_buffer("y_std", torch.ones(dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean


def fit(probe: Probe, X, Y, args, device) -> Probe:
    """AdamW on standardised features and targets (training-split statistics only)."""
    X, Y = X.to(device), Y.to(device)
    probe.to(device)
    probe.x_mean.copy_(X.mean(0))
    probe.x_std.copy_(X.std(0) + 1e-6)
    probe.y_mean.copy_(Y.mean(0))
    probe.y_std.copy_(Y.std(0) + 1e-6)
    Xn, Yn = (X - probe.x_mean) / probe.x_std, (Y - probe.y_mean) / probe.y_std

    probe.train()
    opt = torch.optim.AdamW(probe.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for _ in range(args.steps):
        idx = torch.randint(len(Xn), (args.batch_size,), device=device)
        loss = F.mse_loss(probe.net(Xn[idx]), Yn[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return probe.eval()


def score(pred: torch.Tensor, true: torch.Tensor, moving_cm: float) -> dict:
    """Displacement metrics for one body: all held-out steps, and the steps where it moves > moving_cm."""
    err = (pred - true).norm(dim=-1)
    r2 = 1 - (pred - true).pow(2).sum(0) / (true - true.mean(0)).pow(2).sum(0)
    moving = true.norm(dim=-1) > moving_cm
    return {
        "err_cm": err.mean().item(),
        "r2": r2.mean().item(),
        **{f"r2_{a}": r.item() for a, r in zip(AXES, r2)},
        "moving_n": int(moving.sum()),
        "moving_frac": moving.float().mean().item(),
        "moving_true_cm": true[moving].norm(dim=-1).mean().item(),  # how far those steps really moved
        "moving_err_cm": err[moving].mean().item(),
        "moving_cos": F.cosine_similarity(pred[moving], true[moving], dim=-1).mean().item(),
    }


def plot(true: torch.Tensor, real: torch.Tensor, predicted: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(POSITIONS), 3, figsize=(12, 3.8 * len(POSITIONS)))
    for b, body in enumerate(POSITIONS):
        for j, ax in enumerate(axes[b]):
            c = 3 * b + j
            t = true[:, c].numpy()
            ax.scatter(t, real[:, c].numpy(), s=2, alpha=0.2, label="probe [e_t, e_t+1]")
            ax.scatter(t, predicted[:, c].numpy(), s=2, alpha=0.2, label="probe [e_t, ê_t+1]")
            ax.plot([t.min(), t.max()], [t.min(), t.max()], "k--", lw=1)
            ax.set(title=f"{body} Δ{AXES[j]}", xlabel="true (cm)", ylabel="probe (cm)")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, markerscale=5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Probe consecutive latents for the displacement between them.")
    parser.add_argument("--ckpt", default=str(ROOT / "checkpoints/observation_model/epoch_007.pt"))
    parser.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--train-episodes", type=int, default=500, help="training episodes to fit on")
    parser.add_argument("--val-episodes", type=int, default=200, help="held-out episodes to score on")
    parser.add_argument("--moving-cm", type=float, default=MOVING_CM, help="a step counts as moving above this (cm)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="default: ablations/results/<run>_<epoch>")
    args = parser.parse_args(argv)
    set_seed(args.seed)

    ckpt = Path(args.ckpt)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt, device=str(device))
    if cfg.data.backend != "h5":
        raise SystemExit("this analysis reads the OGBench cube recording; the checkpoint was trained on the scene one")

    reader = H5Reader(cfg.data.h5_path, rdcc_mb=cfg.data.rdcc_mb)
    state_key = model.oracle_key if model.oracle_encoder is not None else None
    normalizer = get_normalizer(cfg, reader, (cfg.data.obs_key, "action") + ((state_key,) if state_key else ()))
    transform = ImageTransform(cfg.data.img_size)

    # The world model's own split: scored on episodes the encoder never trained on.
    train_eps, val_eps = split_episodes(reader.num_episodes, cfg.data.val_episodes, cfg.seed, cfg.data.max_episodes)
    rng = np.random.default_rng(args.seed)
    train_eps = rng.choice(train_eps, min(args.train_episodes, len(train_eps)), replace=False)
    val_eps = rng.choice(val_eps, min(args.val_episodes, len(val_eps)), replace=False)

    cur_tr, nxt_tr, _, Y_tr = collect(
        model, EpisodeSteps(reader, train_eps, cfg, transform, normalizer, state_key), device, args.workers, "encode train"
    )
    cur_va, nxt_va, pred_va, Y_va = collect(
        model, EpisodeSteps(reader, val_eps, cfg, transform, normalizer, state_key), device, args.workers, "encode val"
    )

    dim_out = Y_tr.size(1)
    pair = fit(Probe(2 * cur_tr.size(1), dim_out, args.probe), torch.cat([cur_tr, nxt_tr], -1), Y_tr, args, device)
    current = fit(Probe(cur_tr.size(1), dim_out, args.probe), cur_tr, Y_tr, args, device)

    with torch.no_grad():
        preds = {
            "static (Δ=0)": torch.zeros_like(Y_va),
            "current frame only": current(cur_va.to(device)).cpu(),
            "pair, real e_t+1": pair(torch.cat([cur_va, nxt_va], -1).to(device)).cpu(),
            "pair, predicted ê_t+1": pair(torch.cat([cur_va, pred_va], -1).to(device)).cpu(),
        }
    results = {
        name: {
            body: score(p[:, 3 * b : 3 * b + 3], Y_va[:, 3 * b : 3 * b + 3], args.moving_cm)
            for b, body in enumerate(POSITIONS)
        }
        for name, p in preds.items()
    }

    out_dir = Path(args.out) if args.out else ROOT / "ablations" / "results" / f"{ckpt.parent.name}_{ckpt.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        out_dir / f"latent_analysis_{args.probe}.json",
        {"results": results, "num_train_pairs": len(Y_tr), "num_val_pairs": len(Y_va), "args": vars(args)},
    )
    plot(Y_va, preds["pair, real e_t+1"], preds["pair, predicted ê_t+1"], out_dir / f"latent_analysis_{args.probe}.png")

    fs = cfg.data.frameskip
    print(f"\nΔ position over one latent step ({fs} env steps), {args.probe} probe on {cfg.model.encoder} latents"
          + (" (image token only)" if state_key else ""))
    print(f"{len(Y_tr)} train pairs, {len(Y_va)} held-out pairs; 'mov' = steps where the body moves > {args.moving_cm:g} cm\n")
    header = f"{'err cm':>7} {'R2':>6} {'mov err':>8} {'mov cos':>8}"
    print(f"{'':23s} | {'cube':^32s} | {'end-effector':^32s}")
    print(f"{'variant':23s} | {header} | {header}")
    for name, m in results.items():
        cells = [f"{m[b]['err_cm']:7.3f} {m[b]['r2']:6.3f} {m[b]['moving_err_cm']:8.3f} {m[b]['moving_cos']:8.3f}"
                 for b in POSITIONS]
        print(f"{name:23s} | " + " | ".join(cells))
    print(f"\nsteps moving > {args.moving_cm:g} cm (the 'mov' columns):")
    for body, label in (("cube", "cube"), ("ee", "end-effector")):
        m = results["static (Δ=0)"][body]
        print(f"  {label:13s} {m['moving_n']:6d} of {len(Y_va)} held-out pairs ({m['moving_frac']:.0%}), "
              f"moving {m['moving_true_cm']:.2f} cm per step on average")
    print(f"saved {out_dir}")
    reader.close()


if __name__ == "__main__":
    main()
