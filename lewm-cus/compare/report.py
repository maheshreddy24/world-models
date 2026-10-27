"""Cross-model summary of the regulariser comparison: tables, correlations, figures.

    python compare/report.py --root checkpoints/compare

Reads <root>/<task>/<model>/{analysis/analysis.json, cem/cem.json} and writes
<root>/report/:

    summary.json, summary.md     every number below, per task
    cem_success.png              CEM success rate per model (mean +- std over seeds, seeds as dots)
    probe_error.png              std_err vs horizon, one line per model (+ the real-frame floor, dashed)
    probe_error_<quantity>.png   the same per physical quantity (px / deg)
    offmanifold.png              off-manifold distance vs horizon
    saturation.png               normalised latent distance vs state distance (folding curves)
    folding_rho.png              across models: Spearman rho(folding number, std_err at h) vs h,
                                 and within each model: rho(local stretch, std_err at h) vs h

Across models (n = number of models, 6): for each folding number and each h,
Spearman rho between the per-model number and the per-model mean std_err at h,
with an exact permutation p-value (all n! orderings). Signs: d80_rel and PR are
"less folded when larger", so a negative rho means more folding goes with more
error; 1 - trust is "more folded when larger" (positive rho, same reading).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis import rankdata, spearman  # noqa: E402

FOLDING_NUMBERS = ("d80_rel", "participation_ratio", "folding_1_minus_trust")

# Reference categorical palette (dataviz skill, light mode), fixed order: color follows the model.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def permutation_p(a, b) -> float:
    """Exact two-sided p of Spearman rho under H0 (every ordering equally likely)."""
    a, b = rankdata(a), rankdata(b)
    if len(a) > 9:
        raise ValueError("exact permutation test is for small n")
    obs = abs(np.corrcoef(a, b)[0, 1])
    perms = [abs(np.corrcoef(a, np.array(p))[0, 1]) for p in itertools.permutations(b)]
    return float(np.mean(np.array(perms) >= obs - 1e-12))


def load(root: Path, tasks, models, subdir: str = "", seeds=None) -> dict:
    data = {}
    for task in tasks:
        data[task] = {}
        for m in models:
            d = root / task / m / subdir
            entry = {}
            path = d / "analysis" / "analysis.json"
            if path.exists():
                entry["analysis"] = json.loads(path.read_text())
            # CEM from the per-seed files, not cem.json: two jobs evaluating the same model can leave
            # a cem.json that lists fewer seeds than have been run
            runs = [json.loads(f.read_text()) for f in sorted((d / "cem").glob("seed*.json"))]
            runs = [r for r in runs if seeds is None or r["seed"] in seeds]
            if runs:
                rates = np.array([r["success_rate"] for r in runs])
                entry["cem"] = {"seeds": [r["seed"] for r in runs], "success_rates": rates.tolist(),
                                "success_rate_mean": float(rates.mean()), "success_rate_std": float(rates.std())}
            if entry:
                data[task][m] = entry
    return data


def has_models(task_data: dict) -> bool:
    return any(not k.startswith("_") for k in task_data)


def correlations(task_data: dict) -> dict:
    """rho(folding number, mean std_err at h) across models, for every h."""
    models = [m for m, e in task_data.items() if "analysis" in e]
    if len(models) < 3:
        return {}
    H = min(task_data[m]["analysis"]["horizon"] for m in models)
    out = {}
    for name in FOLDING_NUMBERS:
        x = [task_data[m]["analysis"]["folding"][name] for m in models]
        rows = []
        for h in range(H):
            y = [task_data[m]["analysis"]["per_h"]["std_err"][h] for m in models]
            rows.append({"h": h + 1, "rho": spearman(x, y), "p_exact": permutation_p(x, y)})
        out[name] = rows
    # does folding also track planning? (one number, not per h)
    cem_models = [m for m in models if "cem" in task_data[m]]
    if len(cem_models) >= 3:
        for name in FOLDING_NUMBERS:
            x = [task_data[m]["analysis"]["folding"][name] for m in cem_models]
            y = [task_data[m]["cem"]["success_rate_mean"] for m in cem_models]
            out[f"{name}_vs_cem"] = {"rho": spearman(x, y), "p_exact": permutation_p(x, y), "models": cem_models}
    out["models"] = models
    return out


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def figures(data: dict, models: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    tasks = [t for t in data if has_models(data[t])]

    def panels(n, w=5.2, h=3.6):
        fig, axes = plt.subplots(1, n, figsize=(w * n, h), squeeze=False)
        return fig, axes[0]

    def save(fig, name, axes, legend=True, ncol=6):
        seen = {}
        for ax in axes:  # one legend for the figure, each label once
            for hd, lb in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(lb, hd)
        if legend and seen:
            fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=min(len(seen), ncol),
                       bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(out / name, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # CEM success
    fig, axes = panels(len(tasks))
    for ax, task in zip(axes, tasks):
        ms = [m for m in models if "cem" in data[task].get(m, {})]
        for k, m in enumerate(ms):
            c = data[task][m]["cem"]
            ax.bar(k, c["success_rate_mean"], width=0.7, color=color[m], label=m, zorder=2)
            ax.errorbar(k, c["success_rate_mean"], yerr=c["success_rate_std"], color=INK_2, capsize=3, lw=1, zorder=3)
            ax.scatter([k] * len(c["success_rates"]), c["success_rates"], s=14, color=INK, zorder=4)
            top = max(c["success_rate_mean"] + c["success_rate_std"], *c["success_rates"])
            ax.text(k, top + 3, f"{c['success_rate_mean']:.0f}", ha="center", va="bottom", fontsize=9, color=INK_2)
        ax.set_xticks(range(len(ms)), ms, rotation=30, ha="right")
        ax.set_ylim(0, 112)
        ax.set_ylabel("CEM success rate (%)")
        ax.set_title(task)
        ax.grid(axis="x", visible=False)
    save(fig, "cem_success.png", axes, legend=False)  # the x labels name the bars

    def per_h_plot(key, ylabel, name, floor_key=None):
        fig, axes = panels(len(tasks))
        drew = False
        for ax, task in zip(axes, tasks):
            for m in models:
                a = data[task].get(m, {}).get("analysis")
                if a is None or key not in a["per_h"]:
                    continue
                drew = True
                ax.plot(a["per_h"]["h"], a["per_h"][key], "o-", ms=4, color=color[m], label=m)
                if floor_key and floor_key in a["per_h"]:
                    ax.plot(a["per_h"]["h"], a["per_h"][floor_key], "--", lw=1, color=color[m])
            ax.set_xlabel("horizon h (latent steps, 5 env steps each)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{task} (dashed: probe on the real frame)" if floor_key else task)
        if drew:
            save(fig, name, axes)
        else:
            plt.close(fig)

    per_h_plot("std_err", "probe error, RMS of (pred - true) / std", "probe_error.png", "std_err_floor")
    per_h_plot("offmanifold", "off-manifold distance (1 = real frames)", "offmanifold.png")
    quantities = sorted({q for t in tasks for e in data[t].values() if "analysis" in e
                         for q in e["analysis"]["quantities"]})
    for q in quantities:
        per_h_plot(q, f"probe error: {q.rsplit('_', 1)[0]} ({q.rsplit('_', 1)[1]})", f"probe_error_{q}.png", f"{q}_floor")

    # saturation curves
    fig, axes = panels(len(tasks))
    for ax, task in zip(axes, tasks):
        for m in models:
            a = data[task].get(m, {}).get("analysis")
            if a is None:
                continue
            f = a["folding"]
            x = np.array(f["curve_state_dist"]) / f["median_state_dist"]
            ax.plot(x, f["curve_latent_frac"], color=color[m], label=m)
            ax.axvline(f["d80_rel"], color=color[m], lw=1, ls=":")
        ax.axhline(0.8, color=INK_2, lw=1, ls="--")
        ax.set_xlabel("state distance / median pair distance")
        ax.set_ylabel("latent distance / plateau")
        ax.set_title(f"{task}: saturation (dotted = d80)")
    save(fig, "saturation.png", axes)

    # folding <-> error correlations: across models (left) and within each model (right), per task
    fig, axes = panels(2 * len(tasks))
    styles = {"d80_rel": ("o-", INK, 2.0), "participation_ratio": ("s-", INK_2, 1.2),
              "folding_1_minus_trust": ("^-", "#9a9893", 1.2)}
    for t, task in enumerate(tasks):
        across, within = axes[2 * t], axes[2 * t + 1]
        corr = data[task].get("_corr", {})
        for name in FOLDING_NUMBERS:
            if name in corr:
                fmt, c, lw = styles[name]
                rows = corr[name]
                across.plot([r["h"] for r in rows], [r["rho"] for r in rows], fmt, ms=5, color=c, lw=lw, label=name)
        for m in models:
            a = data[task].get(m, {}).get("analysis")
            if a is not None:
                rows = a["stretch"]["within_rho_vs_std_err"]
                within.plot([r["h"] for r in rows], [r["rho"] for r in rows], "-", color=color[m], label=m)
        n = len(corr.get("models", []))
        for ax, title in ((across, f"{task}: folding number vs error, across models (n={n})"),
                          (within, f"{task}: local stretch vs error, within each model")):
            ax.axhline(0, color=INK_2, lw=1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("horizon h")
            ax.set_ylabel("Spearman rho with probe error at h")
            ax.set_title(title, fontsize=10)
    save(fig, "folding_rho.png", axes, ncol=9)


# --------------------------------------------------------------------------- #
#  Tables
# --------------------------------------------------------------------------- #
def markdown(data: dict, models: list[str]) -> str:
    lines = ["# Regulariser comparison", ""]
    for task, td in data.items():
        if not has_models(td):
            continue
        lines += [f"## {task}", "",
                  "| model | CEM success % (mean ± std) | d80_rel | PR | 1 − trust | 1 − cont | std_err h=1 | "
                  "std_err h=H | off-manifold h=H |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            e = td.get(m, {})
            c, a = e.get("cem"), e.get("analysis")
            cem = f"{c['success_rate_mean']:.1f} ± {c['success_rate_std']:.1f} (n={len(c['success_rates'])})" if c else "-"
            if a:
                f, ph = a["folding"], a["per_h"]
                row = (f"{f['d80_rel']:.3f} | {f['participation_ratio']:.1f} | {f['folding_1_minus_trust']:.4f} | "
                       f"{1 - f['continuity']:.4f} | {ph['std_err'][0]:.3f} | {ph['std_err'][-1]:.3f} | "
                       f"{ph['offmanifold'][-1]:.2f}")
            else:
                row = " | ".join("-" * 1 for _ in range(7))
            lines.append(f"| {m} | {cem} | {row} |")
        corr = td.get("_corr", {})
        if corr:
            H = len(corr["d80_rel"])
            lines += ["", f"Across models (n = {len(corr['models'])}), Spearman rho(folding number, std_err at h), "
                      "exact permutation p:", "",
                      "| h | " + " | ".join(FOLDING_NUMBERS) + " |", "|---|" + "---|" * len(FOLDING_NUMBERS)]
            for h in range(H):
                cells = [f"{corr[n][h]['rho']:+.2f} (p={corr[n][h]['p_exact']:.2f})" for n in FOLDING_NUMBERS]
                lines.append(f"| {h + 1} | " + " | ".join(cells) + " |")
            vs = [n for n in FOLDING_NUMBERS if f"{n}_vs_cem" in corr]
            if vs:
                lines += ["", "Folding number vs CEM success, across models: " + ", ".join(
                    f"{n} rho {corr[n + '_vs_cem']['rho']:+.2f} (p={corr[n + '_vs_cem']['p_exact']:.2f})" for n in vs)]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["tworoom", "pusht"])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--subdir", default="", help="read <root>/<task>/<model>/<subdir>/ (keep: an earlier epoch)")
    p.add_argument("--out", type=Path, default=None, help="default: <root>/report")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="CEM seeds to include (default: all run)")
    args = p.parse_args(argv)

    data = load(args.root, args.tasks, args.models, args.subdir, args.seeds)
    for task in data:
        data[task]["_corr"] = correlations({m: e for m, e in data[task].items() if not m.startswith("_")})
    out = args.out or args.root / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary = {t: {m: {k: v for k, v in e.items()} for m, e in td.items()} for t, td in data.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "summary.md").write_text(markdown(data, args.models))
    figures(data, args.models, out)
    print(markdown(data, args.models))
    print(f"report in {out}")


if __name__ == "__main__":
    main()
