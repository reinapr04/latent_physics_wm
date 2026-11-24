"""Figures for the README from the result files under runs/ (skips what is missing):

  docs/data_scaling.png      val error vs. training-set size (scripts/ablate.py data12/25/50 + full)
  docs/latent_inference.png  prediction error on unseen physics vs. number of probes (eval_latent_inference.py)
  docs/mpc_success.png       goal-reaching success with mean / fitted latents (eval_mpc.py)
  docs/ablation_<run>.png    val error of every ablation variant (scripts/ablate.py)

    python scripts/plot_results.py --runs runs --out docs
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference palette (validated): categorical slots 1-3, light surface, text inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def by_variant(results):
    out = {}
    for r in results.values():
        out.setdefault(r["variant"], []).append(r)
    return out


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.plot([x], [y], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)


def data_scaling(runs, out):
    """val error vs. #training transitions, for every ablation dir that has the data variants."""
    panels = []
    for res in sorted(runs.glob("*/ablation*/results.json")):
        bv = by_variant(load(res))
        pts = [(np.mean([r["n_train"] for r in rs]), rs) for v, rs in bv.items() if v in ("data12", "data25", "data50", "pointnet")]
        if len(pts) >= 3:
            panels.append((res.parent.parent.name, sorted(pts)))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4), squeeze=False)
    for ax, (name, pts) in zip(axes[0], panels):
        x = [p[0] for p in pts]
        for k, (key, lab) in enumerate([("val_err_cm", "all pushes"), ("val_err_moved_cm", "object moved > 5 mm")]):
            y = np.array([[r[key] for r in rs] for _, rs in pts], dtype=float)
            m, s = y.mean(1), y.std(1)
            ax.plot(x, m, color=SERIES[k], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
            if (s > 0).any():
                ax.fill_between(x, m - s, m + s, color=SERIES[k], alpha=0.15, linewidth=0)
            ax.annotate(f"{m[-1]:.2f}", (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v):,}" for v in x])
        ax.minorticks_off()
        ax.set_xlabel("training transitions")
        ax.set_ylabel("val point-flow error (cm)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{name}: more data")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "data_scaling.png", dpi=160)
    plt.close(fig)


def latent_inference(runs, out):
    files = sorted(p for p in runs.glob("**/latent_inference.json") if "smoke" not in str(p))
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 3.4), squeeze=False)
    for ax, f in zip(axes[0], files):
        d = load(f)["results"]
        ks = sorted(int(k) for k in d)
        key = "err_moved_cm"
        fitted = [[v for kk, v in d[str(k)].items() if kk.startswith("fitted")][0][key] for k in ks]
        mean = [d[str(k)]["mean latent (no adaptation)"][key] for k in ks]
        oracle = [d[str(k)]["oracle (fitted incl. held-out)"][key] for k in ks]
        ax.plot(ks, mean, color=SERIES[1], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label="mean latent (no adaptation)")
        ax.plot(ks, fitted, color=SERIES[0], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label="latent fitted on k probes")
        ax.plot(ks, oracle, color=SERIES[2], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5, label="oracle (fitted incl. held-out)")
        for y, c in [(mean, SERIES[1]), (fitted, SERIES[0]), (oracle, SERIES[2])]:
            ax.annotate(f"{y[-1]:.2f}", (ks[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("probing pushes used to fit the latent (k)")
        ax.set_ylabel("error on held-out pushes that moved (cm)")
        ax.set_ylim(bottom=0)
        ax.set_title(str(f.parent.relative_to(runs)))
    axes[0][0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "latent_inference.png", dpi=160)
    plt.close(fig)


def mpc_success(runs, out):
    """Two figures: the main runs / ensembles (mpc_success.png) and the ablation models
    (mpc_ablations.png), standard setting only (the hard setting is in docs/results.md)."""
    files = sorted(p for p in runs.glob("**/mpc_eval.json") if "smoke" not in str(p))
    main = [f for f in files if "ablation" not in str(f)]
    abl = [f for f in files if "ablation" in str(f)]
    _mpc_bars(runs, main, out / "mpc_success.png")
    _mpc_bars(runs, abl, out / "mpc_ablations.png", short=True)


def _mpc_bars(runs, files, path, short=False):
    if not files:
        return
    names = [str(f.parent.relative_to(runs)) for f in files]
    if short:
        names = [n.split("/")[0] + "\n" + n.split("/")[-1].replace("_s0", "") for n in names]
    data = [load(f) for f in files]
    conds = ["mean", "fitted", "fitted_N"]
    labels = {"mean": "mean latent", "fitted": "fitted (4 probes)", "fitted_N": "fitted (16 probes)"}
    fig, axes = plt.subplots(1, 2, figsize=(max(7, 1.6 * len(files) + 3), 3.8))
    for ax, (key, title) in zip(axes, [("success", "success: within 2 cm"), ("success_5cm", "within 5 cm")]):
        w = 0.26
        for j, c in enumerate(conds):
            ys = [100 * d["results"].get(c, {}).get(key, np.nan) for d in data]
            xs = np.arange(len(files)) + (j - 1) * w
            ax.bar(xs, ys, width=w - 0.03, color=SERIES[j], label=labels[c], linewidth=0)
            for x, y in zip(xs, ys):
                if np.isfinite(y):
                    ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(np.arange(len(files)))
        ax.set_xticklabels([n.replace("/ablation_", "/\n").replace("/", "/\n") for n in names], fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% of unseen environments")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ablations(runs, out):
    for res in sorted(runs.glob("*/ablation*/results.json")):
        if "smoke" in str(res):
            continue
        bv = by_variant(load(res))
        rows = sorted(((np.mean([r["val_err_moved_cm"] for r in rs]), np.std([r["val_err_moved_cm"] for r in rs]),
                        np.mean([r["val_err_cm"] for r in rs]), v) for v, rs in bv.items()), reverse=True)
        fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(rows) + 1.4))
        y = np.arange(len(rows))
        ax.barh(y, [r[0] for r in rows], xerr=[r[1] for r in rows], color=SERIES[0], height=0.62, linewidth=0,
                error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
        for yi, r in zip(y, rows):
            ax.annotate(f"{r[0]:.2f}  (all pushes {r[2]:.2f})", (r[0], yi), xytext=(14, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([r[3] for r in rows])
        ax.set_xlabel("val error on pushes that moved the object (cm)")
        ax.set_xlim(0, max(r[0] + r[1] for r in rows) * 1.35)
        ax.grid(axis="y", visible=False)
        group = res.parent.name.replace("ablation", "").strip("_").replace("_", " ")
        name = res.parent.parent.name + ("_" + group.replace(" ", "_") if group else "")
        ax.set_title(f"{res.parent.parent.name}: {group or 'ablation'} (lower is better)")
        fig.tight_layout()
        fig.savefig(out / f"ablation_{name}.png", dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_scaling(runs, out)
    latent_inference(runs, out)
    mpc_success(runs, out)
    ablations(runs, out)
    print("figures:", sorted(str(p) for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
