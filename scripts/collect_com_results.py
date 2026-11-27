"""Collect the off-centre COM results (scripts/com_experiments.sh) into markdown tables.

    modal volume get latent-physics-runs <file> runs/<file>    # or run on the volume
    python scripts/collect_com_results.py --runs runs > results_tables.md
"""

import argparse
import json
from pathlib import Path


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, f="{:.2f}"):
    return "-" if x is None or x != x else f.format(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    R = Path(args.runs)
    out = []

    # 1. control + prediction: mean / fitted / fitted_N, uniform vs COM
    out += ["## 1. MPC on unseen settings (64 envs): mean vs fitted latent", "",
            "| model | test blocks | latent | success (2 cm) | success (5 cm) | final dist (cm) | MPC pred err (cm) |",
            "|---|---|---|---|---|---|---|"]
    rows = [("uniform_nomirror", "uniform", R / "uniform_nomirror/mpc_eval.json"),
            ("uniform_nomirror", "COM 1.5 cm", R / "uniform_nomirror_on_com015/mpc_eval.json"),
            ("com015", "COM 1.5 cm", R / "com015/mpc_eval.json"),
            ("com015 ensemble (5)", "COM 1.5 cm", R / "com015_ensemble/mpc_eval.json")]
    for model, blocks, p in rows:
        d = load(p)
        if not d:
            continue
        for c, r in d["results"].items():
            out.append(f"| {model} | {blocks} | {c} | {r['success']:.0%} | {r['success_5cm']:.0%} | "
                       f"{fmt(r['final_dist_cm_mean'], '{:.1f}')} | {fmt(r['prediction_err_cm'])} |")

    out += ["", "## 1b. One-push prediction error on held-out pushes (eval_latent_inference, k = fitting pushes)", "",
            "| model | k | latent | err (cm) | err moved (cm) | rot err (deg) | rot err moved (deg) |",
            "|---|---|---|---|---|---|---|"]
    for run in ["uniform_nomirror", "com015"]:
        d = load(R / run / "latent_inference.json")
        if not d:
            continue
        for k, block in d["results"].items():
            for name, r in block.items():
                if name == "envs":
                    continue
                out.append(f"| {run} | {k} | {name} | {fmt(r['err_cm'])} | {fmt(r['err_moved_cm'])} | "
                           f"{fmt(r.get('rot_err_deg'))} | {fmt(r.get('rot_err_moved_deg'))} |")

    # 2. encoders + ensemble on the COM dataset
    out += ["", "## 2. Encoders and ensemble on the COM dataset (val split, offline)", "",
            "| model | val err (cm) | val err moved (cm) | val rot err (deg) | val rot err moved (deg) |",
            "|---|---|---|---|---|"]
    enc = load(R / "com015_encoders/results.json") or {}
    for name, r in enc.items():
        out.append(f"| {name} | {fmt(r['val_err_cm'])} | {fmt(r['val_err_moved_cm'])} | "
                   f"{fmt(r['val_rot_err_deg'])} | {fmt(r['val_rot_err_moved_deg'])} |")
    ens = load(R / "com015_ensemble/results.json") or {}
    for name in [k for k in ens if k == "member0" or k.startswith("ensemble_of_")]:
        r = ens[name]
        out.append(f"| cross_attn {name.replace('_', ' ')} | {fmt(r['val_err_cm'])} | {fmt(r['val_err_moved_cm'])} | "
                   f"{fmt(r.get('val_rot_err_deg'))} | {fmt(r.get('val_rot_err_moved_deg'))} |")

    # 3. latent check
    out += ["", "## 3. Latent check: leave-one-out linear R^2 from fitted latent to true physics", "",
            "| model | log mass | mu_s | mu_d | com_x | com_y | |com| |", "|---|---|---|---|---|---|---|"]
    for run in ["uniform_nomirror", "com015"]:
        d = load(R / run / "latent_inference.json")
        if d and "latent_r2" in d:
            r = d["latent_r2"]
            out.append(f"| {run} | " + " | ".join(fmt(r.get(k), "{:.3f}") for k in
                       ["log_mass", "static_friction", "dynamic_friction", "com_x", "com_y", "com_radius"]) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
