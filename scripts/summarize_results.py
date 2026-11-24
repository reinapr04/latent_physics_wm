"""Collect the results under runs/ into Markdown tables (for the README).

Reads, for every run directory: metrics.jsonl (data collected, final validation error),
latent_inference.json (eval_latent_inference.py), mpc_eval.json (eval_mpc.py) and
ablation*/results.json (scripts/ablate.py), including those of ablation variants.

    python scripts/summarize_results.py --runs runs > results.md
"""

import argparse
import json
from pathlib import Path


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_summary(run: Path):
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    # a run restarted from scratch in the same directory (e.g. requeued by SLURM) appends a second
    # record sequence: keep the last one (it starts with random rollout 0)
    starts = [i for i, l in enumerate(lines) if l["phase"] == "random" and l.get("rollout") == 0]
    if starts:
        lines = lines[starts[-1]:]
    rollouts = [l for l in lines if l["phase"] in ("random", "mpc")]
    trains = [l for l in lines if "train" in l["phase"]]
    cfg = load_json(run / "config.json") or {}
    last = trains[-1] if trains else {}
    mpc = [l for l in rollouts if l["phase"] == "mpc"]
    return {
        "run": run.name, "task": cfg.get("task", "block"), "encoder": cfg.get("encoder", "pointnet"),
        "settings": cfg.get("pool_size"), "rollouts": len(rollouts), "transitions": sum(l["samples"] for l in rollouts),
        "val_err_cm": last.get("val_err_cm"), "val_err_moved_cm": last.get("val_err_moved_cm"),
        "val_rel_err_moved": last.get("val_rel_err_moved"), "no_motion_cm": last.get("val_no_motion_baseline_cm"),
        "mpc_rollout_success": sum(l["success"] for l in mpc) / len(mpc) if mpc else None,
        "contact_rate": sum(l["contact_rate"] for l in rollouts) / len(rollouts) if rollouts else None,
        "hours": lines[-1]["time"] / 3600 if lines else None,
    }


def f(x, d=2):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    root = Path(args.runs)
    runs = sorted(p for p in root.iterdir() if (p / "metrics.jsonl").exists() and not p.name.startswith("smoke"))
    out = []

    out.append("### Training runs\n")
    out.append("| run | task | encoder | settings | rollouts | transitions | val err (cm) | val err moved (cm) | rel err moved | no-motion baseline (cm) | MPC rollout success | hours |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        s = run_summary(r)
        out.append(f"| {s['run']} | {s['task']} | {s['encoder']} | {s['settings']} | {s['rollouts']} | {s['transitions']} | "
                   f"{f(s['val_err_cm'])} | {f(s['val_err_moved_cm'])} | {f(s['val_rel_err_moved'])} | {f(s['no_motion_cm'])} | "
                   f"{pct(s['mpc_rollout_success'])} | {f(s['hours'], 1)} |")

    li = sorted(p for p in root.glob("**/latent_inference.json") if "smoke" not in str(p))
    if li:
        out.append("\n### Test-time latent inference (unseen physics)\n")
        out.append("Prediction error (cm) on held-out pushes of new environments; `moved` = pushes that moved the object > 5 mm.\n")
        out.append("| checkpoint | k probes | envs | mean latent | fitted on k | oracle | mean (moved) | fitted (moved) | oracle (moved) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for p in li:
            d = load_json(p)
            if not d:
                continue
            name = str(p.parent.relative_to(root))
            for k, b in d["results"].items():
                m, fk, o = b["mean latent (no adaptation)"], [v for kk, v in b.items() if kk.startswith("fitted")][0], b["oracle (fitted incl. held-out)"]
                out.append(f"| {name} | {k} | {b['envs']} | {m['err_cm']:.2f} | {fk['err_cm']:.2f} | {o['err_cm']:.2f} | "
                           f"{m['err_moved_cm']:.2f} | {fk['err_moved_cm']:.2f} | {o['err_moved_cm']:.2f} |")

    me = sorted(p for p in root.glob("**/mpc_eval*.json") if "smoke" not in str(p))
    if me:
        out.append("\n### Goal reaching with MPC (unseen physics)\n")
        out.append("`hard` = goal 15-30 cm away and only 2 pushes.\n")
        out.append("| checkpoint | task | envs | start→goal (cm) | pushes | latent | success | < 5 cm | final dist mean / median (cm) | pushes to success | off table | pred err (cm) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in me:
            d = load_json(p)
            if not d:
                continue
            name = str(p.parent.relative_to(root)) + (" (hard)" if "hard" in p.name else "")
            a = d["args"]
            for c, r in d["results"].items():
                out.append(f"| {name} | {d.get('task', 'block')} | {a['num_envs']} | {d['start_to_goal_cm']:.1f} | {a['max_pushes']} | {c} | "
                           f"{pct(r['success'])} | {pct(r['success_5cm'])} | {r['final_dist_cm_mean']:.1f} / {r['final_dist_cm_median']:.1f} | "
                           f"{f(r['pushes_to_success'], 1)} | {pct(r['off_table'])} | {r['prediction_err_cm']:.2f} |")

    ab = sorted(p for p in root.glob("**/results.md") if "smoke" not in str(p))
    for p in ab:
        out.append(f"\n### {p.parent.relative_to(root)}\n")
        out.append(p.read_text().split("\n", 2)[2].strip())
    print("\n".join(out))


if __name__ == "__main__":
    main()
