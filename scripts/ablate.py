"""Offline ablations on a saved dataset (no simulation): train model variants from scratch on the
same train/val split and compare their validation error.

Variants (VARIANTS below) cover the observation encoder (point cloud encoders vs. frozen DINOv2
image features), the head architecture, the latent, the loss and augmentation, and the amount
of training data. Each variant trains with train_dynamics.train (early stopping on the val
error, long budget). Results go to <out>/results.json and <out>/results.md; TensorBoard logs
to <out>/<variant>_s<seed>.

    python scripts/dino_features.py --dataset runs/block/dataset.pt          # once, for image variants
    python scripts/ablate.py --dataset runs/block/dataset.pt --out runs/block/ablation
    python scripts/ablate.py --dataset runs/block/dataset.pt --out runs/block/ablation_arch \\
        --variants pointnet pointnet_wide pointnet_res cross_attn transformer --seeds 0 1
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import train_dynamics as T  # noqa: E402
from dynamics_model.data import TransitionBuffer  # noqa: E402
from dynamics_model.model import ENCODER_IO, DynamicsModel, LatentTable, true_flow_magnitude  # noqa: E402
from env import get_task  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

# name -> (model kwargs overrides, training config overrides)
VARIANTS = {
    # observation encoder (point cloud in the action frame)
    "pointnet": ({}, {}),
    "pointnet_early": ({"encoder": "pointnet_early"}, {}),
    "cross_attn": ({"encoder": "cross_attn"}, {}),
    "transformer": ({"encoder": "transformer"}, {}),
    # observation = frozen DINOv2 features of the camera image (world frame; no mirror augmentation possible)
    "image_cls": ({"encoder": "image_cls"}, {"mirror_aug": False, "point_noise": 0.0}),
    "image_attn": ({"encoder": "image_attn"}, {"mirror_aug": False, "point_noise": 0.0}),
    # representations (dynamics_model/model.py): richer point features, object-centric points, point
    # cloud + image fusion, action-frame image crops (scripts/image_crops.py), CNNs trained from scratch
    "pointnet_geom": ({"encoder": "pointnet_geom"}, {}),
    "pointnet_objcentric": ({"encoder": "pointnet_objcentric"}, {}),
    "pointnet_dino": ({"encoder": "pointnet_dino"}, {"mirror_aug": False, "point_noise": 0.0}),
    "image_af_cls": ({"encoder": "image_af_cls"}, {"mirror_aug": False, "point_noise": 0.0}),
    "image_cnn": ({"encoder": "image_cnn"}, {"mirror_aug": False, "point_noise": 0.0}),
    "image_af_cnn": ({"encoder": "image_af_cnn"}, {"point_noise": 0.0}),
    # architecture
    "pointnet_wide": ({"feat_dim": 256, "hidden": 512, "layers": 4}, {}),
    "pointnet_res": ({"residual": True, "layers": 4}, {}),
    "pointnet_res_wide": ({"residual": True, "layers": 4, "feat_dim": 256, "hidden": 512}, {}),
    "cross_attn_res": ({"encoder": "cross_attn", "residual": True, "layers": 4}, {}),
    # latent
    "latent0": ({"latent_dim": 0}, {}),  # no physics latent at all
    "latent16": ({"latent_dim": 16}, {}),
    # loss / augmentation
    "relative_loss": ({}, {"loss": "relative"}),
    "no_mirror": ({}, {"mirror_aug": False}),
    "no_noise": ({}, {"point_noise": 0.0}),
    "lr3e-4": ({}, {"lr": 3e-4}),
    # amount of training data (same val split)
    "data12": ({}, {"data_fraction": 0.125}),
    "data25": ({}, {"data_fraction": 0.25}),
    "data50": ({}, {"data_fraction": 0.5}),
}
GROUPS = {  # every group contains the baseline, so each table has its own reference
    "encoders": ["pointnet", "pointnet_early", "cross_attn", "transformer"],
    "image_vs_pc": ["pointnet", "no_mirror", "image_cls", "image_attn"],
    "arch": ["pointnet", "pointnet_wide", "pointnet_res", "pointnet_res_wide", "cross_attn_res"],
    "latent_training": ["pointnet", "latent0", "latent16", "relative_loss", "no_noise", "lr3e-4"],
    "data": ["data12", "data25", "data50", "pointnet"],
    "big": ["data12", "data25", "data50", "pointnet", "cross_attn", "pointnet_res_wide", "image_cls", "image_attn"],
    "repr": ["pointnet", "pointnet_geom", "pointnet_objcentric", "pointnet_dino",
             "image_cls", "image_af_cls", "image_cnn", "image_af_cnn"],
}
BINS = [(0.005, 0.05), (0.05, 0.15), (0.15, 10.0)]


def summarize(err, mag):
    moved = mag > 0.005
    r = {"val_err_cm": 100 * err.mean().item(),
         "val_err_moved_cm": 100 * err[moved].mean().item() if moved.any() else float("nan"),
         "val_rel_err_moved": (err[moved] / mag[moved]).median().item() if moved.any() else float("nan")}
    for lo, hi in BINS:
        m = (mag >= lo) & (mag < hi)
        r[f"err_{100 * lo:.1f}-{100 * hi:.0f}cm"] = 100 * err[m].mean().item() if m.any() else float("nan")
    return r


def fmt_table(results, variants, seeds):
    cols = ["val_err_cm", "val_err_moved_cm", "val_rel_err_moved"] + [f"err_{100 * lo:.1f}-{100 * hi:.0f}cm" for lo, hi in BINS]
    head = ["variant", "n_train", "steps", "params"] + cols + ["min"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v in variants:
        rows = [results[f"{v}_s{s}"] for s in seeds if f"{v}_s{s}" in results]
        if not rows:
            continue
        cells = [v, str(rows[0]["n_train"]), str(int(np.mean([r["steps"] for r in rows]))), f"{rows[0]['params'] / 1e3:.0f}k"]
        for c in cols:
            vals = np.array([r[c] for r in rows], dtype=float)
            cells.append(f"{vals.mean():.2f}" + (f" ± {vals.std():.2f}" if len(vals) > 1 else ""))
        cells.append(f"{np.mean([r['minutes'] for r in rows]):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=None, help=f"default: all. Names: {list(VARIANTS)}")
    ap.add_argument("--group", choices=list(GROUPS), default=None, help="a predefined set of variants")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--patience", type=int, default=30, help="val checks (x100 steps) without improvement")
    ap.add_argument("--features", default=None, help="DINOv2 tokens (default: <dataset dir>/dino_feats.pt)")
    ap.add_argument("--af-features", default=None, help="DINOv2 tokens of the action-frame crops (default: <dataset dir>/dino_af_feats.pt)")
    ap.add_argument("--images", default=None, help="image crops file (default: <dataset dir>/image_af.pt)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    variants = args.variants or (GROUPS[args.group] if args.group else list(VARIANTS))
    unknown = set(variants) - set(VARIANTS)
    assert not unknown, f"unknown variants {unknown}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    buf = TransitionBuffer.load(args.dataset)
    task = get_task(buf.meta.get("task", "block"))
    # extra inputs the chosen variants need (dynamics_model.model.ENCODER_IO), attached to the buffer
    need = set()
    for v in variants:
        key, aux, _ = ENCODER_IO[VARIANTS[v][0].get("encoder", "pointnet")]
        need |= {key, aux} - {None, "pts_in"}
    ddir = Path(args.dataset).parent
    if "img_feat" in need:
        feats = Path(args.features) if args.features else ddir / "dino_feats.pt"
        assert feats.exists(), f"{feats} missing: run scripts/dino_features.py --dataset {args.dataset}"
        buf.attach("img_feat", torch.load(feats)["tokens"])
    if "img_af_feat" in need:
        feats = Path(args.af_features) if args.af_features else ddir / "dino_af_feats.pt"
        assert feats.exists(), f"{feats} missing: run scripts/image_crops.py then scripts/dino_features.py --images"
        buf.attach("img_af_feat", torch.load(feats)["tokens"])
    if need & {"image_96", "image_af_96"}:
        imgs = Path(args.images) if args.images else ddir / "image_af.pt"
        assert imgs.exists(), f"{imgs} missing: run scripts/image_crops.py --dataset {args.dataset}"
        d = torch.load(imgs)
        buf.attach("image_96", d["world96"])
        buf.attach("image_af_96", d["af96"])
    t = buf.tensors(dev)
    tr, va = buf.split_indices(dev)
    n_settings = int(t["setting"].max()) + 1
    mag = true_flow_magnitude(t["pts_loss"][va], t["R_gt"][va], t["t_gt"][va])
    saved_cfg = {k: tuple(v) if isinstance(v, list) else v for k, v in buf.meta.get("config", {}).items()
                 if k in T.Config.__dataclass_fields__}
    base = T.Config(**saved_cfg) if saved_cfg else T.Config(task=task.name)
    print(f"dataset {args.dataset}: {len(buf)} transitions (train {len(tr)}, val {len(va)}), task {task.name}, "
          f"{n_settings} settings; variants {variants}, seeds {args.seeds}", flush=True)

    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for v in variants:
        mk, tk = VARIANTS[v]
        for seed in args.seeds:
            key = f"{v}_s{seed}"
            if key in results:
                print(f"{key}: done already", flush=True)
                continue
            torch.manual_seed(seed)
            cfg = replace(base, max_train_steps=args.max_steps, patience=args.patience, seed=seed, figure_every=0, **tk)
            kw = {**T.model_kwargs(cfg, task), **mk}
            model = DynamicsModel(**kw).to(dev)
            lat = LatentTable(n_settings, kw["latent_dim"]).to(dev)
            opt = torch.optim.AdamW([{"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
                                     {"params": lat.parameters(), "lr": cfg.latent_lr, "weight_decay": 0.0}])
            t0 = time.time()
            r = T.train(model, lat, opt, buf, cfg, dev, SummaryWriter(str(out / key)), [0])
            err = T.evaluate(model, lat, t, va)
            r.update(summarize(err, mag), minutes=(time.time() - t0) / 60, variant=v, seed=seed, model_kwargs=kw,
                     params=sum(p.numel() for p in model.parameters()), train_overrides=tk)
            results[key] = r
            torch.save({"model": model.state_dict(), "latents": lat.state_dict(), "model_kwargs": kw, "task": task.name,
                        "encoder": kw["encoder"], "config": {**saved_cfg, **tk}}, out / key / "model.pt")
            results_path.write_text(json.dumps(results, indent=2))
            print(f"{key:24s} steps {r['steps']:6d}  val {r['val_err_cm']:.2f} cm  moved {r['val_err_moved_cm']:.2f} cm "
                  f"rel {r['val_rel_err_moved']:.2f}  ({r['minutes']:.1f} min)", flush=True)
    seeds = sorted({r["seed"] for r in results.values()})
    all_variants = [v for v in VARIANTS if any(k.startswith(v + "_s") for k in results)]
    md = (f"# Ablation on {args.dataset}\n\ntask {task.name}, {len(buf)} transitions (train {len(tr)}, val {len(va)}), "
          f"no-motion baseline {100 * mag.mean():.2f} cm (moved: {100 * mag[mag > 0.005].mean():.2f} cm), seeds {seeds}\n\n"
          "Errors in cm (mean point-flow error on the val split); `moved` = pushes that moved the object > 5 mm; "
          "`rel` = median error / distance moved; the last columns bin by distance moved.\n\n"
          + fmt_table(results, all_variants, seeds) + "\n")
    (out / "results.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
