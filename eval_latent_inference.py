"""Does test-time latent inference work? Evaluate on physics settings never seen in training.

For each new environment: collect pushes (by default informative probes aimed through the
observed block centroid, see policy.mpc.ProbeSampler; --probe random uses the random policy),
fit a latent (dynamics model frozen) on the first k pushes, and measure the prediction error on
the last `heldout` pushes.

Compared against:
  mean latent   mean of the training latents (no adaptation)
  oracle        latent fitted on all of that env's pushes, held-out ones included (best case)
If the error drops well below "mean latent" within a few pushes, the latent captures physics
that transfers to new environments.

Run from the repo root (any time after the dynamics run has written a checkpoint):
    python eval_latent_inference.py --checkpoint runs/dynamics1/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dynamics_model.data import TransitionBuffer
from dynamics_model.latent_inference import infer_latents
from dynamics_model.model import DynamicsModel, LatentTable, point_flow_loss
from env import PushPrimitiveConfig, make_push_env
from policy.mpc import ActionSampler, ProbeSampler
from train_dynamics import TEST_SEED, Config, make_physics_pool, run_rollout

warnings.filterwarnings("ignore")


def errors(model, data, idx, z):
    """Per-transition error (m) and how far the block moved (m)."""
    with torch.no_grad():
        R, t = model(data["pts_in"][idx], data["act_feat"][idx], z)
        err = point_flow_loss(data["pts_loss"][idx], R, t, data["R_gt"][idx], data["t_gt"][idx])
        eye = torch.eye(3, device=R.device).expand(len(idx), 3, 3)
        moved = point_flow_loss(data["pts_loss"][idx], eye, torch.zeros_like(t), data["R_gt"][idx], data["t_gt"][idx])
    return err, moved


def summarize(err, moved):
    m = moved > 0.005
    return {
        "err_cm": 100 * err.mean().item(),
        "err_moved_cm": 100 * err[m].mean().item() if m.any() else float("nan"),
        "rel_err_moved": (err[m] / moved[m]).median().item() if m.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/dynamics1/checkpoint.pt")
    ap.add_argument("--num-envs", type=int, default=32, help="new environments (one physics setting each)")
    ap.add_argument("--pushes", type=int, default=16, help="pushes collected per environment")
    ap.add_argument("--heldout", type=int, default=6, help="last pushes of each env used for evaluation")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8], help="numbers of pushes to fit on")
    ap.add_argument("--seed", type=int, default=TEST_SEED, help="seed of the held-out test settings (training never uses it)")
    ap.add_argument("--fit-steps", type=int, default=200)
    ap.add_argument("--probe", choices=["centroid", "random"], default="centroid",
                    help="pushes used to probe the new envs: aimed through the block centroid, or random")
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/latent_inference.json")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint)
    known = Config.__dataclass_fields__
    saved = {k: tuple(v) if isinstance(v, list) else v for k, v in ckpt["config"].items() if k in known}
    cfg = replace(Config(**saved),
                  num_envs=args.num_envs, pool_size=args.num_envs, seed=args.seed, pushes_per_rollout=args.pushes)
    pool, pool_t = make_physics_pool(cfg)  # same ranges as training, different seed -> unseen settings

    env = make_push_env(num_envs=cfg.num_envs, obs_mode="pointcloud", push_config=PushPrimitiveConfig(phase_timeout=2.0))
    dev = env.unwrapped.device
    model = DynamicsModel.from_checkpoint(ckpt, dev)
    train_latents = LatentTable(ckpt["latents"]["emb.weight"].shape[0], cfg.latent_dim).to(dev)
    train_latents.load_state_dict(ckpt["latents"])
    z_mean = train_latents.emb.weight.mean(0)

    print(f"collecting {args.pushes} pushes in each of {cfg.num_envs} new environments ...", flush=True)
    buf = TransitionBuffer(val_fraction=0.0, seed=args.seed)
    sampler = ProbeSampler() if args.probe == "centroid" else ActionSampler(uniform_fraction=cfg.random_uniform_fraction)
    stats = run_rollout(env, cfg, pool, pool_t, np.arange(cfg.num_envs), lambda c, i, g: sampler(c, 1)[:, 0], buf, 0)
    env.close()
    data = buf.tensors(dev)
    env_of = data["setting"]
    print(f"kept {len(env_of)} transitions (contact rate {stats['contact_rate']:.2f})", flush=True)

    # per env: transitions in the order they happened; last `heldout` for evaluation
    fit_pos, held = [], []
    for e in range(cfg.num_envs):
        idx = (env_of == e).nonzero().flatten()
        if len(idx) <= args.heldout:
            continue
        held.append(idx[-args.heldout:])
        fit_pos.append(idx[:-args.heldout])
    envs = len(held)
    held_idx = torch.cat(held)
    held_env = torch.cat([torch.full((len(h),), k, device=dev) for k, h in enumerate(held)])
    print(f"{envs} envs with enough pushes; {len(held_idx)} held-out pushes, "
          f"{(errors(model, data, held_idx, z_mean.expand(len(held_idx), -1))[1] > 0.005).float().mean():.0%} of them moved the block\n")

    def fit(idx_lists, steps=args.fit_steps):
        idx = torch.cat(idx_lists)
        grp = torch.cat([torch.full((len(x),), k, device=dev) for k, x in enumerate(idx_lists)])
        sub = {k: data[k][idx] for k in ["pts_in", "act_feat", "pts_loss", "R_gt", "t_gt"]}
        floor = cfg.relative_floor if cfg.loss == "relative" else None  # same objective as training
        return infer_latents(model, sub, grp, len(idx_lists), z_mean, steps=steps, lr=cfg.latent_lr, l2=cfg.latent_l2,
                             relative_floor=floor)

    # For each k: the envs that have at least k fit pushes. Every row of a k-block is computed on
    # the same envs' held-out pushes, so the rows within it are directly comparable.
    z_oracle = fit([torch.cat([f, h]) for f, h in zip(fit_pos, held)])
    results = {}
    for k in args.ks:
        envs_k = [e for e in range(envs) if len(fit_pos[e]) >= k]
        if not envs_k:
            print(f"(no env has {k} fit pushes)")
            continue
        sel = torch.isin(held_env, torch.tensor(envs_k, device=dev))
        h_idx, h_env = held_idx[sel], held_env[sel]
        z_k = fit([fit_pos[e][:k] for e in envs_k])
        remap = torch.full((envs,), -1, dtype=torch.long, device=dev)
        remap[torch.tensor(envs_k, device=dev)] = torch.arange(len(envs_k), device=dev)
        results[k] = {
            "envs": len(envs_k),
            "mean latent (no adaptation)": summarize(*errors(model, data, h_idx, z_mean.expand(len(h_idx), -1))),
            f"fitted on {k} push{'es' if k > 1 else ''}": summarize(*errors(model, data, h_idx, z_k[remap[h_env]])),
            "oracle (fitted incl. held-out)": summarize(*errors(model, data, h_idx, z_oracle[h_env])),
        }

    print(f"{'latent':34s} {'err (cm)':>9s} {'err moved (cm)':>15s} {'rel err moved':>14s}")
    for k, block in results.items():
        print(f"-- k = {k} ({block['envs']} envs)")
        for name, r in block.items():
            if name != "envs":
                print(f"   {name:31s} {r['err_cm']:9.2f} {r['err_moved_cm']:15.2f} {r['rel_err_moved']:14.2f}")
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "latent_inference.json"
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
