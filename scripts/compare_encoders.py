"""Offline comparison of dynamics models on a saved dataset (no simulation): encoders and sizes.

Every (encoder, size) combination is trained from scratch with learned latents on the same
dataset and validated on the same val split, with a long early-stopping budget. Logs to
TensorBoard under <out>/<encoder>_<size>; each model is saved to <out>/<encoder>_<size>/model.pt.

Sizes are FEATxHIDDENxLAYERS: point feature width, MLP head hidden width, head layers
(the default model is 128x256x3).

    python scripts/compare_encoders.py --dataset runs/dynamics1/dataset.pt --out runs/arch_tests_long
    python scripts/compare_encoders.py --task push_t --dataset runs/push_t0/dataset.pt --out runs/push_t_sizes \\
        --encoders cross_attn --sizes 128x256x3 256x512x3 512x1024x4
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import train_dynamics as T  # noqa: E402
from dynamics_model.data import TransitionBuffer  # noqa: E402
from dynamics_model.model import DynamicsModel, LatentTable, true_flow_magnitude  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402


@torch.no_grad()
def rotation_error_deg(model, latents, t, idx):
    """Angle between predicted and true rotation per transition (degrees)."""
    model.eval()
    out = []
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        R, _ = model(t["pts_in"][b], t["act_feat"][b], latents(t["setting"][b]))
        cos = ((R.transpose(-1, -2) @ t["R_gt"][b]).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        out.append(torch.rad2deg(torch.acos(cos.clamp(-1, 1))))
    model.train()
    return torch.cat(out)


@torch.no_grad()
def true_rotation_deg(t, idx):
    cos = (t["R_gt"][idx].diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="runs/dynamics1/dataset.pt")
    ap.add_argument("--out", default="runs/arch_tests_long")
    ap.add_argument("--task", default="push_block", choices=T.TASKS, help="sets the action (force) normalization")
    ap.add_argument("--encoders", nargs="+", default=["pointnet", "cross_attn", "transformer"])
    ap.add_argument("--sizes", nargs="+", default=["128x256x3"], help="FEATxHIDDENxLAYERS, e.g. 256x512x3")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--patience", type=int, default=30, help="val checks (x100 steps) without improvement")
    args = ap.parse_args()

    buf = TransitionBuffer.load(args.dataset)
    dev = "cuda"
    t = buf.tensors(dev)
    tr, va = buf.split_indices(dev)
    n_settings = int(t["setting"].max()) + 1
    mag = true_flow_magnitude(t["pts_loss"][va], t["R_gt"][va], t["t_gt"][va])
    rot_true = true_rotation_deg(t, va)
    turned = rot_true > 5  # pushes that turned the object by more than 5 degrees
    action_scale = tuple(T.task_spec(args.task).action_scale)
    bins = [(0.005, 0.05), (0.05, 0.15), (0.15, 0.3)]
    print(f"dataset {len(buf)} (train {len(tr)}, val {len(va)}), task {args.task}; "
          f"{turned.float().mean():.0%} of val pushes turn the object > 5 deg", flush=True)
    print(f"{'model':24s} {'params':>8s} {'steps':>6s} {'train-set':>9s} {'val':>6s}" +
          "".join(f"{f'{100*lo:.1f}-{100*hi:.0f}cm':>16s}" for lo, hi in bins) +
          f"{'rot err':>9s}{'rot err >5deg':>14s}{'minutes':>8s}", flush=True)
    for enc in args.encoders:
        for size in args.sizes:
            feat, hidden, layers = (int(x) for x in size.split("x"))
            name = f"{enc}_{size}"
            torch.manual_seed(0)
            c = T.Config(max_train_steps=args.max_steps, patience=args.patience, encoder=enc)
            kwargs = dict(latent_dim=c.latent_dim, encoder=enc, feat_dim=feat, hidden=hidden, layers=layers,
                          action_scale=action_scale)
            model = DynamicsModel(**kwargs).to(dev)
            n_params = sum(p.numel() for p in model.parameters())
            lat = LatentTable(n_settings, c.latent_dim).to(dev)
            opt = torch.optim.AdamW([{"params": model.parameters(), "lr": c.lr, "weight_decay": c.weight_decay},
                                     {"params": lat.parameters(), "lr": c.latent_lr, "weight_decay": 0.0}])
            t0 = time.time()
            r = T.train(model, lat, opt, buf, c, dev, SummaryWriter(f"{args.out}/{name}"), [0])
            err = T.evaluate(model, lat, t, va)
            tr_err = T.evaluate(model, lat, t, tr).mean().item()
            rot = rotation_error_deg(model, lat, t, va)
            cells = "".join(f"{100*err[(mag >= lo) & (mag < hi)].mean():8.2f}cm"
                            f"({(err[(mag >= lo) & (mag < hi)] / mag[(mag >= lo) & (mag < hi)]).median():4.2f})"
                            for lo, hi in bins)
            print(f"{name:24s} {n_params/1e3:7.0f}k {r['steps']:6d} {100*tr_err:8.2f}cm {100*err.mean():5.2f}{cells}"
                  f"{rot.mean():7.1f}deg{rot[turned].mean():11.1f}deg{(time.time()-t0)/60:8.1f}", flush=True)
            torch.save({"model": model.state_dict(), "latents": lat.state_dict(), "encoder": enc, "task": args.task,
                        "model_kwargs": kwargs}, f"{args.out}/{name}/model.pt")


if __name__ == "__main__":
    main()
