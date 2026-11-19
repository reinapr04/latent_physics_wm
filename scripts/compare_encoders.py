"""Offline comparison of dynamics-model encoders on a saved dataset (no simulation).

Each encoder is trained from scratch with learned latents on the same dataset and validated on
the same val split, with a long early-stopping budget. Logs to TensorBoard under <out>/<encoder>.

    python scripts/compare_encoders.py --dataset runs/dynamics1/dataset.pt --out runs/arch_tests_long
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="runs/dynamics1/dataset.pt")
    ap.add_argument("--out", default="runs/arch_tests_long")
    ap.add_argument("--encoders", nargs="+", default=["pointnet", "cross_attn", "transformer"])
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--patience", type=int, default=30, help="val checks (x100 steps) without improvement")
    args = ap.parse_args()

    buf = TransitionBuffer.load(args.dataset)
    dev = "cuda"
    t = buf.tensors(dev)
    tr, va = buf.split_indices(dev)
    n_settings = int(t["setting"].max()) + 1
    mag = true_flow_magnitude(t["pts_loss"][va], t["R_gt"][va], t["t_gt"][va])
    bins = [(0.005, 0.05), (0.05, 0.15), (0.15, 0.3)]
    print(f"dataset {len(buf)} (train {len(tr)}, val {len(va)})", flush=True)
    print(f"{'encoder':12s} {'steps':>6s} {'train-set':>9s} {'val':>6s}" +
          "".join(f"{f'{100*lo:.1f}-{100*hi:.0f}cm':>16s}" for lo, hi in bins) + f"{'minutes':>8s}", flush=True)
    for enc in args.encoders:
        torch.manual_seed(0)
        c = T.Config(max_train_steps=args.max_steps, patience=args.patience, encoder=enc)
        model = DynamicsModel(latent_dim=c.latent_dim, encoder=enc).to(dev)
        lat = LatentTable(n_settings, c.latent_dim).to(dev)
        opt = torch.optim.AdamW([{"params": model.parameters(), "lr": c.lr, "weight_decay": c.weight_decay},
                                 {"params": lat.parameters(), "lr": c.latent_lr, "weight_decay": 0.0}])
        t0 = time.time()
        r = T.train(model, lat, opt, buf, c, dev, SummaryWriter(f"{args.out}/{enc}"), [0])
        err = T.evaluate(model, lat, t, va)
        tr_err = T.evaluate(model, lat, t, tr).mean().item()
        cells = "".join(f"{100*err[(mag >= lo) & (mag < hi)].mean():8.2f}cm"
                        f"({(err[(mag >= lo) & (mag < hi)] / mag[(mag >= lo) & (mag < hi)]).median():4.2f})" for lo, hi in bins)
        print(f"{enc:12s} {r['steps']:6d} {100*tr_err:8.2f}cm {100*err.mean():5.2f}{cells}{(time.time()-t0)/60:8.1f}", flush=True)
        torch.save({"model": model.state_dict(), "latents": lat.state_dict(), "encoder": enc}, f"{args.out}/{enc}/model.pt")


if __name__ == "__main__":
    main()
