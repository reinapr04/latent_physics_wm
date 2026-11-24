"""Train an ensemble of dynamics models on a saved dataset (no simulation) and evaluate the
averaged prediction.

Each member is trained from scratch with its own seed (initial weights, batch order,
augmentation) and its own latents, on the same train/val split. The ensemble prediction is the
average of the members' predicted point positions (each member moves the block's points with
its predicted rigid motion). After each member, the error of the ensemble so far is printed, so
you can see what each extra member buys.

    python scripts/train_ensemble.py --dataset runs/dynamics1/dataset.pt --out runs/ensemble --members 5
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import train_dynamics as T  # noqa: E402
from dynamics_model.data import TransitionBuffer  # noqa: E402
from dynamics_model.model import DynamicsModel, LatentTable, apply_rigid, true_flow_magnitude  # noqa: E402
from env import get_task  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

BINS = [(0.005, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 10.0)]


@torch.no_grad()
def predicted_points(model, latents, t, idx):
    """Block points moved by the model's predicted motion, (len(idx), N, 3)."""
    model.eval()
    out = []
    for i in range(0, len(idx), 1024):
        b = idx[i:i + 1024]
        R, tr = model(t[model.input_key][b], t["act_feat"][b], latents(t["setting"][b]), act_world=t["action"][b])
        out.append(apply_rigid(t["pts_loss"][b], R, tr))
    return torch.cat(out)


def report(name, pred, gt, mag):
    err = (pred - gt).norm(dim=-1).mean(-1)
    cells = ""
    for lo, hi in BINS:
        m = (mag >= lo) & (mag < hi)
        cells += f"{100 * err[m].mean():8.2f}cm({(err[m] / mag[m]).median():4.2f})"
    moved = mag > 0.005
    print(f"{name:22s} {100 * err.mean():6.2f}cm {100 * err[moved].mean():8.2f}cm{cells}", flush=True)
    return {"val_err_cm": 100 * err.mean().item(), "val_err_moved_cm": 100 * err[moved].mean().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="runs/dynamics1/dataset.pt")
    ap.add_argument("--out", default="runs/ensemble")
    ap.add_argument("--encoder", default="cross_attn")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20000, help="training steps per member (no early stopping)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual", action="store_true", help="residual MLP head (dynamics_model.model)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--feat-dim", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = TransitionBuffer.load(args.dataset)
    task = get_task(buf.meta.get("task", "block"))
    dev = "cuda"
    t = buf.tensors(dev)
    _, va = buf.split_indices(dev)
    n_settings = int(t["setting"].max()) + 1
    mag = true_flow_magnitude(t["pts_loss"][va], t["R_gt"][va], t["t_gt"][va])
    gt = apply_rigid(t["pts_loss"][va], t["R_gt"][va], t["t_gt"][va])
    print(f"dataset {len(buf)}, val {len(va)}, encoder {args.encoder}, {args.members} members x {args.steps} steps", flush=True)
    header = f"{'':22s} {'val':>8s} {'val moved':>10s}" + "".join(
        f"{f'{100*lo:.1f}-{100*hi:.0f}cm':>16s}" for lo, hi in BINS)

    preds, results = [], {}
    for k in range(args.members):
        seed = args.seed + k
        torch.manual_seed(seed)
        c = T.Config(max_train_steps=args.steps, patience=10 ** 9, encoder=args.encoder, task=task.name,
                     residual=args.residual, hidden=args.hidden, layers=args.layers, feat_dim=args.feat_dim)
        kw = T.model_kwargs(c, task)
        model = DynamicsModel(**kw).to(dev)
        lat = LatentTable(n_settings, c.latent_dim).to(dev)
        opt = torch.optim.AdamW([{"params": model.parameters(), "lr": c.lr, "weight_decay": c.weight_decay},
                                 {"params": lat.parameters(), "lr": c.latent_lr, "weight_decay": 0.0}])
        t0 = time.time()
        T.train(model, lat, opt, buf, c, dev, SummaryWriter(str(out / f"member{k}")), [0])
        torch.save({"model": model.state_dict(), "latents": lat.state_dict(), "encoder": args.encoder, "seed": seed,
                    "model_kwargs": kw, "task": task.name}, out / f"member{k}" / "model.pt")
        preds.append(predicted_points(model, lat, t, va))
        print(f"\nmember {k} trained in {(time.time() - t0) / 60:.1f} min", flush=True)
        print(header, flush=True)
        results[f"member{k}"] = report(f"member {k} alone", preds[-1], gt, mag)
        results[f"ensemble_of_{k + 1}"] = report(f"ensemble of {k + 1}", torch.stack(preds).mean(0), gt, mag)
        (out / "results.json").write_text(json.dumps(results, indent=2))
    print("\ncells: mean error (median error / distance moved)", flush=True)


if __name__ == "__main__":
    main()
