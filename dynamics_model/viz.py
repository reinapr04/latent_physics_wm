"""Figures of predicted vs. true dense point flow of the block, for TensorBoard."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dynamics_model.model import apply_rigid  # noqa: E402


@torch.no_grad()
def flow_figure(model, latents, t, idx, max_arrows=40):
    """One row per transition: top view (x-y) and side view (x-z) of the block's observed points
    in the action frame (push from the origin along +x, red arrow), zoomed to the block.

    grey  = block points before the push
    green = moved by the true rigid motion (arrows: true flow)
    red   = moved by the predicted rigid motion (arrows: predicted flow)
    """
    model.eval()
    R, tr = model(t[model.input_key][idx], t["act_feat"][idx], latents(t["setting"][idx]), act_world=t["action"][idx],
                  aux=t[model.aux_key][idx] if model.aux_key else None)
    model.train()
    pts = t["pts_loss"][idx]
    gt = apply_rigid(pts, t["R_gt"][idx], t["t_gt"][idx])
    pred = apply_rigid(pts, R, tr)
    err = (gt - pred).norm(dim=-1).mean(-1)
    pts, gt, pred = pts.cpu().numpy(), gt.cpu().numpy(), pred.cpu().numpy()
    act = t["act_feat"][idx].cpu().numpy()
    phys = t["physics"][idx].cpu().numpy()

    n = len(idx)
    fig, axes = plt.subplots(n, 2, figsize=(9, 4.2 * n), squeeze=False)
    rng = np.random.default_rng(0)
    for i in range(n):
        uniq = np.unique(pts[i], axis=0, return_index=True)[1]  # loss points repeat to fill slots
        arrows = rng.choice(uniq, min(max_arrows, len(uniq)), replace=False)
        allp = np.concatenate([pts[i], gt[i], pred[i]])
        for col, (a, b, name) in enumerate([(0, 1, "top (x-y)"), (0, 2, "side (x-z)")]):
            ax = axes[i, col]
            ax.scatter(pts[i][uniq, a], pts[i][uniq, b], s=4, c="0.6", label="before")
            ax.scatter(gt[i][uniq, a], gt[i][uniq, b], s=4, c="tab:green", label="true")
            ax.scatter(pred[i][uniq, a], pred[i][uniq, b], s=4, c="tab:red", label="predicted")
            for k, color in [(gt, "tab:green"), (pred, "tab:red")]:
                d = k[i][arrows] - pts[i][arrows]
                ax.quiver(pts[i][arrows, a], pts[i][arrows, b], d[:, a], d[:, b], color=color,
                          angles="xy", scale_units="xy", scale=1, width=0.003, alpha=0.6)
            # push: from the origin along +x, at height p_z
            push_b = 0.0 if b == 1 else act[i, 0]
            ax.annotate("", xy=(0.0, push_b), xytext=(-0.03, push_b),
                        arrowprops=dict(color="red", lw=2, arrowstyle="->"))
            lo, hi = allp.min(0) - 0.015, allp.max(0) + 0.015
            lo[a], hi[a] = min(lo[a], -0.035), max(hi[a], 0.005)
            if b == 2:
                lo[b] = min(lo[b], -0.005)
            ax.set_xlim(lo[a], hi[a]), ax.set_ylim(lo[b], hi[b])
            ax.set_aspect("equal")
            ax.set_xlabel("xyz"[a] + " (m, push dir)" if a == 0 else "xyz"[a])
            ax.set_ylabel("xyz"[b] + " (m)")
            ax.grid(alpha=0.3)
            if i == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")
            ax.set_title(
                f"{name} | F {act[i, 1]:.2f}N T {act[i, 2]:.2f}s h {100 * act[i, 0]:.1f}cm | "
                f"m {phys[i, 0]:.2f} mu_s {phys[i, 1]:.2f} mu_d {phys[i, 2]:.2f} | err {100 * err[i]:.2f}cm",
                fontsize=7,
            )
    fig.tight_layout()
    return fig
