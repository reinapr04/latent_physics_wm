"""Test-time latent inference: fit a latent to a new environment's pushes with the dynamics
model frozen (only the latent is optimized, with the same loss and L2 prior as in training)."""

from __future__ import annotations

import torch

from dynamics_model.model import point_flow_loss, training_objective, true_flow_magnitude


def infer_latents(model, data, group, n_groups, z_init, steps=200, lr=1e-2, l2=1e-3, relative_floor=None):
    """Fit one latent per group (environment) at once.

    data: dict with the model's input (pts_in / img_feat), act_feat, action, pts_loss, R_gt, t_gt;
    group (N,): which group each transition belongs to; z_init (L,) or (n_groups, L).
    Each group's latent minimizes  mean point-flow objective over its transitions + l2 * |z|^2
    (relative_floor set = the relative objective, as in training with --loss relative),
    independently of the other groups. Returns (n_groups, L)."""
    flow_mag = true_flow_magnitude(data["pts_loss"], data["R_gt"], data["t_gt"])
    dev = data["act_feat"].device
    z = z_init.detach().to(dev).expand(n_groups, -1).clone().requires_grad_(True)
    count = torch.bincount(group, minlength=n_groups).clamp_min(1).float()
    opt = torch.optim.Adam([z], lr=lr)
    was_training = model.training
    model.eval()
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        R, t = model(data[model.input_key], data["act_feat"], z[group], act_world=data.get("action"),
                     aux=data.get(model.aux_key) if model.aux_key else None)
        err = training_objective(point_flow_loss(data["pts_loss"], R, t, data["R_gt"], data["t_gt"]), flow_mag, relative_floor)
        per_group = torch.zeros(n_groups, device=dev).index_add_(0, group, err) / count
        loss = per_group.sum() + l2 * z.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p, r in zip(model.parameters(), req):
        p.requires_grad_(r)
    model.train(was_training)
    return z.detach()
