"""Latent-conditioned rigid-body dynamics model.

    point cloud (action frame) --encoder--> 128-d feature
    [feature, action (p_z, F, T), latent z]  --3-layer MLP (hidden 256)--> 6-DoF motion

The output is the block's rigid motion caused by the push, in the action frame: an axis-angle
rotation about the frame origin (the contact point on the table) and a translation.

Encoders (all see the push-conditioning c = [action, latent]; the head always sees it too):
  pointnet        per-point MLP + max pool; c only enters in the head (late fusion)
  pointnet_early  c is appended to every point before the per-point MLP, so the pooled feature
                  depends on the push (early fusion)
  cross_attn      per-point features; a query built from c attends over the points
                  ("which points matter for this push, with this physics") + max pool
  transformer     64 local groups of 16 points (farthest point centers + nearest neighbours)
                  -> mini-PointNet tokens -> 2 self-attention layers (geometry between groups)
                  -> the same c-query cross-attention readout + max pool

Each unique physics setting gets its own latent vector (a row of an nn.Embedding indexed by the
setting id), learned by backprop together with the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dynamics_model.geometry import axis_angle_to_matrix, farthest_point_indices

POINT_SCALE = 0.05  # m; points are divided by this before the encoder
ACTION_SCALE = (0.04, 5.0, 1.2)  # p_z (m), F (N), T (s)
TRANSLATION_SCALE = 0.1  # m per unit of network output
ENCODERS = ["pointnet", "pointnet_early", "cross_attn", "transformer"]


def mlp(dims):
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.ReLU())
    return nn.Sequential(*mods)


class PointNet(nn.Module):
    """Per-point MLP + max pool. cond_dim > 0: c is appended to every point (early fusion)."""

    def __init__(self, out_dim=128, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = mlp([3 + cond_dim, 64, 128, out_dim])

    def forward(self, pts, cond=None):  # (B, N, 3) -> (B, out_dim)
        x = pts
        if self.cond_dim:
            x = torch.cat([pts, cond[:, None].expand(-1, pts.shape[1], -1)], -1)
        return self.mlp(x).max(dim=1).values


class CondReadout(nn.Module):
    """A query built from the conditioning attends over point/group tokens; the result is
    concatenated with the max-pooled tokens."""

    def __init__(self, dim, cond_dim, heads=4):
        super().__init__()
        self.query = mlp([cond_dim, dim, dim])
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(2 * dim, dim)

    def forward(self, tokens, cond):
        q = self.query(cond)[:, None]
        att, _ = self.attn(q, tokens, tokens)
        return self.out(torch.cat([att[:, 0], tokens.max(dim=1).values], -1))


class CrossAttnEncoder(nn.Module):
    def __init__(self, out_dim=128, cond_dim=11):
        super().__init__()
        self.point_mlp = mlp([3, 64, 128, out_dim])
        self.readout = CondReadout(out_dim, cond_dim)

    def forward(self, pts, cond):
        return self.readout(self.point_mlp(pts), cond)


class TransformerEncoder(nn.Module):
    def __init__(self, out_dim=128, cond_dim=11, groups=64, group_size=16, layers=2, heads=4):
        super().__init__()
        self.groups, self.group_size = groups, group_size
        self.local = mlp([3, 64, out_dim])  # mini-PointNet on each group (relative coordinates)
        self.pos = mlp([3, 64, out_dim])  # group center position embedding
        layer = nn.TransformerEncoderLayer(out_dim, heads, dim_feedforward=2 * out_dim, batch_first=True,
                                           dropout=0.0, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.readout = CondReadout(out_dim, cond_dim, heads)

    def forward(self, pts, cond):
        B, N, _ = pts.shape
        with torch.no_grad():
            c_idx = farthest_point_indices(pts, torch.ones(B, N, dtype=torch.bool, device=pts.device), self.groups)
            centers = torch.gather(pts, 1, c_idx[..., None].expand(-1, -1, 3))  # (B, G, 3)
            nn_idx = torch.cdist(centers, pts).topk(self.group_size, largest=False).indices  # (B, G, K)
        grouped = torch.gather(pts[:, None].expand(-1, self.groups, -1, -1), 2,
                               nn_idx[..., None].expand(-1, -1, -1, 3))  # (B, G, K, 3)
        tokens = self.local(grouped - centers[:, :, None]).max(dim=2).values + self.pos(centers)
        return self.readout(self.blocks(tokens), cond)


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=8, feat_dim=128, hidden=256, layers=3, encoder="pointnet"):
        super().__init__()
        assert encoder in ENCODERS, f"encoder must be one of {ENCODERS}"
        self.latent_dim = latent_dim
        self.encoder_type = encoder
        cond_dim = 3 + latent_dim
        if encoder == "pointnet":
            self.encoder = PointNet(feat_dim)
        elif encoder == "pointnet_early":
            self.encoder = PointNet(feat_dim, cond_dim)
        elif encoder == "cross_attn":
            self.encoder = CrossAttnEncoder(feat_dim, cond_dim)
        else:
            self.encoder = TransformerEncoder(feat_dim, cond_dim)
        dims = [feat_dim + cond_dim] + [hidden] * (layers - 1) + [6]
        mods = []
        for i in range(layers):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < layers - 1:
                mods.append(nn.ReLU())
        self.head = nn.Sequential(*mods)
        nn.init.zeros_(self.head[-1].weight)  # start by predicting "no motion"
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("action_scale", torch.tensor(ACTION_SCALE))

    def forward(self, pts_frame, act_feat, z):
        """pts_frame (B, N, 3): input cloud in the action frame; act_feat (B, 3): (p_z, F, T);
        z (B, latent_dim). Returns rotation matrices (B, 3, 3) and translations (B, 3), m."""
        cond = torch.cat([act_feat / self.action_scale, z], -1)
        if self.encoder_type == "pointnet":
            f = self.encoder(pts_frame / POINT_SCALE)
        else:
            f = self.encoder(pts_frame / POINT_SCALE, cond)
        out = self.head(torch.cat([f, cond], -1))
        return axis_angle_to_matrix(out[:, :3]), out[:, 3:] * TRANSLATION_SCALE


class LatentTable(nn.Module):
    """setting id -> latent vector."""

    def __init__(self, num_settings, latent_dim=8, init_std=0.01):
        super().__init__()
        self.emb = nn.Embedding(num_settings, latent_dim)
        nn.init.normal_(self.emb.weight, std=init_std)

    def forward(self, ids):
        return self.emb(ids)


class Ensemble(nn.Module):
    """Several dynamics models used as one. Each member has its own latent space, so the
    ensemble's latent is the members' latents concatenated (member k uses z[:, kL:(k+1)L]).

    Returns the mean R and mean t: since x' = R x + t is linear in R and t, moving points with
    them gives exactly the mean of the members' predicted points (mean R is not a rotation, but
    it is only ever applied to points)."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.latent_dim = members[0].latent_dim

    def forward(self, pts, act, z):
        L = self.latent_dim
        outs = [m(pts, act, z[:, k * L:(k + 1) * L]) for k, m in enumerate(self.members)]
        return torch.stack([R for R, _ in outs]).mean(0), torch.stack([t for _, t in outs]).mean(0)


def apply_rigid(pts, R, t):
    return pts @ R.transpose(-1, -2) + t[:, None, :]


def point_flow_loss(pts, R_pred, t_pred, R_gt, t_gt):
    """Mean distance (m) between the block's points moved by the predicted and by the true
    rigid motion. pts (B, N, 3) are the block's observed points in the action frame. Returns
    per-sample errors (B,)."""
    return (apply_rigid(pts, R_pred, t_pred) - apply_rigid(pts, R_gt, t_gt)).norm(dim=-1).mean(-1)


def true_flow_magnitude(pts, R_gt, t_gt):
    """How far the block's points actually moved (mean distance, m), per sample (B,)."""
    return (apply_rigid(pts, R_gt, t_gt) - pts).norm(dim=-1).mean(-1)


def training_objective(err, flow_mag, relative_floor=None):
    """Per-sample objective: the absolute point-flow error (m), or with relative_floor set, the
    error relative to how far the block moved: err / (flow_mag + relative_floor). The relative
    form makes small, physics-revealing pushes count as much as long slides."""
    return err if relative_floor is None else err / (flow_mag + relative_floor)
