"""Latent-conditioned rigid-body dynamics model.

    observation --encoder--> feat_dim feature
    [feature, action, latent z]  --head (MLP)--> 6-DoF motion

The output is the object's rigid motion caused by the push, in the action frame: an axis-angle
rotation about the frame origin (the contact point on the table) and a translation. With
rotation=False (a ball: its rotation is unobservable) the rotation is fixed to the identity.

Observation encoders. Point encoders see the observed point cloud in the push's action frame
(dynamics_model.geometry); all of them (and the head) see the push conditioning
c = [(p_z, F, T), z]:
  pointnet        per-point MLP + max pool; c only enters in the head (late fusion)
  pointnet_early  c is appended to every point before the per-point MLP (early fusion)
  cross_attn      per-point features; a query built from c attends over the points
                  ("which points matter for this push, with this physics") + max pool
  transformer     64 local groups of 16 points (farthest point centers + nearest neighbours)
                  -> mini-PointNet tokens -> 2 self-attention layers -> the same c-query
                  cross-attention readout + max pool
Image encoders see frozen DINOv2 ViT-S/14 features of the camera image instead
(scripts/dino_features.py computes them once per dataset). The image is in the world frame, so
these models also get the push's world position and direction (p_x, p_y, d_x, d_y) in c:
  image_cls       [CLS token, mean patch token] -> MLP
  image_attn      patch tokens (16 x 16) -> linear -> the c-query cross-attention readout + max pool

Representation variants (scripts/ablate.py --group repr):
  pointnet_geom       per-point input [xyz, raised flag, |y| (distance to the push line), r] instead of xyz
  pointnet_objcentric points centred on the observed object (its xy centroid is appended to c)
  pointnet_dino       PointNet feature + frozen DINOv2 feature of the world-frame image (fusion)
  image_af_cls        frozen DINOv2 [CLS, mean patch] of the image rectified to the table plane and
                      cropped/rotated into the push's action frame (scripts/image_crops.py)
  image_cnn           small CNN trained from scratch on the 96 x 96 world-frame image
  image_af_cnn        the same CNN on the 96 x 96 action-frame crop (mirror augmentation = flip)

Head: a plain ReLU MLP (`layers` linear layers, width `hidden`), or with residual=True a
pre-LayerNorm residual MLP (`layers` blocks). The last layer starts at zero ("no motion").

Each unique physics setting gets its own latent vector (a row of an nn.Embedding indexed by the
setting id), learned by backprop together with the model. latent_dim=0 disables the latent
(an ablation: one model for all physics).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dynamics_model.geometry import axis_angle_to_matrix, farthest_point_indices

POINT_SCALE = 0.05  # m; points are divided by this before the encoder
ACTION_SCALE = (0.04, 5.0, 1.2)  # p_z (m), F (N), T (s)
WORLD_ACTION_SCALE = (0.3, 0.3, 1.0, 1.0)  # p_x, p_y (m), d_x, d_y: image models only
TRANSLATION_SCALE = 0.1  # m per unit of network output
POINT_ENCODERS = ["pointnet", "pointnet_early", "cross_attn", "transformer", "pointnet_geom", "pointnet_objcentric",
                  "pointnet_dino"]
IMAGE_ENCODERS = ["image_cls", "image_attn", "image_af_cls", "image_cnn", "image_af_cnn"]
ENCODERS = POINT_ENCODERS + IMAGE_ENCODERS
IMAGE_FEAT_DIM = 384  # DINOv2 ViT-S/14 token width
TABLE_Z = 0.005  # m, points above this are the object (dynamics_model.geometry.TABLE_Z)
CENTROID_SCALE = 0.1  # m, object centroid offset fed to the head (pointnet_objcentric)
# what each encoder consumes: the buffer key of its input, an optional second input, and how
# training batches may be augmented ("points": mirror y -> -y + noise, "flip": mirror an
# action-frame image by flipping its rows, "none": no augmentation possible)
ENCODER_IO = {
    "pointnet": ("pts_in", None, "points"), "pointnet_early": ("pts_in", None, "points"),
    "cross_attn": ("pts_in", None, "points"), "transformer": ("pts_in", None, "points"),
    "pointnet_geom": ("pts_in", None, "points"), "pointnet_objcentric": ("pts_in", None, "points"),
    "pointnet_dino": ("pts_in", "img_feat", "none"),
    "image_cls": ("img_feat", None, "none"), "image_attn": ("img_feat", None, "none"),
    "image_af_cls": ("img_af_feat", None, "none"),
    "image_cnn": ("image_96", None, "none"), "image_af_cnn": ("image_af_96", None, "flip"),
}


def point_features(pts, mode="xyz"):
    """Per-point input features from the action-frame cloud (B, N, 3), m.
    xyz: scaled coordinates (3). geom: + raised flag, distance to the push line |y|, distance to
    the contact point r (6). Returns (B, N, F)."""
    x = pts / POINT_SCALE
    if mode == "xyz":
        return x
    raised = (pts[..., 2] > TABLE_Z).float()
    lat = pts[..., 1].abs() / POINT_SCALE
    r = pts[..., :2].norm(dim=-1) / POINT_SCALE
    return torch.cat([x, raised[..., None], lat[..., None], r[..., None]], -1)


def object_centroid(pts):
    """xy centroid of the raised points (the object) of each action-frame cloud, (B, N, 3) -> (B, 2)."""
    raised = (pts[..., 2] > TABLE_Z).float()
    return (pts[..., :2] * raised[..., None]).sum(1) / raised.sum(1, keepdim=True).clamp_min(1)


def mlp(dims, act=nn.ReLU):
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(act())
    return nn.Sequential(*mods)


class ResMLP(nn.Module):
    """Pre-LayerNorm residual MLP: in -> hidden, `blocks` x (LN -> Linear -> GELU -> Linear) with
    skip connections, LN -> out."""

    def __init__(self, in_dim, hidden, blocks, out_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
             for _ in range(blocks)]
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


class PointNet(nn.Module):
    """Per-point MLP + max pool. cond_dim > 0: c is appended to every point (early fusion)."""

    def __init__(self, out_dim=128, cond_dim=0, in_dim=3):
        super().__init__()
        self.cond_dim = cond_dim
        self.mlp = mlp([in_dim + cond_dim, 64, 128, out_dim])

    def forward(self, pts, cond=None):  # (B, N, 3) -> (B, out_dim)
        x = pts
        if self.cond_dim:
            x = torch.cat([pts, cond[:, None].expand(-1, pts.shape[1], -1)], -1)
        return self.mlp(x).max(dim=1).values


class CondReadout(nn.Module):
    """A query built from the conditioning attends over point/group/patch tokens; the result is
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


class ImageEncoder(nn.Module):
    """Frozen DINOv2 tokens of the camera image, (B, 1 + P, D) = [CLS, patches] (fp16 or fp32).
    image_cls: MLP on [CLS, mean patch]. image_attn: patches -> linear -> CondReadout."""

    def __init__(self, out_dim=128, cond_dim=15, mode="image_cls", in_dim=IMAGE_FEAT_DIM):
        super().__init__()
        self.mode = mode
        if mode == "image_cls":
            self.proj = mlp([2 * in_dim, 2 * out_dim, out_dim])
        else:
            self.proj = nn.Linear(in_dim, out_dim)
            self.readout = CondReadout(out_dim, cond_dim)

    def forward(self, tokens, cond):
        tokens = tokens.float()
        if self.mode == "image_cls":
            return self.proj(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], -1))
        return self.readout(self.proj(tokens[:, 1:]), cond)


class SmallCNN(nn.Module):
    """Small image encoder trained from scratch: (B, H, W, 3) uint8 -> (B, out_dim).
    4 stride-2 conv blocks (GroupNorm + ReLU), 96 -> 6 px, then global max + mean pool."""

    def __init__(self, out_dim=128, width=32):
        super().__init__()
        chans = [3, width, 2 * width, 4 * width, 8 * width]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.ReLU()]
        self.conv = nn.Sequential(*blocks)
        self.out = nn.Linear(2 * chans[-1], out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img, cond=None):
        x = (img.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        h = self.conv(x)
        return self.out(torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))], -1))


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim=8, feat_dim=128, hidden=256, layers=3, encoder="pointnet", rotation=True,
                 residual=False):
        super().__init__()
        assert encoder in ENCODERS, f"encoder must be one of {ENCODERS}"
        self.kwargs = dict(latent_dim=latent_dim, feat_dim=feat_dim, hidden=hidden, layers=layers, encoder=encoder,
                           rotation=rotation, residual=residual)
        self.latent_dim = latent_dim
        self.encoder_type = encoder
        self.rotation = rotation
        self.input_kind = "image" if encoder in IMAGE_ENCODERS else "points"
        self.input_key, self.aux_key, self.aug_kind = ENCODER_IO[encoder]
        # world-frame image models also get the push's world position and direction; action-frame
        # image crops only its world direction (the rectified crop is not rotation invariant: the
        # object's sides lean away from the camera); object-centric points get the object's offset
        self.world_cond = {"image_cls": 4, "image_attn": 4, "image_cnn": 4, "pointnet_dino": 4,
                           "image_af_cls": 2, "image_af_cnn": 2, "pointnet_objcentric": 2}.get(encoder, 0)
        cond_dim = 3 + latent_dim + self.world_cond
        self.aux_encoder = None
        if encoder == "pointnet":
            self.encoder = PointNet(feat_dim)
        elif encoder == "pointnet_geom":
            self.encoder = PointNet(feat_dim, in_dim=6)
        elif encoder == "pointnet_objcentric":
            self.encoder = PointNet(feat_dim, in_dim=4)
        elif encoder == "pointnet_dino":
            self.encoder = PointNet(feat_dim)
            self.aux_encoder = ImageEncoder(feat_dim, cond_dim, "image_cls")
        elif encoder == "pointnet_early":
            self.encoder = PointNet(feat_dim, cond_dim)
        elif encoder == "cross_attn":
            self.encoder = CrossAttnEncoder(feat_dim, cond_dim)
        elif encoder == "transformer":
            self.encoder = TransformerEncoder(feat_dim, cond_dim)
        elif encoder in ("image_cnn", "image_af_cnn"):
            self.encoder = SmallCNN(feat_dim)
        else:  # image_cls, image_attn, image_af_cls
            self.encoder = ImageEncoder(feat_dim, cond_dim, "image_attn" if encoder == "image_attn" else "image_cls")
        head_in = feat_dim * (2 if self.aux_encoder is not None else 1) + cond_dim
        if residual:
            self.head = ResMLP(head_in, hidden, layers, 6)
            last = self.head.out[-1]
        else:
            self.head = mlp([head_in] + [hidden] * (layers - 1) + [6])
            last = self.head[-1]
        nn.init.zeros_(last.weight)  # start by predicting "no motion"
        nn.init.zeros_(last.bias)
        self.register_buffer("action_scale", torch.tensor(ACTION_SCALE))
        self.register_buffer("world_action_scale", torch.tensor(WORLD_ACTION_SCALE))

    def conditioning(self, act_feat, z, act_world=None, x=None):
        parts = [act_feat / self.action_scale, z]
        if self.world_cond == 4:
            assert act_world is not None, "world-frame image models need the world-frame action (B, 7)"
            parts.append(act_world[:, [0, 1, 3, 4]] / self.world_action_scale)
        elif self.world_cond == 2 and self.encoder_type == "pointnet_objcentric":
            parts.append(object_centroid(x) / CENTROID_SCALE)
        elif self.world_cond == 2:
            assert act_world is not None, "action-frame image models need the world-frame action (B, 7)"
            parts.append(act_world[:, 3:5])
        return torch.cat(parts, -1)

    def forward(self, x, act_feat, z, act_world=None, aux=None):
        """x: the encoder's input (ENCODER_IO): point models (B, N, 3) input cloud in the action
        frame; image models (B, 1 + P, D) DINOv2 tokens or (B, H, W, 3) uint8 images. act_feat
        (B, 3): (p_z, F, T); z (B, latent_dim); act_world (B, 7): the world-frame action (image
        models); aux: the second input of fusion models (pointnet_dino: world-frame DINOv2
        tokens). Returns rotation matrices (B, 3, 3) and translations (B, 3), m."""
        cond = self.conditioning(act_feat, z, act_world, x)
        e = self.encoder_type
        if e in ("pointnet", "pointnet_dino"):
            f = self.encoder(x / POINT_SCALE)
        elif e == "pointnet_geom":
            f = self.encoder(point_features(x, "geom"))
        elif e == "pointnet_objcentric":
            c = object_centroid(x)
            rel = torch.cat([x[..., :2] - c[:, None], x[..., 2:]], -1)
            f = self.encoder(torch.cat([rel / POINT_SCALE, (x[..., 2] > TABLE_Z).float()[..., None]], -1))
        elif self.input_kind == "points":
            f = self.encoder(x / POINT_SCALE, cond)
        else:
            f = self.encoder(x, cond)
        if self.aux_encoder is not None:
            assert aux is not None, f"{e} needs its second input ({self.aux_key})"
            f = torch.cat([f, self.aux_encoder(aux, cond)], -1)
        out = self.head(torch.cat([f, cond], -1))
        w = out[:, :3] if self.rotation else torch.zeros_like(out[:, :3])
        return axis_angle_to_matrix(w), out[:, 3:] * TRANSLATION_SCALE

    @classmethod
    def from_checkpoint(cls, ck, device="cpu"):
        """Rebuild the model saved by train_dynamics.py / scripts (ck = torch.load(...))."""
        kw = ck.get("model_kwargs")
        if kw is None:  # checkpoints from before model_kwargs were saved
            kw = dict(latent_dim=ck["latents"]["emb.weight"].shape[1], encoder=ck.get("encoder", "pointnet"))
        m = cls(**kw).to(device)
        m.load_state_dict(ck["model"])
        return m


class LatentTable(nn.Module):
    """setting id -> latent vector."""

    def __init__(self, num_settings, latent_dim=8, init_std=0.01):
        super().__init__()
        self.emb = nn.Embedding(num_settings, max(latent_dim, 1))
        self.latent_dim = latent_dim
        nn.init.normal_(self.emb.weight, std=init_std)

    def forward(self, ids):
        return self.emb(ids)[:, : self.latent_dim]

    @property
    def weight(self):
        return self.emb.weight[:, : self.latent_dim]


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
        self.input_kind = members[0].input_kind
        self.input_key, self.aux_key, self.aug_kind = members[0].input_key, members[0].aux_key, members[0].aug_kind
        self.encoder_type = members[0].encoder_type

    def forward(self, x, act, z, act_world=None, aux=None):
        L = self.latent_dim
        outs = [m(x, act, z[:, k * L:(k + 1) * L], act_world, aux) for k, m in enumerate(self.members)]
        return torch.stack([R for R, _ in outs]).mean(0), torch.stack([t for _, t in outs]).mean(0)


def apply_rigid(pts, R, t):
    return pts @ R.transpose(-1, -2) + t[:, None, :]


def point_flow_loss(pts, R_pred, t_pred, R_gt, t_gt):
    """Mean distance (m) between the object's points moved by the predicted and by the true
    rigid motion. pts (B, N, 3) are the object's observed points in the action frame. Returns
    per-sample errors (B,)."""
    return (apply_rigid(pts, R_pred, t_pred) - apply_rigid(pts, R_gt, t_gt)).norm(dim=-1).mean(-1)


def true_flow_magnitude(pts, R_gt, t_gt):
    """How far the object's points actually moved (mean distance, m), per sample (B,)."""
    return (apply_rigid(pts, R_gt, t_gt) - pts).norm(dim=-1).mean(-1)


def training_objective(err, flow_mag, relative_floor=None):
    """Per-sample objective: the absolute point-flow error (m), or with relative_floor set, the
    error relative to how far the object moved: err / (flow_mag + relative_floor). The relative
    form makes small, physics-revealing pushes count as much as long slides."""
    return err if relative_floor is None else err / (flow_mag + relative_floor)
