"""Geometry helpers: action frame, rigid transforms, point cloud preprocessing.

Action frame: origin at the push contact point projected onto the table, (p_x, p_y, 0); x axis
along the push direction, z up. Expressed in this frame a push is just (p_z, F, T) and the
point cloud says where things are relative to the push, so the model doesn't have to learn
the same physics for every push position and direction.

Model input: all observed points above the table plane (found by the table's known height, not
segmentation - i.e. the block) plus a fixed number of evenly spread table points, expressed in
the action frame. That is only a change of coordinates: nothing is cropped, so the model sees
where the block is relative to the push however far away it is, and where on the table it is.
"""

from __future__ import annotations

import torch

# workspace (world frame, m): the table area the block can be on; z_max drops the parked arm
WORKSPACE_LOW = (-0.35, -0.5, -0.01)
WORKSPACE_HIGH = (0.5, 0.5, 0.15)
# table top is z = 0; points above this are "raised" (not table)
TABLE_Z = 0.005


# ---------------------------------------------------------------- rotations
def skew(w):
    o = torch.zeros_like(w[..., 0])
    return torch.stack([o, -w[..., 2], w[..., 1], w[..., 2], o, -w[..., 0], -w[..., 1], w[..., 0], o], -1).reshape(
        *w.shape[:-1], 3, 3
    )


def axis_angle_to_matrix(w):
    """Rodrigues' formula, (..., 3) -> (..., 3, 3); smooth at w = 0."""
    theta2 = (w * w).sum(-1, keepdim=True)[..., None]  # (..., 1, 1)
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    K = skew(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return eye + a * K + b * (K @ K)


def quat_to_matrix(q):
    """(..., 4) wxyz -> (..., 3, 3)"""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


# ------------------------------------------------------------- action frame
def action_frame(action):
    """action (..., 7) = [p(3), d(2), F, T] -> origin (..., 3), rotation R_a (..., 3, 3) whose
    columns are the frame axes in world coordinates (x_world = R_a @ x_frame + origin)."""
    p, d = action[..., 0:3], action[..., 3:5]
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    origin = torch.cat([p[..., :2], torch.zeros_like(p[..., :1])], -1)
    zero, one = torch.zeros_like(d[..., 0]), torch.ones_like(d[..., 0])
    x_ax = torch.stack([d[..., 0], d[..., 1], zero], -1)
    y_ax = torch.stack([-d[..., 1], d[..., 0], zero], -1)
    z_ax = torch.stack([zero, zero, one], -1)
    return origin, torch.stack([x_ax, y_ax, z_ax], -1)


def world_to_frame(points, origin, R_a):
    """points (..., N, 3) world -> action frame."""
    return (points - origin[..., None, :]) @ R_a


def frame_to_world(points, origin, R_a):
    return points @ R_a.transpose(-1, -2) + origin[..., None, :]


def action_features(action):
    """What is left of the action in its own frame: (p_z, F, T)."""
    return torch.stack([action[..., 2], action[..., 5], action[..., 6]], -1)


def world_transform_to_frame(R, t, origin, R_a):
    """Rigid motion x' = R x + t (world) expressed in the action frame."""
    R_f = R_a.transpose(-1, -2) @ R @ R_a
    t_f = ((R @ origin[..., None])[..., 0] + t - origin)[..., None, :] @ R_a
    return R_f, t_f[..., 0, :]


def pose_change(p0, q0, p1, q1):
    """Block pose (position, wxyz quat) before/after -> world rigid motion R, t with
    x_after = R x_before + t for every point of the block."""
    R0, R1 = quat_to_matrix(q0), quat_to_matrix(q1)
    R = R1 @ R0.transpose(-1, -2)
    t = p1 - (R @ p0[..., None])[..., 0]
    return R, t


# ------------------------------------------------------------ point clouds
def farthest_point_indices(xyz, mask, n):
    """Farthest point sampling among the masked points: n indices per cloud that cover the
    cloud evenly. xyz (B, N, 3), mask (B, N) -> (B, n). Starts from the first masked point."""
    B = xyz.shape[0]
    rows = torch.arange(B, device=xyz.device)
    dist = torch.where(mask, torch.full_like(mask, float("inf"), dtype=xyz.dtype),
                       torch.full_like(mask, float("-inf"), dtype=xyz.dtype))  # -inf: never picked
    idx = torch.empty(B, n, dtype=torch.long, device=xyz.device)
    cur = mask.float().argmax(dim=1)
    for i in range(n):
        idx[:, i] = cur
        d = ((xyz - xyz[rows, cur][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        cur = dist.argmax(dim=1)
    return idx


def fixed_size_points(xyz, mask, n):
    """Exactly n points per cloud from the masked points, keeping all of them when possible.

    - at most n masked points: every one of them is included, and the remaining slots repeat
      them in order (PointNet max-pools over points, so repeats change nothing);
    - more than n: farthest point sampling picks n that cover the cloud evenly.
    Deterministic. xyz (B, N, 3), mask (B, N) -> (B, n, 3), number of masked points (B,)."""
    count = mask.sum(dim=1)
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)  # masked points first
    slots = torch.arange(n, device=xyz.device)[None] % count.clamp_min(1)[:, None]
    idx = order.gather(1, slots)
    too_many = count > n
    if too_many.any():
        idx[too_many] = farthest_point_indices(xyz[too_many], mask[too_many], n)
    return torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3)), count


def _in_box(xyz, lo, hi):
    lo = torch.tensor(lo, device=xyz.device)
    hi = torch.tensor(hi, device=xyz.device)
    return ((xyz > lo) & (xyz < hi)).all(-1)


def observed_cloud(obs, n, table_points=100, workspace=(WORKSPACE_LOW, WORKSPACE_HIGH)):
    """The observed cloud the model works with (world frame), exactly n points per env:
      - the observed points above the table plane (the object): all of them when there are at
        most n - table_points, otherwise n - table_points of them by farthest point sampling
        (evenly spread over the object);
      - object slots left over repeat the object points (PointNet max-pools, so repeats change
        nothing);
      - then table_points table points, evenly spread over the observed table by farthest
        point sampling (the table's outline: where on the table it pushes).
    For the block (at most ~330 points) this is every block point plus table_points table
    points; for the T (~1500-2500 points) the T is subsampled so the table points still fit.
    workspace: (low, high) world-frame box the cloud is cropped to (the table area the object
    can be on; the top drops the parked arm).
    Returns (B, n, 3) and the number of observed points above the table (B,)."""
    xyzw = obs["pointcloud"]["xyzw"]
    xyz = xyzw[..., :3]
    in_ws = (xyzw[..., 3] > 0) & _in_box(xyz, *workspace)
    raised = in_ws & (xyz[..., 2] > TABLE_Z)
    obj, count = fixed_size_points(xyz, raised, n - table_points)  # object points first, then repeats
    if table_points == 0:
        return obj, count
    idx = farthest_point_indices(xyz, in_ws & ~raised, table_points)
    table = torch.gather(xyz, 1, idx[..., None].expand(-1, -1, 3))  # (B, table_points, 3)
    return torch.cat([obj, table], 1), count


def model_input(clouds, actions):
    """Model input for each push: the observed cloud in the push's action frame.
    clouds (B, n, 3) world, actions (B, 7) -> (B, n, 3)."""
    origin, R_a = action_frame(actions)
    return world_to_frame(clouds, origin, R_a)


def block_points(obs, block_seg_id, n, min_points=5):
    """Observed points of the block (from simulator segmentation; used only for the loss), as
    exactly n points (all of them when there are at most n). block_seg_id: int or (B,) tensor."""
    pc = obs["pointcloud"]
    seg_id = torch.as_tensor(block_seg_id, device=pc["segmentation"].device).reshape(-1, 1)
    mask = (pc["segmentation"][..., 0] == seg_id) & (pc["xyzw"][..., 3] > 0)
    pts, count = fixed_size_points(pc["xyzw"][..., :3], mask, n)
    return pts, count >= min_points
