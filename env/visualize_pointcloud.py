"""Visualise the point cloud observation of PushBlock-v1 (with the push-primitive wrapper).

For each env, plots the observed point cloud (world frame, cropped to the workspace, coloured
by RGB) from the camera's side and from above, and marks one example action: a p_contact on
the block face opposite a random push direction (often a face the camera cannot see) and the
push direction.

Run from the repo root:
    python -m env.visualize_pointcloud               # 4 envs -> env/examples/pointcloud_seed0.png
    python -m env.visualize_pointcloud --seed 3 --num-envs 2
    python -m env.visualize_pointcloud --after-push  # also show the cloud after executing the push
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env import BlockPhysicsConfig, get_task, make_push_env  # noqa: E402

warnings.filterwarnings("ignore")

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
CROP_LOW = np.array([-0.30, -0.35, -0.005])  # m, world frame; drops the floor and far table
CROP_HIGH = np.array([0.40, 0.35, 0.30])


def cloud(obs, i):
    """Cropped xyz (m) and rgb (0-1) of env i."""
    pc = obs["pointcloud"]
    xyzw = pc["xyzw"][i].cpu().numpy()
    rgb = pc["rgb"][i].cpu().numpy() / 255.0
    keep = (xyzw[:, 3] > 0) & np.all((xyzw[:, :3] > CROP_LOW) & (xyzw[:, :3] < CROP_HIGH), axis=1)
    return xyzw[keep, :3], rgb[keep]


def plot_env(fig, row, ncols, xyz, rgb, p, d, title, cam_eye, center):
    v = cam_eye - np.array([0.05, 0.0, 0.0])
    cam_elev = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    cam_azim = math.degrees(math.atan2(v[1], v[0]))
    full = (CROP_LOW, CROP_HIGH)
    zoom = (np.array([center[0] - 0.07, center[1] - 0.07, 0.0]), np.array([center[0] + 0.07, center[1] + 0.07, 0.07]))
    for col, (name, elev, azim, (lo, hi)) in enumerate([
        ("camera side", cam_elev, cam_azim, full),
        ("top down", 90, -90, full),
        ("zoom on block", 35, cam_azim + 180, zoom),  # from behind: shows the unobserved faces
    ]):
        ax = fig.add_subplot(row[0], ncols, row[1] * ncols + col + 1, projection="3d")
        m = np.all((xyz > lo - 1e-3) & (xyz < hi + 1e-3), axis=1)
        size = 1.5 if lo is CROP_LOW else 6
        ax.scatter(xyz[m, 0], xyz[m, 1], xyz[m, 2], c=rgb[m], s=size, depthshade=False, linewidths=0)
        if p is not None:
            arrow = 0.05 if lo is CROP_LOW else 0.03
            ax.scatter(*p, c="red", s=80, marker="*", depthshade=False, zorder=10)
            ax.quiver(*(p - (arrow + 0.005) * np.append(d, 0)), *(arrow * np.append(d, 0)), color="red",
                      linewidth=2, arrow_length_ratio=0.35)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
        ax.tick_params(labelsize=6)
        ax.set_title(f"{title} - {name}", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--after-push", action="store_true", help="also plot the cloud after the push")
    ap.add_argument("--out", default=None, help="default: env/examples/pointcloud_<task>_seed<seed>.png")
    ap.add_argument("--task", default="block", help="block | ball (env/tasks.py)")
    args = ap.parse_args()
    n = args.num_envs
    task = get_task(args.task)

    e = make_push_env(
        num_envs=n,
        task=task,
        obs_mode="pointcloud",
        physics=BlockPhysicsConfig(mass=(0.1, 0.5), static_friction=(0.3, 0.8), dynamic_friction=(0.3, 0.8),
                                   linear_damping=task.linear_damping[1], angular_damping=task.angular_damping[1]),
    )
    obs, info = e.reset(seed=args.seed)
    u = e.unwrapped
    cam_eye = u._sensor_configs["base_camera"].pose.raw_pose[0, :3].cpu().numpy() \
        if hasattr(u, "_sensor_configs") else np.array([0.6, 0.25, 0.6])

    # one example push per env: a random direction, contact on the face opposite it
    gen = torch.Generator().manual_seed(args.seed)
    ang = torch.rand(n, generator=gen) * 2 * math.pi
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    if task.shape == "sphere":
        c = u.obj.pose.p.cpu()
        p = torch.cat([c[:, :2] - task.half_size * d, torch.full((n, 1), task.half_size)], 1).numpy()
    else:
        local = torch.zeros((n, 3))
        local[:, :2] = -d
        p = e.block_frame_to_world(local, d).cpu().numpy()
    d = d.numpy()

    stages = [("reset", obs, p, u.obj.pose.p.cpu().numpy())]
    if args.after_push:
        a = torch.cat([torch.as_tensor(p), torch.as_tensor(d), torch.full((n, 1), 1.5), torch.full((n, 1), 0.8)], 1)
        obs2, *_ = e.step(a)
        stages.append(("after push", obs2, None, u.obj.pose.p.cpu().numpy()))

    nrows = n * len(stages)
    fig = plt.figure(figsize=(15, 4.2 * nrows))
    r = 0
    for i in range(n):
        for name, o, pp, centers in stages:
            xyz, rgb = cloud(o, i)
            title = f"env {i} {name} ({len(xyz)} pts)"
            if pp is not None:
                title += f"\np=({100 * pp[i, 0]:+.1f},{100 * pp[i, 1]:+.1f},{100 * pp[i, 2]:.1f})cm " \
                         f"dir {math.degrees(math.atan2(d[i, 1], d[i, 0])):+.0f}deg"
            plot_env(fig, (nrows, r), 3, xyz, rgb, None if pp is None else pp[i], d[i], title, cam_eye, centers[i])
            r += 1
    fig.tight_layout()
    out = Path(args.out) if args.out else EXAMPLES_DIR / f"pointcloud_{task.name}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")

    plot_model_input(obs, p, d, n, out.with_name(out.stem.replace("pointcloud", "model_input") + ".png"))
    e.close()


def plot_model_input(obs, p, d, n, out):
    """What the dynamics model receives for the example push: the observed cloud with the table
    removed (n_in points: all observed points, repeated to fill), in world coordinates and in
    the push's action frame (origin = contact point on the table, x = push direction)."""
    from dynamics_model.geometry import model_input, observed_cloud

    a = torch.cat([torch.as_tensor(p), torch.as_tensor(d), torch.full((n, 1), 1.5), torch.full((n, 1), 0.8)], 1)
    a = a.to(obs["pointcloud"]["xyzw"].device).float()
    cloud, n_obs = observed_cloud(obs, 512)
    pts_f = model_input(cloud, a)
    cloud, pts_f, a = cloud.cpu().numpy(), pts_f.cpu().numpy(), a.cpu().numpy()

    fig = plt.figure(figsize=(10, 4.4 * n))
    for i in range(n):
        uniq = len(np.unique(cloud[i], axis=0))
        for col, (pts, star, arrow, frame) in enumerate([
            (cloud[i], a[i, :3], np.append(a[i, 3:5], 0), "world frame"),
            (pts_f[i], np.array([0, 0, a[i, 2]]), np.array([1.0, 0, 0]), "action frame"),
        ]):
            ax = fig.add_subplot(n, 2, 2 * i + col + 1, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=8, depthshade=False)
            ax.scatter(*star, c="red", s=90, marker="*", depthshade=False)
            ax.quiver(*(star - 0.035 * arrow), *(0.03 * arrow), color="red", linewidth=2, arrow_length_ratio=0.35)
            c = np.concatenate([pts, star[None]]).mean(0)
            half = max(0.05, np.abs(np.concatenate([pts, star[None]]) - c).max() + 0.01)
            ax.set_xlim(c[0] - half, c[0] + half), ax.set_ylim(c[1] - half, c[1] + half), ax.set_zlim(0, 2 * half)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=30, azim=-60 if col == 0 else -120)
            ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
            ax.tick_params(labelsize=6)
            ax.set_title(f"env {i} model input, {frame}\n{int(n_obs[i])} observed points -> {uniq} distinct "
                         f"of 512 slots (rest repeat)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
