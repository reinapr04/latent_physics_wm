"""Visualize PushTPhysics-v1's point cloud and the dynamics model's input.

One row per env:
  1. the camera image (RGB)
  2. the full observed point cloud in the crop, top view, coloured with its RGB
  3. the same, oblique 3D view (shows the T's thickness and the camera's view of its sides)
  4. the model input (dynamics_model.geometry.observed_cloud): T points (red) + table points
     (grey), with a push drawn on it: p (black dot) and d (arrow)
  5. the model input from the side, in that push's action frame - exactly what the model gets:
     x along the push direction (the push at x = 0), z up (height axis stretched)

Run from the repo root:
    python -m env.push_t.visualize_pointcloud                 # 3 envs -> env/push_t/examples/
    python -m env.push_t.visualize_pointcloud --num-envs 4 --seed 1 --n-in 512
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dynamics_model.geometry import TABLE_Z, _in_box, model_input, observed_cloud  # noqa: E402
from env.push_t import WORKSPACE, make_push_t_env  # noqa: E402
from policy.mpc import ActionSampler  # noqa: E402

warnings.filterwarnings("ignore")
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-in", type=int, default=1024)
    ap.add_argument("--table-points", type=int, default=100)
    ap.add_argument("--out", default=None, help="default: env/push_t/examples/pointcloud_seed<seed>.png")
    args = ap.parse_args()

    e = make_push_t_env(num_envs=args.num_envs, obs_mode="pointcloud")
    obs, _ = e.reset(seed=args.seed, options={"reconfigure": True})
    # the camera image comes from a separate rgbd env with the same seed (same T placement)
    e_rgb = make_push_t_env(num_envs=args.num_envs, obs_mode="rgbd")
    rgb_obs, _ = e_rgb.reset(seed=args.seed, options={"reconfigure": True})

    pc = obs["pointcloud"]
    xyz, rgb = pc["xyzw"][..., :3], pc["rgb"].float() / 255
    in_ws = (pc["xyzw"][..., 3] > 0) & _in_box(xyz, *WORKSPACE)
    cloud, count = observed_cloud(obs, args.n_in, args.table_points, WORKSPACE)
    torch.manual_seed(args.seed)
    # an aimed push with push_t's ranges (train_dynamics.TASK_DEFAULTS["push_t"])
    push = ActionSampler(uniform_fraction=0.0, height=(0.01, 0.035), force=(1.0, 25.0), duration=(0.2, 0.6))(cloud, 1)[:, 0]

    n = args.num_envs
    fig = plt.figure(figsize=(26, 5 * n))
    pts_frame = model_input(cloud, push)  # (n, N, 3) in each push's action frame
    lo, hi = WORKSPACE
    for i in range(n):
        ax = fig.add_subplot(n, 5, 5 * i + 1)
        ax.imshow(rgb_obs["sensor_data"]["base_camera"]["rgb"][i].cpu().numpy())
        ax.set_title(f"env {i}: camera (256x256)")
        ax.axis("off")

        m = in_ws[i]
        p, c = xyz[i][m].cpu().numpy(), rgb[i][m].cpu().numpy()
        ax = fig.add_subplot(n, 5, 5 * i + 2)
        ax.scatter(p[:, 0], p[:, 1], s=0.5, c=c)
        ax.set_title(f"observed cloud in the crop, top view ({m.sum().item()} points)")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

        ax = fig.add_subplot(n, 5, 5 * i + 3, projection="3d")
        raised = p[:, 2] > TABLE_Z
        sub = torch.randperm(len(p))[:15000].numpy()  # thin the table for the 3D plot
        keep = raised.copy(); keep[sub] = True
        ax.scatter(p[keep, 0], p[keep, 1], p[keep, 2], s=0.5, c=c[keep])
        ax.set_title("observed cloud, 3D (table thinned)")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(0, 0.15)
        ax.view_init(elev=35, azim=-60)

        q = cloud[i].cpu().numpy()
        t_mask = q[:, 2] > TABLE_Z
        ax = fig.add_subplot(n, 5, 5 * i + 4)
        ax.scatter(q[~t_mask, 0], q[~t_mask, 1], s=8, c="0.6", label=f"table ({(~t_mask).sum()})")
        ax.scatter(q[t_mask, 0], q[t_mask, 1], s=1.5, c="tab:red", label=f"T ({min(int(count[i]), args.n_in - args.table_points)} of {int(count[i])} seen)")
        a = push[i].cpu().numpy()
        ax.plot(a[0], a[1], "ko", ms=5, label=f"push p (F {a[5]:.0f} N, T {a[6]:.2f} s)")
        ax.arrow(a[0], a[1], 0.08 * a[3], 0.08 * a[4], width=0.004, color="k")
        ax.set_title(f"model input: {args.n_in} points + a sampled push")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=8)

        f = pts_frame[i].cpu().numpy()
        t_f = f[:, 2] > TABLE_Z
        ax = fig.add_subplot(n, 5, 5 * i + 5)
        ax.scatter(f[~t_f, 0], f[~t_f, 2], s=8, c="0.6", label="table")
        ax.scatter(f[t_f, 0], f[t_f, 2], s=1.5, c="tab:red", label="T")
        ax.plot(0, a[2], "ko", ms=5, label=f"p: push height {100 * a[2]:.1f} cm")
        ax.arrow(0, a[2], 0.06, 0, width=0.0008, head_length=0.015, color="k")
        ax.set_title("model input, side view (action frame)")
        ax.set_xlabel("x along the push direction (m)"); ax.set_ylabel("z (m)")
        ax.set_xlim(-0.35, 0.45); ax.set_ylim(-0.01, 0.06)
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = Path(args.out) if args.out else EXAMPLES_DIR / f"pointcloud_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=60)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
