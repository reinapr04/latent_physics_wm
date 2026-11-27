"""Sanity check for the off-centre centre of mass: push blocks straight at the middle of a face
and check that they turn, and in opposite directions for opposite offsets.

Every env gets the same mass and friction; only the COM differs (in the block frame). Each push
goes along the block's +x axis through the middle of its -x face, so a uniform block (or a COM
offset along x) should slide straight, and a COM offset along y should make it yaw.

Run from the repo root:
    python -m env.demo_com_offset                  # offsets +-1.5 cm and +-1.8 cm
    python -m env.demo_com_offset --video runs/com_sanity.mp4
"""

import argparse
import math
import warnings

import numpy as np
import torch
from mani_skill.utils import sapien_utils

from env import PushPrimitiveConfig, make_push_env
from env.demo_push_primitive import _font, annotate, grid
from env.push_primitive import _yaw

warnings.filterwarnings("ignore")

# (com_x, com_y) in m, block frame
OFFSETS = [(0.0, 0.0), (0.0, 0.015), (0.0, -0.015), (0.0, 0.018), (0.0, -0.018), (0.015, 0.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pushes", type=int, default=3)
    ap.add_argument("--mass", type=float, default=0.3)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--force", type=float, default=3.0)
    ap.add_argument("--duration", type=float, default=0.6)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    n = len(OFFSETS)
    render_kw = dict(render_mode="rgb_array", human_render_camera_configs=dict(
        width=256, height=256, pose=sapien_utils.look_at([0.45, 0.45, 0.6], [0.05, 0.0, 0.0]))) if args.video else {}
    e = make_push_env(num_envs=n, obs_mode="state",
                      push_config=PushPrimitiveConfig(phase_timeout=2.0, record_frames=bool(args.video)), **render_kw)
    u = e.unwrapped
    u.set_physics([{"mass": args.mass, "static_friction": args.friction, "dynamic_friction": args.friction,
                    "com_x": x, "com_y": y} for x, y in OFFSETS])
    e.reset(seed=0, options={"reconfigure": True, "block_xy": torch.full((n, 2), -0.05).numpy()})
    frames, font = [], _font(10)
    print(f"{'com (cm)':>14s} " + " ".join(f"{'push ' + str(k + 1) + ' dyaw/moved':>20s}" for k in range(args.pushes)))
    rows = [[] for _ in range(n)]
    for k in range(args.pushes):
        yaw0, p0 = _yaw(u.obj.pose.q).clone(), u.obj.pose.p[:, :2].clone()
        d = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], 1)  # the block's own +x axis
        local = torch.tensor([[-1.0, 0.0, 0.0]]).repeat(n, 1)  # middle of the -x face, mid height
        p = e.block_frame_to_world(local, d)
        a = torch.cat([p, d, torch.full((n, 1), args.force, device=p.device),
                       torch.full((n, 1), args.duration, device=p.device)], 1)
        _, _, _, _, info = e.step(a)
        dyaw = (_yaw(u.obj.pose.q) - yaw0 + math.pi) % (2 * math.pi) - math.pi
        moved = (u.obj.pose.p[:, :2] - p0).norm(dim=1)
        for i in range(n):
            rows[i].append(f"{math.degrees(dyaw[i]):+7.1f}deg/{100 * moved[i]:4.1f}cm")
        if args.video:
            f = info["push"]["frames"]
            for s in range(0, f.shape[0], 2):
                frames.append(grid([annotate(f[s, i].numpy(), [f"com {100 * OFFSETS[i][0]:+.1f},{100 * OFFSETS[i][1]:+.1f} cm",
                                                               f"push {k + 1}"], "", font) for i in range(n)], 3))
    for (x, y), r in zip(OFFSETS, rows):
        print(f"{100 * x:+6.1f},{100 * y:+6.1f}  " + " ".join(f"{c:>20s}" for c in r))
    if args.video:
        import imageio
        imageio.mimsave(args.video, frames, fps=25, macro_block_size=8)
        print(f"saved {args.video}")
    e.close()


if __name__ == "__main__":
    main()
