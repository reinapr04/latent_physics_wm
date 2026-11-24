"""Visual check of the push primitive: N envs with random block physics, each doing a random push,
rendered side by side in one video with each env's settings drawn on top of its tile. Also saves
what the policy camera sees (RGB + depth) at reset and after each push.

Run from the repo root:
    python -m env.demo_push_primitive                         # 4 envs, 1 push each -> env/examples/
    python -m env.demo_push_primitive --pushes 3              # 3 consecutive pushes per env
    python -m env.demo_push_primitive --num-envs 9 --seed 1 --video other.mp4
"""

import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mani_skill.utils import sapien_utils

from env import BlockPhysicsConfig, PushPrimitiveConfig, get_task, make_push_env
from env.push_primitive import PHASE_NAMES, PUSH

warnings.filterwarnings("ignore")

TILE = 384  # px per env view
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


def random_push(e, max_force, gen, task=None):
    """A random push on a random point of the object's side, roughly towards its centre.

    The push is described relative to the object and converted to the world-frame action using
    the simulator's object pose (a policy would instead pick p_contact from the point cloud)."""
    task = get_task(task or "block")
    u = e.unwrapped
    n = u.num_envs
    ang = torch.rand(n, generator=gen) * 2 * math.pi
    d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    if task.shape == "sphere":
        c = u.obj.pose.p.cpu()
        lateral = (torch.rand(n, generator=gen) - 0.5) * task.half_size
        perp = torch.stack([-d[:, 1], d[:, 0]], 1)
        half_chord = (task.half_size ** 2 - lateral ** 2).sqrt()
        p_xy = c[:, :2] + lateral[:, None] * perp - half_chord[:, None] * d
        z = task.push_height[0] + torch.rand(n, generator=gen) * (task.push_height[1] - task.push_height[0])
        p = torch.cat([p_xy, z[:, None]], 1)
    else:
        local = torch.zeros((n, 3))
        # the face opposite the push, with a random offset along it, at a random height
        local[:, :2] = (-d + 0.5 * (torch.rand((n, 2), generator=gen) - 0.5)).clamp(-1, 1)
        local[:, 2] = torch.rand(n, generator=gen) * 2 - 1
        p = e.block_frame_to_world(local, d).cpu()
    force = task.force[0] + torch.rand(n, generator=gen) * (min(max_force, task.force[1]) - task.force[0])
    duration = task.duration[0] + torch.rand(n, generator=gen) * (task.duration[1] - task.duration[0])
    return torch.cat([p, d, force[:, None], duration[:, None]], dim=1)


def camera_row(obs):
    """Policy camera view of every env: RGB on top, depth (table range) below."""
    sd = obs["sensor_data"]["base_camera"]
    rgb = sd["rgb"].cpu().numpy()
    depth = sd["depth"].cpu().numpy()[..., 0].astype(np.float32)  # mm
    dv = (255 * (1 - np.clip((depth - 500) / 600, 0, 1))).astype(np.uint8)
    return np.concatenate(
        [np.concatenate([rgb[i], np.repeat(dv[i][..., None], 3, -1)], 0) for i in range(len(rgb))], 1
    )


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate(frame, header, live, font):
    """Draw a translucent band with text lines at the top of one env's frame."""
    img = Image.fromarray(frame).convert("RGBA")
    lines = header + [live]
    line_h = font.size + 3
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(band).rectangle([0, 0, img.width, 6 + line_h * len(lines)], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img, band)
    draw = ImageDraw.Draw(img)
    for k, text in enumerate(lines):
        color = (255, 230, 120) if k == len(lines) - 1 else (255, 255, 255)
        draw.text((6, 4 + k * line_h), text, font=font, fill=color)
    return np.asarray(img.convert("RGB"))


def grid(tiles, cols):
    rows = math.ceil(len(tiles) / cols)
    h, w, _ = tiles[0].shape
    out = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--pushes", type=int, default=1, help="consecutive pushes per env")
    ap.add_argument("--max-force", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None,
                    help="output path (default: env/examples/push_demo_<task>_seed<seed>.mp4)")
    ap.add_argument("--task", default="block", help="block | ball (env/tasks.py)")
    args = ap.parse_args()
    n = 1 if args.backend == "cpu" else args.num_envs
    task = get_task(args.task)

    e = make_push_env(
        num_envs=n,
        task=task,
        sim_backend=args.backend,
        obs_mode="rgbd",
        physics=BlockPhysicsConfig(
            mass=task.mass,
            static_friction=task.static_friction,
            dynamic_friction=task.static_friction,
            linear_damping=task.linear_damping if task.linear_damping[1] > 0 else 0.0,
            angular_damping=task.angular_damping if task.angular_damping[1] > 0 else 0.0,
        ),
        push_config=PushPrimitiveConfig(record_frames=True),
        render_mode="rgb_array",
        human_render_camera_configs=dict(
            width=TILE, height=TILE, pose=sapien_utils.look_at([0.45, 0.55, 0.75], [-0.05, 0.0, 0.0])
        ),
    )
    gen = torch.Generator().manual_seed(args.seed)
    obs, info = e.reset(seed=args.seed)
    phys = {k: v.cpu() for k, v in info["physics"].items()}
    font = _font(12)
    cols = math.ceil(math.sqrt(n))
    video, t_sim = [], 0.0
    obs_rows = [camera_row(obs)]

    for k in range(args.pushes):
        a = random_push(e, args.max_force, gen, task)
        t = time.time()
        obs, reward, terminated, truncated, info = e.step(a)
        obs_rows.append(camera_row(obs))
        p = {key: v.cpu() if torch.is_tensor(v) else v for key, v in info["push"].items()}
        steps = p["phase"].shape[0]
        print(f"\npush {k + 1}/{args.pushes}: {steps} control steps in {time.time() - t:.1f} s wall")
        print("  env | mass  mu_s  mu_d | p_contact world (cm)   dir    F_des  T    | "
              "touched  F_mean  t_push  moved")

        headers = []
        for i in range(n):
            ang = math.degrees(math.atan2(a[i, 4], a[i, 3]))
            pc = 100 * a[i, :3]
            if p["collided"][i]:
                actual = "bumped block, aborted"
            elif p["made_contact"][i]:
                ac = 100 * p["contact_point"][i]
                actual = f"touched ({ac[0]:+.1f},{ac[1]:+.1f},{ac[2]:.1f})cm"
            else:
                actual = "no contact"
            headers.append([
                f"env {i}  push {k + 1}/{args.pushes}",
                f"mass {phys['mass'][i]:.2f}kg  mu_s {phys['static_friction'][i]:.2f}  "
                f"mu_d {phys['dynamic_friction'][i]:.2f}"
                + (f"  damp {phys['linear_damping'][i]:.1f}/{phys['angular_damping'][i]:.1f}" if task.shape == "sphere" else ""),
                f"p ({pc[0]:+.1f},{pc[1]:+.1f},{pc[2]:.1f})cm  {actual}",
                f"dir {ang:+.0f}deg  F {a[i, 5]:.2f}N  T {a[i, 6]:.2f}s",
            ])
            print(f"  {i:3d} | {phys['mass'][i]:.2f}  {phys['static_friction'][i]:.2f}  "
                  f"{phys['dynamic_friction'][i]:.2f} | "
                  f"({pc[0]:+5.1f},{pc[1]:+5.1f},{pc[2]:4.1f})  {ang:+5.0f}  {a[i, 5]:.2f}  {a[i, 6]:.2f} | "
                  f"{str(bool(p['made_contact'][i])):7s}  {p['mean_push_force'][i]:.2f}    "
                  f"{p['push_time'][i]:.2f}    {p['block_displacement'][i, :2].norm():.3f}m")

        frames = p["frames"].numpy()  # (steps, n, H, W, 3)
        for s in range(steps):
            t_sim += p["dt"]
            tiles = []
            for i in range(n):
                ph = int(p["phase"][s, i])
                live = f"t={t_sim:5.2f}s  {PHASE_NAMES[ph]:8s}"
                if ph == PUSH or p["force_along_dir"][s, i] > 0.01:
                    live += f"  F_meas {p['force_along_dir'][s, i]:.2f}N"
                tiles.append(annotate(frames[s, i], headers[i], live, font))
            video.append(grid(tiles, cols))

    import imageio

    if args.video is None:
        args.video = str(EXAMPLES_DIR / f"push_demo_{task.name}_seed{args.seed}.mp4")
    Path(args.video).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.video, video, fps=int(round(1 / info["push"]["dt"])), macro_block_size=8)
    print(f"\nsaved {args.video} ({len(video)} frames)")
    obs_path = Path(args.video).with_name(Path(args.video).stem.replace("push_demo", "policy_obs") + ".png")
    imageio.imwrite(obs_path, np.concatenate(obs_rows, 0))
    print(f"saved {obs_path} (policy camera RGB + depth; rows: reset, then after each push)")
    e.close()


if __name__ == "__main__":
    main()
