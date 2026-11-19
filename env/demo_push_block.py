"""Sanity-check PushBlock-v1 physics: kick each block and compare its slide distance
with the analytic value d = v^2 / (2 * mu_eff * g), where mu_eff is the average of
block and table dynamic friction (PhysX's default combine mode).

Run from the repo root:
    python -m env.demo_push_block                # GPU, 8 parallel envs
    python -m env.demo_push_block --backend cpu  # CPU, 1 env
    python -m env.demo_push_block --video out.mp4
"""

import argparse
import warnings

import gymnasium as gym
import torch

import env  # noqa: F401  registers PushBlock-v1
from env import BlockPhysicsConfig

warnings.filterwarnings("ignore")

TABLE_MU = 0.5
KICK = 0.5  # m/s, initial block speed along +y


def slide_test(e, label):
    u = e.unwrapped
    n = u.num_envs
    zero = torch.zeros(e.action_space.shape)
    p0 = u.obj.pose.p.clone()
    u.obj.set_linear_velocity(torch.tensor([[0.0, KICK, 0.0]] * n, device=u.device))
    if u.gpu_sim_enabled:
        u.scene._gpu_apply_all()
    for _ in range(40):
        e.step(zero)
    dist = (u.obj.pose.p - p0)[:, 1].cpu()
    phys = {k: v.cpu() for k, v in u.block_physics.items()}
    mu_eff = (phys["dynamic_friction"] + TABLE_MU) / 2
    expected = KICK**2 / (2 * mu_eff * 9.81)
    print(f"\n[{label}]  env | mass   mu_s   mu_d  | slide  expected")
    for i in range(n):
        print(
            f"  {i:3d} | {phys['mass'][i]:.3f}  {phys['static_friction'][i]:.3f}  {phys['dynamic_friction'][i]:.3f}"
            f" | {dist[i]:.3f}  {expected[i]:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--video", default=None, help="save an mp4 of a random-action rollout")
    args = ap.parse_args()
    n = 1 if args.backend == "cpu" else args.num_envs

    physics = BlockPhysicsConfig(
        mass=(0.05, 2.0),
        static_friction=(0.1, 1.0),
        dynamic_friction=(0.1, 1.0),
        table_static_friction=TABLE_MU,
        table_dynamic_friction=TABLE_MU,
    )
    e = gym.make(
        "PushBlock-v1",
        num_envs=n,
        sim_backend=args.backend,
        obs_mode="state",
        physics=physics,
        render_mode="rgb_array" if args.video else None,
    )
    e.reset(seed=0)
    slide_test(e, "seed 0")
    e.reset(options={"reconfigure": True})  # resample physics
    slide_test(e, "after reconfigure")

    if args.video:
        import imageio

        e.reset(seed=1)
        frames = []
        for _ in range(50):
            e.step(e.action_space.sample())
            frames.append(e.render()[0].cpu().numpy())
        imageio.mimsave(args.video, frames, fps=20)
        print(f"\nsaved {args.video}")
    e.close()


if __name__ == "__main__":
    main()
