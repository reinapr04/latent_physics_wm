"""What do random pushes do to the object? Quick statistics for tuning a task's push and
physics ranges (env/tasks.py): contact / collision / off-table rates, how far the object moves,
how long it keeps moving after the primitive (ball), by push force.

    python scripts/push_stats.py --task ball --num-envs 32 --pushes 4
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import train_dynamics as T  # noqa: E402
from dynamics_model.geometry import observed_cloud  # noqa: E402
from env import PushPrimitiveConfig, get_task, make_push_env  # noqa: E402
from env.observation import OBS_MODE, split_obs  # noqa: E402
from policy.mpc import ActionSampler  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="ball")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--pushes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform-fraction", type=float, default=0.0, help="share of pushes placed anywhere (misses)")
    ap.add_argument("--force", type=float, nargs=2, default=None)
    ap.add_argument("--duration", type=float, nargs=2, default=None)
    ap.add_argument("--angular-damping", type=float, nargs=2, default=None)
    ap.add_argument("--linear-damping", type=float, nargs=2, default=None)
    args = ap.parse_args()
    task = get_task(args.task)
    cfg = T.Config(task=task.name, seed=args.seed, num_envs=args.num_envs, pool_size=args.num_envs, **T.task_defaults(task))
    if args.force:
        cfg.min_force, cfg.max_force = args.force
    if args.duration:
        cfg.min_duration, cfg.max_duration = args.duration
    if args.angular_damping:
        cfg.angular_damping = tuple(args.angular_damping)
    if args.linear_damping:
        cfg.linear_damping = tuple(args.linear_damping)
    pool, pool_t = T.make_physics_pool(cfg)
    env = make_push_env(num_envs=cfg.num_envs, obs_mode=OBS_MODE, task=task, push_config=PushPrimitiveConfig(phase_timeout=2.0))
    u = env.unwrapped
    sampler = ActionSampler(uniform_fraction=args.uniform_fraction, force=(cfg.min_force, cfg.max_force),
                            duration=(cfg.min_duration, cfg.max_duration), height=task.push_height, shape=task.shape)
    u.set_physics(pool)
    obs, _ = env.reset(seed=args.seed, options={"reconfigure": True})
    rows = []
    t0 = time.time()
    for k in range(args.pushes):
        obs, img = split_obs(obs, env)
        cloud, n_obs = observed_cloud(obs, cfg.n_in, cfg.table_points)
        a = sampler(cloud, 1)[:, 0]
        p0 = u.obj.pose.p.clone()
        t1 = time.time()
        obs, _, _, _, info = env.step(a)
        p = info["push"]
        moved = (u.obj.pose.p[:, :2] - p0[:, :2]).norm(dim=1)
        for i in range(cfg.num_envs):
            rows.append(dict(force=a[i, 5].item(), duration=a[i, 6].item(), height=a[i, 2].item(),
                             contact=bool(p["made_contact"][i]), collided=bool(p["collided"][i]),
                             moved_cm=100 * moved[i].item(), off=bool(u.obj.pose.p[i, 2] < 0),
                             visible=int(n_obs[i]) >= 10, settle=p["settle_time"][i].item(),
                             final_speed=p["final_speed"][i].item(), mean_force=p["mean_push_force"][i].item(),
                             push_time=p["push_time"][i].item(), mass=pool[i]["mass"],
                             ang=pool[i]["angular_damping"], lin=pool[i]["linear_damping"]))
        print(f"push {k + 1}: {time.time() - t1:.1f} s wall, {p['phase'].shape[0]} control steps", flush=True)
    env.close()
    r = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    print(f"\n{task.name}: {len(rows)} pushes in {time.time() - t0:.0f} s; force {cfg.min_force}-{cfg.max_force} N, "
          f"duration {cfg.min_duration}-{cfg.max_duration} s, height {task.push_height}")
    print(f"contact {r['contact'].mean():.0%}  collided {r['collided'].mean():.0%}  off table {r['off'].mean():.0%}  "
          f"object visible before push {r['visible'].mean():.0%}")
    c = r["contact"] & ~r["off"]
    q = np.percentile(r["moved_cm"][c], [10, 25, 50, 75, 90]) if c.any() else np.zeros(5)
    print(f"moved (contact, cm) p10/25/50/75/90: " + " ".join(f"{x:5.1f}" for x in q) + f"   > 30 cm: {(r['moved_cm'][c] > 30).mean():.0%}")
    print(f"settle time (s) mean {r['settle'].mean():.2f} max {r['settle'].max():.2f}; final speed mean {r['final_speed'].mean():.3f} "
          f"max {r['final_speed'].max():.3f} m/s; mean push force {r['mean_force'][c].mean():.2f} N; push time {r['push_time'][c].mean():.2f} s")
    print("\nby force bin (contact pushes): force      n   moved p50   p90   >30cm")
    edges = np.linspace(cfg.min_force, cfg.max_force, 5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = c & (r["force"] >= lo) & (r["force"] < hi)
        if m.any():
            print(f"   {lo:4.1f}-{hi:4.1f} N  {m.sum():4d}   {np.median(r['moved_cm'][m]):6.1f} {np.percentile(r['moved_cm'][m], 90):6.1f}"
                  f"  {(r['moved_cm'][m] > 30).mean():5.0%}")
    if task.name == "ball":
        print("\nby angular damping (contact pushes): damping     n   moved p50")
        e = np.linspace(cfg.angular_damping[0], cfg.angular_damping[1] + 1e-6, 4)
        for lo, hi in zip(e[:-1], e[1:]):
            m = c & (r["ang"] >= lo) & (r["ang"] < hi)
            if m.any():
                print(f"   {lo:4.1f}-{hi:4.1f}     {m.sum():4d}   {np.median(r['moved_cm'][m]):6.1f}")


if __name__ == "__main__":
    main()
