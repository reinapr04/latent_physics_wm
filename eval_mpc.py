"""Goal reaching with MPC on physics settings never seen in training.

For each new environment (one unseen physics setting each, all run in parallel):
  1. probe: a few informative pushes (policy.mpc.ProbeSampler)
  2. fit the latent on them, dynamics model frozen (dynamics_model.latent_inference)
  3. reset the block, pick a random goal, and push it there with MPC planning with that latent:
     every push, sample candidate pushes, predict each outcome with the model, score it with
     the env's reward (env.reward.block_reward), execute the best one, observe, replan.

Step 3 is repeated with each latent condition, from the same start and goal:
  mean      mean of the training latents (no adaptation)
  fitted    fitted on the first --probe-pushes probes
  fitted_N  fitted on all --oracle-pushes probes (more information; a best case for fitting)
If "fitted" reaches goals more accurately than "mean", adapting the latent pays off in control.

The model is a single checkpoint (train_dynamics.py / compare_encoders output) or an ensemble
directory (scripts/train_ensemble.py, member*/model.pt): the ensemble predicts the mean of its
members' predictions, for latent fitting and for planning alike.

With --video-envs K (default 16), the first K envs' episodes are rendered to one video per
condition (<out dir>/mpc_<condition>.mp4, a grid of tiles). Each tile shows the env's
physics and, for the push being shown, "push k: D -> pred P / got G cm (moved M)": distance
to the goal center (red/white target) before the push, the distance the model predicted the
push would leave, the distance it actually left, and how far the block actually moved.

    python eval_mpc.py --checkpoint runs/ensemble
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from mani_skill.utils import sapien_utils

from dynamics_model.data import TransitionBuffer
from dynamics_model.geometry import observed_cloud
from dynamics_model.latent_inference import infer_latents
from dynamics_model.model import DynamicsModel, Ensemble
from env import PushPrimitiveConfig, get_task, make_push_env
from env.demo_push_primitive import _font, annotate, grid
from env.observation import OBS_MODE, split_obs
from policy.mpc import ActionSampler, ProbeSampler, block_position, imagine_next_cloud, random_shooting_mpc
from train_dynamics import TEST_SEED, Config, make_physics_pool, run_rollout, task_defaults

warnings.filterwarnings("ignore")

TILE = 256  # px per env in the videos
# goals stay on the visible part of the table (world xy, m)
GOAL_LOW = (-0.15, -0.25)
GOAL_HIGH = (0.30, 0.25)


def load_model(path, dev):
    """-> model (DynamicsModel or Ensemble), mean training latent (L,) or (members*L,), files, task name."""
    path = Path(path)
    files = sorted(path.glob("member*/model.pt")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no checkpoint at {path}")
    members, z_means, task = [], [], None
    for f in files:
        ck = torch.load(f, map_location=dev, weights_only=False)
        m = DynamicsModel.from_checkpoint(ck, dev)
        members.append(m.eval())
        z_means.append(ck["latents"]["emb.weight"][:, : m.latent_dim].mean(0))
        task = task or ck.get("task") or ck.get("config", {}).get("task")
    model = members[0] if len(members) == 1 else Ensemble(members).eval()
    return model, torch.cat(z_means), [str(f) for f in files], task or "block"


def random_goals(block_xy, dist_range, gen):
    n = len(block_xy)
    ang = torch.rand(n, generator=gen) * 2 * math.pi
    dist = dist_range[0] + torch.rand(n, generator=gen) * (dist_range[1] - dist_range[0])
    g = block_xy.cpu() + dist[:, None] * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return torch.stack([g[:, 0].clamp(*[GOAL_LOW[0], GOAL_HIGH[0]]), g[:, 1].clamp(GOAL_LOW[1], GOAL_HIGH[1])], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/ensemble", help="checkpoint file or ensemble directory")
    ap.add_argument("--num-envs", type=int, default=32, help="new environments (one unseen physics setting each)")
    ap.add_argument("--seed", type=int, default=TEST_SEED, help="seed of the test settings (training never uses it)")
    ap.add_argument("--probe-pushes", type=int, default=4, help="probes the 'fitted' latent is fitted on")
    ap.add_argument("--oracle-pushes", type=int, default=16, help="probes for 'fitted_N' (0 = skip)")
    ap.add_argument("--fit-steps", type=int, default=200)
    ap.add_argument("--max-pushes", type=int, default=10, help="MPC pushes per episode")
    ap.add_argument("--samples", type=int, default=1024, help="candidate pushes per MPC step")
    ap.add_argument("--horizon", type=int, default=1, help="pushes planned ahead")
    ap.add_argument("--goal-dist", type=float, nargs=2, default=None, help="m, start-to-goal distance (default: the task's)")
    ap.add_argument("--task", default=None, help="block | ball (default: read from the checkpoint)")
    ap.add_argument("--success-radius", type=float, default=0.02,
                    help="m, done when the block is this close; also the planner's success radius")
    ap.add_argument("--uniform-fraction", type=float, default=0.1, help="share of MPC candidates placed anywhere")
    ap.add_argument("--conditions", nargs="+", default=["mean", "fitted", "fitted_N"])
    ap.add_argument("--video-envs", type=int, default=16, help="envs rendered to video per condition (0 = no video)")
    ap.add_argument("--video-stride", type=int, default=2, help="keep every n-th control step in the video")
    ap.add_argument("--out", default=None, help="default: <checkpoint dir>/mpc_eval.json")
    args = ap.parse_args()
    t_start = time.time()
    ck = Path(args.checkpoint)
    out = Path(args.out) if args.out else (ck if ck.is_dir() else ck.parent) / "mpc_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    model, z_mean, files, ck_task = load_model(args.checkpoint, "cpu")
    task = get_task(args.task or ck_task)
    if args.goal_dist is None:
        args.goal_dist = list(task.goal_dist)
    cfg = Config(seed=args.seed, num_envs=args.num_envs, pool_size=args.num_envs, task=task.name, **task_defaults(task))
    pool, pool_t = make_physics_pool(cfg)  # same ranges as training, unseen seed
    n_vid = min(args.video_envs, args.num_envs)
    render_kw = dict(render_mode="rgb_array", human_render_camera_configs=dict(
        width=TILE, height=TILE, pose=sapien_utils.look_at([0.45, 0.45, 0.6], [0.05, 0.0, 0.0]))) if n_vid else {}
    env = make_push_env(num_envs=cfg.num_envs, obs_mode=OBS_MODE, task=task,
                        push_config=PushPrimitiveConfig(phase_timeout=2.0, record_envs=n_vid), **render_kw)
    u = env.unwrapped
    dev = u.device
    model, z_mean = model.to(dev), z_mean.to(dev)
    print(f"task {task.name}; model: {len(files)} checkpoint(s) "
          f"{'(ensemble, mean prediction)' if len(files) > 1 else ''}, input {model.input_kind}", flush=True)
    n = cfg.num_envs
    # image-based models (tokens, crops, fusion) compute their inputs from the camera image online
    from dynamics_model.inputs import attach_inputs, camera_from_obs, model_inputs, planner_inputs
    image_based = model.input_kind == "image" or bool(model.aux_key)
    cam = None

    # ---------------------------------------------------------------- 1. probe
    n_probe = max(args.probe_pushes, args.oracle_pushes if "fitted_N" in args.conditions else 0)
    print(f"probing: {n_probe} pushes in each of {n} new environments ...", flush=True)
    buf = TransitionBuffer(val_fraction=0.0, seed=args.seed)
    probe = ProbeSampler(shape=task.shape, height=task.probe_height, force=(0.3, cfg.max_force),
                         duration=(cfg.min_duration + 0.1, cfg.max_duration))
    cfg.pushes_per_rollout = n_probe
    stats = run_rollout(env, cfg, pool, pool_t, np.arange(n), lambda c, i, g, *_: probe(c, 1)[:, 0], buf, 0, task)
    if image_based:
        cam = camera_from_obs(env.unwrapped.get_obs())
        attach_inputs(buf, model, cam, task.half_size, dev)
    data = buf.tensors(dev)
    env_of = data["setting"]
    print(f"kept {len(env_of)} probe transitions (contact rate {stats['contact_rate']:.2f})", flush=True)

    # ---------------------------------------------------------------- 2. fit latents
    def fit(k):
        """Latent per env from its first k kept probes (envs without any keep the mean latent)."""
        idx = torch.cat([(env_of == e).nonzero().flatten()[:k] for e in range(n)])
        keys = [model.input_key, "act_feat", "pts_loss", "R_gt", "t_gt", "action"] + ([model.aux_key] if model.aux_key else [])
        sub = {key: data[key][idx] for key in keys}
        z = infer_latents(model, sub, env_of[idx], n, z_mean, steps=args.fit_steps, lr=cfg.latent_lr, l2=cfg.latent_l2)
        has = torch.bincount(env_of[idx], minlength=n) > 0
        return torch.where(has[:, None], z, z_mean.expand(n, -1))

    latents = {}
    for c in args.conditions:
        if c == "mean":
            latents[c] = z_mean.expand(n, -1).clone()
        elif c == "fitted":
            latents[c] = fit(args.probe_pushes)
        elif c == "fitted_N":
            latents[c] = fit(args.oracle_pushes)
        else:
            raise ValueError(f"unknown condition {c}")

    # ---------------------------------------------------------------- 3. MPC episodes
    env.cfg.record_frames = bool(n_vid)  # frames for the episodes only, not the probes
    sampler = ActionSampler(uniform_fraction=args.uniform_fraction, force=(cfg.min_force, cfg.max_force),
                            duration=(cfg.min_duration, cfg.max_duration), height=task.push_height, shape=task.shape)
    # a push in the air above the far corner: keeps finished envs still while the others push
    idle = torch.tensor([-0.25, 0.35, 0.15, 0.0, 1.0, 0.2, 0.3], device=dev)
    # the same block start and goal for every condition (passed to reset explicitly)
    gen = torch.Generator().manual_seed(args.seed)
    start_xy = torch.rand(n, 2, generator=gen) * 0.2 - 0.1
    goal_cpu = random_goals(start_xy, args.goal_dist, gen)
    goal = goal_cpu.to(dev)
    phys = {k: v.cpu() for k, v in u.block_physics.items()}
    font = _font(10)
    results, curves, pred_errs = {}, {}, {}
    for c in args.conditions:
        torch.manual_seed(args.seed)  # same MPC randomness for every condition
        # no reconfigure: the physics stay
        obs, _ = env.reset(seed=args.seed, options={"block_xy": start_xy, "goal_xy": goal_cpu})
        assert torch.allclose(u.obj.pose.p[:, :2].cpu(), start_xy, atol=2e-3), "block start not applied"
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        pushes_used = torch.full((n,), args.max_pushes, device=dev)
        dist = (u.obj.pose.p[:, :2] - goal).norm(dim=1)
        curve, errs = [100 * dist.mean().item()], []
        writer, status = None, [("", "")] * n
        if n_vid:
            import imageio
            writer = imageio.get_writer(str(out.parent / f"mpc_{c}.mp4"), fps=25, macro_block_size=8)
        t0 = time.time()
        for step in range(args.max_pushes):
            obs, img = split_obs(obs, env)
            cloud, _ = observed_cloud(obs, cfg.n_in, cfg.table_points)
            inputs_fn = planner_inputs(model, img, cam, task.half_size, dev) if image_based else None
            a, _ = random_shooting_mpc(model, cloud, latents[c], goal, sampler, num_samples=args.samples,
                                       horizon=args.horizon, goal_radius=args.success_radius,
                                       half_size=task.half_size, inputs_fn=inputs_fn)
            with torch.no_grad():  # where the model expects this push to leave the object
                chosen = inputs_fn(torch.arange(n, device=dev), a) if inputs_fn else None
                pred_xy = block_position(imagine_next_cloud(model, cloud, a, latents[c], chosen), task.half_size)[:, :2]
            a = torch.where(done[:, None], idle.expand(n, -1), a)
            was_done = done.clone()
            p_before = u.obj.pose.p[:, :2].clone()
            dist_before = (p_before - goal).norm(dim=1)
            obs, _, _, _, info = env.step(a)
            p = u.obj.pose.p
            dist = (p[:, :2] - goal).norm(dim=1)
            off = p[:, 2] < 0
            active = ~was_done & ~off
            errs.append((pred_xy - p[:, :2]).norm(dim=1)[active])
            newly = ~done & ((dist < args.success_radius) | off)
            pushes_used[newly] = step + 1
            done |= newly
            curve.append(100 * dist[~off].mean().item() if (~off).any() else float("nan"))
            if writer is not None:
                pred_d = 100 * (pred_xy - goal).norm(dim=1)
                for i in range(n_vid):
                    if not was_done[i]:
                        moved = 100 * (p[i, :2] - p_before[i]).norm()
                        status[i] = (f"push {step + 1}: {100 * dist_before[i]:.1f} -> pred {pred_d[i]:.1f}"
                                     f" / got {100 * dist[i]:.1f}cm",
                                     f"block moved {moved:.1f}cm"
                                     + ("  DONE" if dist[i] < args.success_radius else "")
                                     + ("  OFF TABLE" if off[i] else ""))
                frames = info["push"]["frames"]  # (steps, n, H, W, 3)
                for s in range(0, frames.shape[0], args.video_stride):
                    tiles = [annotate(frames[s, i].numpy(), [
                        f"env{i} {c}  {phys['mass'][i]:.2f}kg mu {phys['static_friction'][i]:.2f}"
                        f"/{phys['dynamic_friction'][i]:.2f}"
                        + (f" damp {phys['linear_damping'][i]:.1f}/{phys['angular_damping'][i]:.1f}" if task.name == "ball" else ""),
                        status[i][0]], status[i][1], font)
                        for i in range(n_vid)]
                    writer.append_data(grid(tiles, math.ceil(math.sqrt(n_vid))))
            if done.all():
                break
        if writer is not None:
            writer.close()
        pred_errs[c] = torch.cat(errs)
        p = u.obj.pose.p
        dist = (p[:, :2] - goal).norm(dim=1)
        on = p[:, 2] > 0
        reached = (dist < args.success_radius) & on
        results[c] = {
            "success": reached.float().mean().item(),
            "success_5cm": ((dist < 0.05) & on).float().mean().item(),
            "final_dist_cm_mean": 100 * dist[on].mean().item(),
            "final_dist_cm_median": 100 * dist[on].median().item(),
            "pushes_to_success": pushes_used[reached].float().mean().item() if reached.any() else float("nan"),
            "off_table": (~on).float().mean().item(),
            "prediction_err_cm": 100 * pred_errs[c].mean().item(),  # predicted vs actual block xy, per push
        }
        curves[c] = curve
        print(f"  {c}: {time.time() - t0:.0f} s, success {results[c]['success']:.0%}", flush=True)
    env.close()

    d0 = 100 * (start_xy - goal_cpu).norm(dim=1)
    print(f"\n{n} envs, start-to-goal {d0.mean():.1f} cm (mean), up to {args.max_pushes} pushes, "
          f"success = within {100 * args.success_radius:.0f} cm")
    print(f"{'latent':10s} {'success':>8s} {'<5cm':>6s} {'final dist (cm)':>16s} {'pushes':>7s} {'off table':>10s}"
          f" {'pred err (cm)':>14s}")
    for c, r in results.items():
        print(f"{c:10s} {r['success']:8.0%} {r['success_5cm']:6.0%} "
              f"{r['final_dist_cm_mean']:7.1f} (med {r['final_dist_cm_median']:4.1f}) "
              f"{r['pushes_to_success']:7.1f} {r['off_table']:10.0%} {r['prediction_err_cm']:14.2f}")
    print("\nmean distance to goal (cm) after each push:")
    for c, cu in curves.items():
        print(f"{c:10s} " + " ".join(f"{x:5.1f}" for x in cu))
    if n_vid:
        print("\nvideos: " + ", ".join(str(out.parent / f"mpc_{c}.mp4") for c in args.conditions))

    out.write_text(json.dumps({"args": vars(args), "task": task.name, "checkpoints": files, "start_to_goal_cm": d0.mean().item(),
                               "results": results, "distance_curves_cm": curves}, indent=2))
    print(f"\nsaved {out}  ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
