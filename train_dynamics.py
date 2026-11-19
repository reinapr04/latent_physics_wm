"""Model-based RL with on-policy data (Algorithm 1) for learning the latent-conditioned
push dynamics model.

    Run base policy pi_0 (random pushes on the block) to collect D = {(s_t, a_t, s_t+1)}
    while not done:
        train f_theta and the per-setting latents on D
        for rollout m = 1..M:
            for t = 1..T:
                a_t = first action of random-shooting MPC (push block centroid to the goal)
                execute a_t, add (s_t, a_t, s_t+1) to D

Physics settings. A fixed pool of block physics settings (mass, static and dynamic friction)
is sampled once. Each setting has an id and its own latent vector (a row of an nn.Embedding),
learned only by backprop through the dynamics loss. Every rollout assigns settings from the
pool to the parallel envs (env.set_physics + reconfigure), so a setting always uses the same
latent, and that latent keeps being trained as more data from that setting comes in.

Model input: all observed points above the table plane (found by its known height, i.e. the
block) plus --table-points evenly spread table points, expressed in the push's action frame. Pushes
come from a mix of aimed and uniformly placed actions (see policy/mpc.py), so the data also
contains pushes that miss the block.

Run from the repo root:
    python train_dynamics.py --run-dir runs/dynamics0
    python train_dynamics.py --run-dir runs/quick --num-envs 8 --pool-size 16 --iterations 2   # smoke test

TensorBoard (loss curves, rollout stats, predicted vs. true point flow figures):
    .venv/bin/tensorboard --logdir runs --port 6009
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from dynamics_model.data import TransitionBuffer
from dynamics_model.geometry import (
    action_features,
    action_frame,
    block_points,
    model_input,
    observed_cloud,
    pose_change,
    world_to_frame,
    world_transform_to_frame,
)
from dynamics_model.model import (
    ENCODERS,
    DynamicsModel,
    LatentTable,
    point_flow_loss,
    training_objective,
    true_flow_magnitude,
)
from dynamics_model.viz import flow_figure, plt
from env import PushPrimitiveConfig, make_push_env
from policy.mpc import ActionSampler, random_shooting_mpc

warnings.filterwarnings("ignore")


@dataclass
class Config:
    run_dir: str = "runs/dynamics"
    seed: int = 0
    # resume: continue from a previous run (its dataset, physics pool, model and latents, and
    # rollout count). init_model optionally replaces its model + latents with another checkpoint
    # trained on the same dataset (e.g. scripts/compare_encoders.py output); add_settings adds new
    # physics settings (new latent rows, initialised at the mean latent) to the pool.
    resume_from: str = ""
    init_model: str = ""
    add_settings: int = 0
    # environments
    num_envs: int = 32  # parallel envs per rollout
    pool_size: int = 64  # number of unique physics settings (= number of latents)
    mass: tuple = (0.05, 1.0)  # kg, log-uniform
    static_friction: tuple = (0.2, 1.0)
    dynamic_ratio: tuple = (0.5, 1.0)  # mu_d = mu_s * U(ratio)  (keeps mu_d <= mu_s)
    pushes_per_rollout: int = 5  # T
    # algorithm
    init_rollouts: int = 4  # random-policy rollouts for the initial dataset
    iterations: int = 10
    rollouts_per_iter: int = 2  # M
    mpc_samples: int = 256
    mpc_horizon: int = 1
    random_uniform_fraction: float = 0.25  # random policy: share of pushes placed anywhere
    mpc_uniform_fraction: float = 0.5  # MPC proposals: share placed anywhere
    # push force / duration ranges for random and MPC pushes. Pushes longer than ~0.8 s mostly
    # keep accelerating a block that already slides and launch it (unpredictable, > 30 cm)
    min_force: float = 0.2  # N
    max_force: float = 5.0  # N
    min_duration: float = 0.3  # s
    max_duration: float = 0.8  # s
    # model / training
    encoder: str = "pointnet"  # pointnet | pointnet_early | cross_attn | transformer (dynamics_model/model.py)
    latent_dim: int = 8
    # model input points: every observed point above the table (the block shows ~150-330), then
    # min(table_points, n_in - block points) table points spread evenly by farthest point
    # sampling, then repeats of the block points to fill
    n_in: int = 512
    table_points: int = 100
    n_loss: int = 512  # block points used in the loss (same rule)
    # each training phase: up to max_train_steps, checking the val error every eval_every steps;
    # stops after `patience` checks without improvement and keeps the best model + latents
    max_train_steps: int = 10000
    final_train_steps: int = 20000  # budget of the last training phase
    eval_every: int = 100
    patience: int = 20  # = 2000 steps (val is noisy; short patience stopped training too early)
    figure_every: int = 2000  # draw the point-flow figure every this many steps (0 = end of phase only)
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4  # model weights only (AdamW)
    latent_lr: float = 1e-2
    latent_l2: float = 1e-3
    # "absolute": mean point-flow error in m. "relative": each push's error divided by how far
    # the block moved + relative_floor, so small moves count as much as long slides
    # (logged cm errors stay absolute either way; early stopping uses the training objective)
    loss: str = "absolute"
    relative_floor: float = 0.01  # m
    # augmentation (training batches only)
    mirror_aug: bool = True  # reflect samples across the push direction (y -> -y in the action frame)
    point_noise: float = 0.001  # m, Gaussian noise on the input points


# ------------------------------------------------------------------ settings
# Seed of the held-out test settings (eval_latent_inference.py). Training settings never use it,
# and since physics values are drawn from continuous ranges, test settings are always unseen.
TEST_SEED = 1000


def make_physics_pool(cfg: Config, seed=None, size=None):
    seed = cfg.seed if seed is None else seed
    size = cfg.pool_size if size is None else size
    rng = np.random.default_rng(seed)
    mass = np.exp(rng.uniform(np.log(cfg.mass[0]), np.log(cfg.mass[1]), size))
    mu_s = rng.uniform(*cfg.static_friction, size)
    mu_d = mu_s * rng.uniform(*cfg.dynamic_ratio, size)
    pool = [
        {"mass": float(m), "static_friction": float(s), "dynamic_friction": float(d)}
        for m, s, d in zip(mass, mu_s, mu_d)
    ]
    return pool, pool_tensor(pool)


def pool_tensor(pool):
    return torch.tensor([[p["mass"], p["static_friction"], p["dynamic_friction"]] for p in pool], dtype=torch.float32)


def settings_for_rollout(counts, num_envs, rng: np.random.Generator):
    """The num_envs settings with the least data so far (ties broken randomly), in random env
    order. From scratch this cycles through the pool; settings added later catch up first."""
    order = np.lexsort((rng.random(len(counts)), counts))  # by count, then random
    return rng.permutation(order[:num_envs])


# ------------------------------------------------------------------ rollouts
def run_rollout(env, cfg, pool, pool_t, ids, policy, buf, rollout_idx):
    u = env.unwrapped
    dev = u.device
    u.set_physics([pool[i] for i in ids])
    obs, _ = env.reset(seed=cfg.seed * 100_000 + rollout_idx, options={"reconfigure": True})
    seg_ids = torch.tensor([o.per_scene_id for o in u.obj._objs], device=dev)
    ids_t = torch.as_tensor(ids, device=dev)
    goal = u.goal_region.pose.p[:, :2].clone()
    n_added, contact = 0, []
    for _ in range(cfg.pushes_per_rollout):
        cloud, n_obs = observed_cloud(obs, cfg.n_in, cfg.table_points)
        loss_pts, valid_loss = block_points(obs, seg_ids, cfg.n_loss)
        a = policy(cloud, ids_t, goal)
        pts_in = model_input(cloud, a)
        valid_in = n_obs >= 10
        p0, q0 = u.obj.pose.p.clone(), u.obj.pose.q.clone()
        obs, _, _, _, info = env.step(a)
        p1, q1 = u.obj.pose.p.clone(), u.obj.pose.q.clone()

        R, t = pose_change(p0, q0, p1, q1)
        origin, R_a = action_frame(a)
        R_f, t_f = world_transform_to_frame(R, t, origin, R_a)
        still_on_table = (p0[:, 2] > 0) & (p1[:, 2] > 0)
        keep = valid_in & valid_loss & still_on_table & torch.isfinite(t_f).all(-1)
        buf.add(
            pts_in=pts_in[keep],
            pts_loss=world_to_frame(loss_pts, origin, R_a)[keep],
            act_feat=action_features(a)[keep],
            R_gt=R_f[keep],
            t_gt=t_f[keep],
            setting=ids_t[keep],
            action=a[keep],
            physics=pool_t.to(dev)[ids_t][keep],
            made_contact=info["push"]["made_contact"][keep],
        )
        n_added += int(keep.sum())
        contact.append(info["push"]["made_contact"].float().mean().item())
    dist = (u.obj.pose.p[:, :2] - goal).norm(dim=-1)
    on_table = u.obj.pose.p[:, 2] > 0
    return {
        "samples": n_added,
        "contact_rate": float(np.mean(contact)),
        # mean over blocks still on the table (one block knocked onto the floor would dominate)
        "final_goal_dist_cm": 100 * dist[on_table].mean().item() if on_table.any() else float("nan"),
        "off_table": (~on_table).float().mean().item(),
        "success": ((dist < u.goal_radius) & on_table).float().mean().item(),
    }


# ------------------------------------------------------------------ training
def evaluate(model, latents, t, idx):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(idx), 2048):
            b = idx[i:i + 2048]
            R, tr = model(t["pts_in"][b], t["act_feat"][b], latents(t["setting"][b]))
            errs.append(point_flow_loss(t["pts_loss"][b], R, tr, t["R_gt"][b], t["t_gt"][b]))
    model.train()
    return torch.cat(errs) if errs else torch.zeros(0)


def augment(pts_in, pts_loss, R, t, cfg):
    """Training-time augmentation, in the action frame.

    Mirror: reflecting a push across its own direction (y -> -y) gives another valid push - the
    cube, the gripper and the table are symmetric - so with S = diag(1, -1, 1) the points become
    S x and the block's motion R, t becomes S R S, S t.
    Noise: small Gaussian jitter on the input points (not on the loss points)."""
    if cfg.mirror_aug:
        s = torch.ones(len(R), 3, device=R.device)
        s[torch.rand(len(R), device=R.device) < 0.5, 1] = -1
        pts_in, pts_loss = pts_in * s[:, None], pts_loss * s[:, None]
        R = s[:, :, None] * R * s[:, None, :]  # S R S
        t = t * s
    if cfg.point_noise > 0:
        pts_in = pts_in + cfg.point_noise * torch.randn_like(pts_in)
    return pts_in, pts_loss, R, t


def train(model, latents, opt, buf, cfg, dev, tb, step, max_steps=None):
    """One training phase with early stopping on the val error (see Config), at most max_steps
    (default cfg.max_train_steps). Logs the loss of every step and the val error of every check
    to TensorBoard. step: [global step]."""
    t = buf.tensors(dev)
    tr_idx, va_idx = buf.split_indices(dev)
    floor = cfg.relative_floor if cfg.loss == "relative" else None
    val_mag = true_flow_magnitude(t["pts_loss"][va_idx], t["R_gt"][va_idx], t["t_gt"][va_idx])
    moved = val_mag > 0.005  # pushes where the block actually moved

    # flow figure: the same few validation pushes all phase (3 where the block moved, 1 where it
    # didn't), drawn every figure_every steps and at the end
    g = torch.Generator(device="cpu").manual_seed(step[0])
    mv, st = va_idx[moved].cpu(), va_idx[~moved].cpu()
    pick = torch.cat([mv[torch.randperm(len(mv), generator=g)[:3]], st[torch.randperm(len(st), generator=g)[:1]]]).to(dev)

    def log_figure():
        if len(pick):
            fig = flow_figure(model, latents, t, pick)
            tb.add_figure("val/point_flow", fig, step[0])
            plt.close(fig)

    losses = []
    best_err, best_state, since_best = float("inf"), None, 0
    for i in range(1, (max_steps or cfg.max_train_steps) + 1):
        b = tr_idx[torch.randint(len(tr_idx), (cfg.batch_size,), device=dev)]
        pts_in, pts_loss, R_gt, t_gt = augment(t["pts_in"][b], t["pts_loss"][b], t["R_gt"][b], t["t_gt"][b], cfg)
        z = latents(t["setting"][b])
        R, tr = model(pts_in, t["act_feat"][b], z)
        err = point_flow_loss(pts_loss, R, tr, R_gt, t_gt)
        objective = training_objective(err, true_flow_magnitude(pts_loss, R_gt, t_gt), floor).mean()
        loss = objective + cfg.latent_l2 * z.pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        flow = err.mean()
        losses.append(flow.item())
        step[0] += 1
        tb.add_scalar("train/point_flow_err_cm", 100 * flow.item(), step[0])
        tb.add_scalar("train/loss", loss.item(), step[0])
        if cfg.figure_every and i % cfg.figure_every == 0:
            log_figure()

        if len(va_idx) and i % cfg.eval_every == 0:
            val_err = evaluate(model, latents, t, va_idx)
            tb.add_scalar("val/point_flow_err_cm", 100 * val_err.mean().item(), step[0])
            err = training_objective(val_err, val_mag, floor).mean().item()  # early stopping criterion
            if floor is not None:
                tb.add_scalar("val/relative_objective", err, step[0])
            if err < best_err:
                best_err, since_best = err, 0
                best_state = ({k: v.clone() for k, v in model.state_dict().items()},
                              {k: v.clone() for k, v in latents.state_dict().items()})
            else:
                since_best += 1
                if since_best >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state[0])
        latents.load_state_dict(best_state[1])
    steps_run = i

    # validation error, overall and on the pushes where the block actually moved (> 5 mm);
    # "no motion" = error of always predicting that the block stays put (= how far it moved)
    val = evaluate(model, latents, t, va_idx)
    still = val_mag
    log_figure()  # final figure, with the restored best model

    return {
        "steps": steps_run,
        "train_err_cm": 100 * float(np.mean(losses[-100:])),
        "val_err_cm": 100 * val.mean().item(),
        "val_err_moved_cm": 100 * val[moved].mean().item() if moved.any() else float("nan"),
        # error as a fraction of how far the block moved (median over pushes that moved > 5 mm)
        "val_rel_err_moved": (val[moved] / still[moved]).median().item() if moved.any() else float("nan"),
        "val_no_motion_baseline_cm": 100 * still.mean().item(),
        "val_frac_moved": moved.float().mean().item(),
        "n_train": len(tr_idx),
        "n_val": len(va_idx),
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(Config()).items():
        if isinstance(v, bool):
            kw = dict(action=argparse.BooleanOptionalAction)
        elif isinstance(v, tuple):
            kw = dict(nargs=2, type=type(v[0]))
        else:
            kw = dict(type=type(v))
        if f == "loss":
            kw["choices"] = ["absolute", "relative"]
        if f == "encoder":
            kw["choices"] = ENCODERS
        ap.add_argument("--" + f.replace("_", "-"), default=v, **kw)
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in vars(ap.parse_args()).items()})

    run_dir = Path(cfg.run_dir)
    if cfg.resume_from:
        assert run_dir.resolve() != Path(cfg.resume_from).resolve(), "write the resumed run to a new --run-dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # physics settings, data, and the rollout counter (continued when resuming)
    ckpt, rollout = None, 0
    if cfg.resume_from:
        prev = Path(cfg.resume_from)
        pool = json.loads((prev / "physics_pool.json").read_text())
        buf = TransitionBuffer.load(prev / "dataset.pt", seed=cfg.seed)
        ckpt = torch.load(cfg.init_model or prev / "checkpoint.pt")
        saved_enc = ckpt.get("encoder", ckpt.get("config", {}).get("encoder", "pointnet"))
        assert saved_enc == cfg.encoder, f"checkpoint uses encoder {saved_enc!r}; pass --encoder {saved_enc}"
        rollout = sum('"rollout"' in line for line in open(prev / "metrics.jsonl"))
        print(f"resuming from {prev}: {len(buf)} transitions, {len(pool)} settings, {rollout} rollouts done, "
              f"model from {cfg.init_model or prev / 'checkpoint.pt'}", flush=True)
    else:
        pool, _ = make_physics_pool(cfg)
        buf = TransitionBuffer(seed=cfg.seed)
    n_old = len(pool)
    if cfg.add_settings:
        extra_seed = cfg.seed + 1 + n_old  # different for every extension, never the test seed
        assert extra_seed != TEST_SEED
        extra, _ = make_physics_pool(cfg, seed=extra_seed, size=cfg.add_settings)
        pool = pool + extra
        print(f"added {cfg.add_settings} new physics settings (ids {n_old}-{len(pool) - 1})", flush=True)
    pool_t = pool_tensor(pool)
    (run_dir / "physics_pool.json").write_text(json.dumps(pool, indent=2))

    env = make_push_env(num_envs=cfg.num_envs, obs_mode="pointcloud", push_config=PushPrimitiveConfig(phase_timeout=2.0))
    dev = env.unwrapped.device

    model = DynamicsModel(latent_dim=cfg.latent_dim, encoder=cfg.encoder).to(dev)
    latents = LatentTable(len(pool), cfg.latent_dim).to(dev)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        old = ckpt["latents"]["emb.weight"].to(dev)
        assert len(old) == n_old, f"checkpoint has {len(old)} latents but the pool has {n_old} settings"
        with torch.no_grad():
            latents.emb.weight[:n_old] = old
            latents.emb.weight[n_old:] = old.mean(0)  # new settings start at the mean latent
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": latents.parameters(), "lr": cfg.latent_lr, "weight_decay": 0.0},
    ])
    ranges = dict(force=(cfg.min_force, cfg.max_force), duration=(cfg.min_duration, cfg.max_duration))
    random_sampler = ActionSampler(uniform_fraction=cfg.random_uniform_fraction, **ranges)
    mpc_sampler = ActionSampler(uniform_fraction=cfg.mpc_uniform_fraction, **ranges)
    log = open(run_dir / "metrics.jsonl", "a")

    def next_settings():
        counts = np.bincount(torch.cat(buf.data["setting"]).numpy(), minlength=len(pool)) if len(buf) else np.zeros(len(pool))
        return settings_for_rollout(counts, cfg.num_envs, rng)

    def save():
        torch.save({"model": model.state_dict(), "latents": latents.state_dict(), "config": asdict(cfg),
                    "encoder": cfg.encoder, "physics_pool": pool}, run_dir / "checkpoint.pt")
        buf.save(run_dir / "dataset.pt")
    tb = SummaryWriter(str(run_dir / "tb"))
    step = [0]  # global gradient step

    def emit(rec):
        """Print + metrics.jsonl + TensorBoard (training metrics at the gradient step,
        rollout metrics at the rollout index)."""
        rec["time"] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)  # flush: show up immediately even when piped to tee
        log.write(json.dumps(rec) + "\n")
        log.flush()
        phase = rec["phase"]
        tag, x = ("val", step[0]) if "train" in phase else (f"rollout_{phase}", rec["rollout"])
        for k, v in rec.items():
            if isinstance(v, (int, float)) and k not in ("rollout", "iteration", "time") and math.isfinite(v):
                tb.add_scalar(f"{tag}/{k}", v, x)
        tb.flush()

    def random_policy(cloud, ids, goal):
        return random_sampler(cloud, 1)[:, 0]

    def mpc_policy(cloud, ids, goal):
        model.eval()
        with torch.no_grad():
            a, _ = random_shooting_mpc(model, cloud, latents(ids), goal, mpc_sampler,
                                       num_samples=cfg.mpc_samples, horizon=cfg.mpc_horizon)
        model.train()
        return a

    t0 = time.time()
    for _ in range(cfg.init_rollouts):
        emit({"phase": "random", "rollout": rollout,
              **run_rollout(env, cfg, pool, pool_t, next_settings(), random_policy, buf, rollout)})
        rollout += 1

    for it in range(cfg.iterations):
        emit({"phase": "train", "iteration": it, **train(model, latents, opt, buf, cfg, dev, tb, step)})
        save()
        for _ in range(cfg.rollouts_per_iter):
            emit({"phase": "mpc", "iteration": it, "rollout": rollout,
                  **run_rollout(env, cfg, pool, pool_t, next_settings(), mpc_policy, buf, rollout)})
            rollout += 1

    emit({"phase": "final_train", **train(model, latents, opt, buf, cfg, dev, tb, step, max_steps=cfg.final_train_steps)})
    save()
    tb.close()
    env.close()


if __name__ == "__main__":
    main()
