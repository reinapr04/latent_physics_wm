"""Action sampling and random-shooting MPC with the dynamics model.

The same sampler is the random base policy pi_0 (used to collect the initial dataset) and the
proposal distribution of the MPC. It mixes two kinds of pushes:
  - uniform: p anywhere on the workspace, any direction, any height. Most of these miss the
    object; they teach the model that pushes away from the object do nothing.
  - aimed: p on the outline of whatever sticks up from the table (a non-learned cue: height
    above the known table plane), so that random data isn't almost all misses. For a box the
    outline is that of its top face; for a sphere, its center and radius are read off the
    top cap and the object's height (shape="sphere").
The MPC then decides between candidates with the model alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dynamics_model.geometry import TABLE_Z, action_features, action_frame, model_input
from env.reward import block_reward

# where uniform pushes are placed (world frame, m)
UNIFORM_LOW = (-0.25, -0.30)
UNIFORM_HIGH = (0.35, 0.30)


def top_face(clouds, band=0.006):
    """Observed points of the object's top: raised points within `band` of the highest one
    (a box's top face; a sphere's top cap). (M, N, 3) -> (M, N) bool."""
    z = clouds[..., 2]
    raised = z > TABLE_Z
    z_top = torch.where(raised, z, torch.full_like(z, -1.0)).max(dim=1, keepdim=True).values
    return raised & (z > z_top - band)


def outline_entry(clouds, top, q, d, line_width=0.005):
    """Where the line through q along d enters the observed top-face outline (= a box's
    outline, including faces the camera can't see). clouds (M, N, 3), top (M, N), q (M, k, 2),
    d (M, k, 2) unit -> (M, k, 2). If the line misses the top face, returns q."""
    rel = clouds[:, None, :, :2] - q[:, :, None, :]  # (M, k, N, 2)
    s = (rel * d[:, :, None]).sum(-1)  # along the line
    lat = rel[..., 0] * d[:, :, None, 1] - rel[..., 1] * d[:, :, None, 0]  # across it
    on_line = top[:, None] & (lat.abs() < line_width)
    s_min = torch.where(on_line, s, torch.full_like(s, float("inf"))).min(-1).values.clamp(max=0)
    return q + torch.nan_to_num(s_min, posinf=0.0)[..., None] * d


def sphere_geometry(clouds, fallback_radius=0.035):
    """Center xy (M, 2) and radius (M,) of a sphere resting on the table, from the observation
    alone: the centroid of the top cap sits over the center, and the height of the highest point
    is the diameter. Envs without raised points get the fallback radius."""
    top = top_face(clouds).float()
    n = top.sum(1, keepdim=True)
    c = (clouds[..., :2] * top[..., None]).sum(1) / n.clamp_min(1)
    z_top = (clouds[..., 2] * top).sum(1) / n[:, 0].clamp_min(1)
    r = torch.where(n[:, 0] > 0, (z_top / 2).clamp(0.01, 0.1), torch.full_like(z_top, fallback_radius))
    return c, r


def sphere_entry(clouds, q, d):
    """Where the line through q along d enters the sphere's outline. q (M, k, 2) is a point of
    the line (it defines the lateral offset from the center), d (M, k, 2) unit -> (M, k, 2)."""
    c, r = sphere_geometry(clouds)
    rel = c[:, None] - q  # (M, k, 2) line point -> center
    lat = rel[..., 0] * d[..., 1] - rel[..., 1] * d[..., 0]  # lateral offset of the line
    along = (rel * d).sum(-1)  # from q to the point of the line nearest the center
    half_chord = (r[:, None] ** 2 - lat ** 2).clamp_min(0).sqrt()
    return q + (along - half_chord)[..., None] * d


@dataclass
class ActionSampler:
    uniform_fraction: float = 0.25  # share of pushes placed uniformly on the workspace
    height: tuple = (0.01, 0.05)  # m, p_z (the block is 4 cm tall: the top of the range misses it)
    force: tuple = (0.2, 5.0)  # N
    duration: tuple = (0.3, 0.8)  # s (longer pushes mostly just launch the block once it slides)
    top_band: float = 0.006  # m, raised points this close to the highest one count as top face
    line_width: float = 0.005  # m, half-width of the band used to find the entry point
    shape: str = "box"  # "box" | "sphere" (env.tasks.TaskSpec.shape)

    def __call__(self, clouds, k):
        """clouds (M, N, 3) world (table included) -> actions (M, k, 7) = [p(3), d(2), F, T]."""
        M, N, _ = clouds.shape
        dev = clouds.device

        def uniform(lo, hi, *shape):
            return lo + torch.rand(*shape, device=dev) * (hi - lo)

        ang = uniform(0, 2 * math.pi, M, k)
        d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (M, k, 2)

        # aimed: random top-face point q, contact where the line through q along d enters the outline
        top = top_face(clouds, self.top_band)  # (M, N)
        has_top = top.any(dim=1)
        q_idx = torch.multinomial(top.float() + (~has_top)[:, None].float(), k, replacement=True)
        q = torch.gather(clouds, 1, q_idx[..., None].expand(-1, -1, 3))[..., :2]  # (M, k, 2)
        if self.shape == "sphere":
            p_aimed = sphere_entry(clouds, q, d)
        else:
            p_aimed = outline_entry(clouds, top, q, d, self.line_width)

        p_uniform = torch.stack(
            [uniform(UNIFORM_LOW[0], UNIFORM_HIGH[0], M, k), uniform(UNIFORM_LOW[1], UNIFORM_HIGH[1], M, k)], -1
        )
        use_uniform = (torch.rand(M, k, device=dev) < self.uniform_fraction) | ~has_top[:, None]
        p_xy = torch.where(use_uniform[..., None], p_uniform, p_aimed)

        return torch.cat(
            [
                p_xy,
                uniform(*self.height, M, k)[..., None],
                d,
                uniform(*self.force, M, k)[..., None],
                uniform(*self.duration, M, k)[..., None],
            ],
            -1,
        )


def block_centroid(clouds):
    """xy centroid of the observed points above the table plane (the object), (M, N, 3) -> (M, 2),
    and whether any such points exist (M,). Uses only the observation (known table height)."""
    raised = (clouds[..., 2] > TABLE_Z).float()
    c = (clouds[..., :2] * raised[..., None]).sum(1) / raised.sum(1, keepdim=True).clamp_min(1)
    return c, raised.sum(1) > 0


@dataclass
class ProbeSampler:
    """Informative pushes for fitting a new environment's latent.

    Every push goes through the observed object center (small sideways offset so some pushes
    also turn the block / deflect the ball), with the contact point where that line enters the
    object's observed outline (so the gripper starts well clear of it), at a height the
    fingertips can reach, with a force spread log-uniformly over the range (many pushes near the
    force where the object starts to move, which is what reveals mass and friction). The
    direction points roughly towards the middle of the workspace, so objects stay on the table
    and in view.
    """

    height: tuple = (0.015, 0.035)  # m
    force: tuple = (0.3, 5.0)  # N, log-uniform
    duration: tuple = (0.4, 0.8)  # s
    lateral: float = 0.01  # m, max sideways offset of the push line from the centroid
    direction_noise: float = math.pi / 3  # rad, spread around "towards the middle"
    middle: tuple = (0.05, 0.0)  # m, middle of the workspace (world xy)
    near_middle: float = 0.05  # m, closer than this to the middle -> any direction
    shape: str = "box"

    def __call__(self, clouds, k):
        """clouds (M, N, 3) world -> actions (M, k, 7)."""
        M = clouds.shape[0]
        dev = clouds.device

        def uniform(lo, hi):
            return lo + torch.rand(M, k, device=dev) * (hi - lo)

        c = sphere_geometry(clouds)[0] if self.shape == "sphere" else block_centroid(clouds)[0]
        to_mid = torch.tensor(self.middle, device=dev) - c  # (M, 2)
        base = torch.atan2(to_mid[:, 1], to_mid[:, 0])[:, None]
        ang = torch.where(
            (to_mid.norm(dim=1) > self.near_middle)[:, None],
            base + uniform(-self.direction_noise, self.direction_noise),
            uniform(0, 2 * math.pi),
        )
        d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (M, k, 2)
        perp = torch.stack([-d[..., 1], d[..., 0]], -1)
        q = c[:, None] + uniform(-self.lateral, self.lateral)[..., None] * perp
        p_xy = sphere_entry(clouds, q, d) if self.shape == "sphere" else outline_entry(clouds, top_face(clouds), q, d)
        force = torch.exp(uniform(math.log(self.force[0]), math.log(self.force[1])))
        return torch.cat(
            [p_xy, uniform(*self.height)[..., None], d, force[..., None], uniform(*self.duration)[..., None]], -1
        )


def predict_motion(model, clouds, actions, z, inputs=None):
    """Predicted rigid motion of the object in the world frame. clouds (B, N, 3) world, actions
    (B, 7), z (B, L) -> R (B, 3, 3), t (B, 3) with x' = R x + t. Models with other inputs
    (dynamics_model.inputs: image tokens, crops, fusion) get them as inputs = (x, aux)."""
    x, aux = inputs if inputs is not None else (None, None)
    if model.input_kind == "points":
        x = model_input(clouds, actions)
    R_f, t_f = model(x, action_features(actions), z, act_world=actions, aux=aux)
    origin, R_a = action_frame(actions)
    R = R_a @ R_f @ R_a.transpose(-1, -2)
    t = (R_a @ t_f[..., None])[..., 0] + origin - (R @ origin[..., None])[..., 0]
    return R, t


def block_position(clouds, half_size=0.02):
    """Object center estimated from a (real or imagined) cloud, (M, N, 3) -> (M, 3).

    xy: centroid of the top face / top cap (fully visible, so its centroid sits over the center;
    the centroid of all raised points is pulled towards the visible sides).
    z: top height minus half the object height.
    """
    top = top_face(clouds).float()
    n = top.sum(1, keepdim=True).clamp_min(1)
    c = (clouds * top[..., None]).sum(1) / n
    return torch.cat([c[:, :2], c[:, 2:] - half_size], 1)


def planning_reward(clouds, goal_xy, half_size=0.02, goal_radius=0.1):
    """The env's reward (env.reward.block_reward, success -> 4) for the object in each cloud.
    The env's reaching term is dropped: the push primitive does the reaching itself."""
    r, success = block_reward(block_position(clouds, half_size), goal_xy, half_size, goal_radius)
    return torch.where(success, torch.full_like(r, 4.0), 1 + r)


def imagine_next_cloud(model, clouds, actions, z, inputs=None):
    """Apply the predicted motion to the raised points (the object; the table stays put)."""
    R, t = predict_motion(model, clouds, actions, z, inputs)
    moved = clouds @ R.transpose(-1, -2) + t[:, None]
    return torch.where((clouds[..., 2] > TABLE_Z)[..., None], moved, clouds)


@torch.no_grad()
def random_shooting_mpc(model, clouds, z, goal_xy, sampler, num_samples=256, horizon=1, chunk=1024,
                        goal_radius=0.1, half_size=0.02, inputs_fn=None):
    """Random-shooting MPC.

    For each env, sample num_samples action sequences of length `horizon` (each action sampled
    on that candidate's imagined cloud at that step), roll them out with the model, and return
    the first action of the sequence whose imagined final object position gets the highest env
    reward (planning_reward).

    clouds (B, N, 3) world (the observed cloud the model works with), z (B, L),
    goal_xy (B, 2) -> actions (B, 7), predicted reward (B,)
    Image-based models get their inputs from inputs_fn(env_idx, actions) -> (x, aux)
    (dynamics_model.inputs.planner_inputs) and support horizon 1 only: there is no imagined
    image for later steps.
    """
    B, N, _ = clouds.shape
    K = num_samples
    cur = clouds[:, None].expand(B, K, N, 3).reshape(B * K, N, 3)
    zz = z[:, None].expand(B, K, -1).reshape(B * K, -1)
    if inputs_fn is not None:
        assert horizon == 1, "image-based models plan one push ahead only"

    def inputs(i, acts):  # the inputs of each candidate's env, computed per chunk (never all B*K at once)
        if inputs_fn is None:
            return None
        return inputs_fn(torch.arange(i, min(i + chunk, B * K), device=clouds.device) // K, acts)

    first = None
    for _ in range(horizon):
        acts = torch.cat([sampler(cur[i:i + chunk], 1)[:, 0] for i in range(0, B * K, chunk)])
        if first is None:
            first = acts
        cur = torch.cat([
            imagine_next_cloud(model, cur[i:i + chunk], acts[i:i + chunk], zz[i:i + chunk], inputs(i, acts[i:i + chunk]))
            for i in range(0, B * K, chunk)
        ])
    goal = goal_xy[:, None].expand(B, K, 2).reshape(B * K, 2)
    reward = planning_reward(cur, goal, half_size=half_size, goal_radius=goal_radius).reshape(B, K)
    best = reward.argmax(dim=1)
    rows = torch.arange(B, device=clouds.device)
    return first.reshape(B, K, 7)[rows, best], reward[rows, best]
