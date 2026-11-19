"""Push-primitive action space for PushBlock-v1.

Instead of controlling the arm joints directly, each action is one physically meaningful push

    a = (p_contact, d_push, F_push, T)

  p_contact  (3,)  the intended contact point: where the fingertips should touch the block, in
                   world coordinates (the same frame as the env's point cloud and object poses).
                   It can be any 3D point; it need not be an observed one (a point on a face the
                   camera can't see is fine). The controller uses no knowledge of the block: the
                   gripper goes to standoff behind p_contact (along -d_push) at height
                   p_contact.z and creeps along d_push until it feels contact, at most
                   approach_extra past p_contact. Then:
                     - wrong-but-close p: contact happens a bit earlier/later along the line;
                       info["push"]["contact_point"] is where contact actually happened.
                     - p beside / above the block: nothing is touched, the gripper backs off
                       (made_contact False).
                     - p inside the block or too low for the block's height: the gripper bumps
                       into the block while moving into position; it stops and lifts away
                       (collided True). Whatever that bump did to the block is part of the outcome.
                   The only clamp is p.z >= min_push_height (fingertips vs. table).
  d_push     (2,)  push direction in the world xy plane (normalised internally).
  F_push     ()    desired pushing force in N, in [0, max_force].
  T          ()    push duration in s, in [0, max_duration].

As a flat Box action this is [px, py, pz, dx, dy, F, T] with shape (num_envs, 7). A dict with
keys "contact", "direction", "force" and "duration" also works. For scripted pushes described
relative to the block (scripted data collection), PushPrimitiveWrapper.block_frame_to_world()
converts to a reachable p_contact on the block's surface.

One step() runs the whole primitive with the arm's end-effector pose controller, for all
parallel envs at once, each with its own phase:

  1. LIFT     go to safe height (and yaw the gripper so its jaw axis is perpendicular to d_push:
              the fingertips' front edges are the pushing surface)
  2. TRAVEL   move above the pre-contact point  p_contact - standoff * d_push
  3. DESCEND  lower to p_contact.z
              (touching the block during TRAVEL/DESCEND -> stop, skip to RAISE, collided=True)
  4. APPROACH creep along d_push until the gripper touches the block
  5. PUSH     force control for T seconds: the commanded EE target leads the actual EE along
              d_push by a PI function of (F_push - F_measured). Ends early if contact is lost
              for longer than contact_loss_grace.
  6. BACKOFF  retreat along -d_push
  7. RAISE    lift to safe height
  8. HOME     go to the home pose, behind the workspace as seen from the camera, so the next
              observation shows the block unoccluded. reset() also ends at the home pose.

The full per-control-step trajectory (phase, EE position, block pose, measured contact force,
...) is returned in info["push"].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles, quaternion_to_matrix
from mani_skill.utils.structs import Pose

LIFT, TRAVEL, DESCEND, APPROACH, PUSH, BACKOFF, RAISE, HOME, DONE = range(9)
PHASE_NAMES = ["lift", "travel", "descend", "approach", "push", "backoff", "raise", "home", "done"]
_WAYPOINT_PHASES = (LIFT, TRAVEL, DESCEND, BACKOFF, RAISE, HOME)


@dataclass
class PushPrimitiveConfig:
    # action bounds
    max_force: float = 20.0  # N
    max_duration: float = 2.0  # s
    contact_low: tuple = (-0.5, -0.5, 0.0)  # m, world-frame box for p_contact
    contact_high: tuple = (0.5, 0.5, 0.15)
    # geometry
    home_position: tuple = (-0.35, 0.0, 0.30)  # m, TCP position between pushes (behind the workspace)
    safe_height: float = 0.10  # m above the table for free-space moves
    standoff: float = 0.03  # m, pre-contact gap between the fingertips and p_contact
    pusher_radius: float = 0.01  # m, TCP to the fingertips' front edge along d_push
    min_push_height: float = 0.015  # m, lowest TCP height (keeps fingertips off the table)
    approach_extra: float = 0.03  # m past p_contact before giving up
    # Reachable push heights, used only by block_frame_to_world (scripted pushes): stay
    # top_margin below the block's top face (fingers slip over otherwise), and no more than
    # finger_reach below it (the hand, which sticks out past the fingertips, would hit the
    # block's top edge). Measured for the Panda gripper.
    top_margin: float = 0.005  # m
    finger_reach: float = 0.035  # m
    retreat_distance: float = 0.03  # m
    # motion
    move_speed: float = 0.25  # m/s, free-space moves
    # m/s, final descent to the push height. Slower, so that if p_contact puts the gripper on
    # top of the block, the bump is caught before it turns into a hard impact.
    descend_speed: float = 0.1
    approach_speed: float = 0.04  # m/s
    pos_tol: float = 0.005  # m, waypoint reached tolerance
    phase_timeout: float = 3.0  # s, a waypoint phase gives up after this long
    # force control (PI on the EE target lead distance)
    contact_threshold: float = 0.05  # N, force along d_push that counts as contact
    contact_loss_grace: float = 0.1  # s without contact before the push is aborted
    kp: float = 3e-4  # m / N
    ki: float = 1e-2  # m / (N s)
    # Extra lead = velocity_feedforward * (EE speed along d_push). The joint PD controller damps
    # motion towards a static target, so without this the applied force drops by ~D*v while the
    # block slides. None = use the arm's damping / stiffness ratio (0.1 s for the Panda).
    velocity_feedforward: Optional[float] = None  # s
    max_lead: float = 0.05  # m
    # episode
    max_pushes: int = 10  # truncate after this many primitives
    record_frames: bool = False  # store env.render() for every control step in info["push"]["frames"]
    record_envs: Optional[int] = None  # only keep the first record_envs envs' frames (None = all)


def _yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class PushPrimitiveWrapper(gym.Wrapper):
    """Wraps a PushBlock-v1 env created with control_mode="pd_ee_pose" and track_contact_force=True.
    Prefer make_push_env(), which sets those up."""

    def __init__(self, env: gym.Env, config: Optional[PushPrimitiveConfig] = None):
        super().__init__(env)
        self.cfg = config or PushPrimitiveConfig()
        u = self.env.unwrapped
        assert u.control_mode == "pd_ee_pose", "PushPrimitiveWrapper needs control_mode='pd_ee_pose'"
        assert u.track_contact_force, "PushPrimitiveWrapper needs track_contact_force=True"
        n = u.num_envs
        low = np.array([*self.cfg.contact_low, -1, -1, 0, 0], dtype=np.float32)
        high = np.array([*self.cfg.contact_high, 1, 1, self.cfg.max_force, self.cfg.max_duration], dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(np.tile(low, (n, 1)), np.tile(high, (n, 1)), dtype=np.float32)
        self._push_count = 0
        if self.cfg.velocity_feedforward is None:
            arm = u.agent.controller.controllers["arm"].config
            self._kv = float(np.mean(arm.damping) / np.mean(arm.stiffness))
        else:
            self._kv = self.cfg.velocity_feedforward

    @property
    def u(self):
        return self.env.unwrapped

    # ------------------------------------------------------------------ gym API
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self._push_count = 0
        # gripper orientation at reset (pointing down); each push yaws it about the vertical axis
        q = (self.u.agent.robot.pose.inv() * self.u.agent.tcp.pose).q
        self._ee_rot0 = quaternion_to_matrix(q)  # (n, 3, 3), in the robot base frame
        self._ee_euler0 = matrix_to_euler_angles(self._ee_rot0, "XYZ")
        self._ee_euler = self._ee_euler0.clone()
        f1, f2 = self.u.agent.finger1_link.pose.p, self.u.agent.finger2_link.pose.p
        self._jaw_axis_yaw0 = torch.atan2(f2[:, 1] - f1[:, 1], f2[:, 0] - f1[:, 0])
        obs, info = self._go_home()
        return obs, info

    def step(self, action):
        contact, direction, force, duration = self._parse_action(action)
        obs, reward, terminated, truncated, info, push_info = self._execute_push(contact, direction, force, duration)
        self._push_count += 1
        info = dict(info)
        info["push"] = push_info
        terminated = info["success"].clone()
        truncated = torch.full_like(terminated, self._push_count >= self.cfg.max_pushes)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ public helpers
    def block_frame_to_world(self, contact_local, direction):
        """Convert a push described relative to the block into a world-frame p_contact.

        contact_local (n, 3): (cx, cy) in the block's xy frame, normalised by its half size and
        projected onto its side faces ((-1, 0) = centre of the -x face; (0, 0) = the face
        opposite the push direction); cz = height on the face (-1 bottom, 0 middle, +1 top),
        clamped to the band the fingertips can reach for this block (see top_margin /
        finger_reach). direction (n, 2): world-frame push direction. Returns (n, 3) points on
        the block's surface. Uses the simulator's block pose (privileged; for scripted pushes).
        """
        u = self.u
        contact_local = torch.as_tensor(contact_local, dtype=torch.float32, device=u.device).reshape(-1, 3)
        direction = torch.as_tensor(direction, dtype=torch.float32, device=u.device).reshape(-1, 2)
        block, h = u.obj.pose, u.block_half_size
        yaw = _yaw(block.q)
        cos, sin = torch.cos(yaw), torch.sin(yaw)
        d = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        d_local = torch.stack([cos * d[:, 0] + sin * d[:, 1], -sin * d[:, 0] + cos * d[:, 1]], dim=1)
        c = contact_local[:, :2].clamp(-1, 1)
        c = torch.where(c.norm(dim=1, keepdim=True) < 1e-6, -d_local, c)
        c = c / c.abs().max(dim=1, keepdim=True).values.clamp_min(1e-6) * h[:, None]
        c_world = torch.stack([cos * c[:, 0] - sin * c[:, 1], sin * c[:, 0] + cos * c[:, 1]], dim=1)
        z = h * (1 + contact_local[:, 2].clamp(-1, 1))
        z_low = torch.clamp(2 * h - self.cfg.finger_reach, min=self.cfg.min_push_height)
        z = torch.minimum(torch.maximum(z, z_low), 2 * h - self.cfg.top_margin)
        return torch.cat([block.p[:, :2] + c_world, z[:, None]], dim=1)

    # ---------------------------------------------------------------- helpers
    def _parse_action(self, action):
        dev = self.u.device
        if isinstance(action, dict):
            contact = torch.as_tensor(action["contact"], dtype=torch.float32, device=dev).reshape(-1, 3)
            direction = torch.as_tensor(action["direction"], dtype=torch.float32, device=dev).reshape(-1, 2)
            force = torch.as_tensor(action["force"], dtype=torch.float32, device=dev).reshape(-1)
            duration = torch.as_tensor(action["duration"], dtype=torch.float32, device=dev).reshape(-1)
        else:
            a = torch.as_tensor(action, dtype=torch.float32, device=dev).reshape(-1, 7)
            contact, direction, force, duration = a[:, 0:3], a[:, 3:5], a[:, 5], a[:, 6]
        n = self.u.num_envs
        lo = torch.tensor(self.cfg.contact_low, device=dev)
        hi = torch.tensor(self.cfg.contact_high, device=dev)
        contact = torch.maximum(torch.minimum(contact.expand(n, 3), hi), lo)
        direction = direction.expand(n, 2)
        force = force.expand(n).clamp(0, self.cfg.max_force)
        duration = duration.expand(n).clamp(0, self.cfg.max_duration)
        return contact, direction, force, duration

    def _set_push_orientation(self, d):
        """Yaw the gripper so its jaw axis is perpendicular to the push direction d. Then the
        narrow side of the hand and the front edges of both fingertips lead the push, so the
        contact geometry is the same for every direction."""
        push_yaw = torch.atan2(d[:, 1], d[:, 0])
        rel = push_yaw + np.pi / 2 - self._jaw_axis_yaw0
        rel = torch.remainder(rel + np.pi / 2, np.pi) - np.pi / 2  # jaws are symmetric: use the smallest turn
        c, s = torch.cos(rel), torch.sin(rel)
        rz = torch.zeros((len(rel), 3, 3), device=rel.device)
        rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = c, -s, s, c, 1
        self._ee_euler = matrix_to_euler_angles(rz @ self._ee_rot0, "XYZ")

    def _low_level_action(self, target_world, euler):
        base = self.u.agent.robot.pose
        p_base = (base.inv() * Pose.create_from_pq(p=target_world)).p
        gripper = -torch.ones((self.u.num_envs, 1), device=self.u.device)  # closed
        return torch.cat([p_base, euler, gripper], dim=1)

    def _home(self):
        return torch.tensor(self.cfg.home_position, dtype=torch.float32, device=self.u.device).expand(self.u.num_envs, 3)

    def _sim_step(self, action):
        """Advance physics + controllers by one control step without computing observations
        (rendering the cameras every control step would dominate the run time; the policy only
        sees the observation at the end of a primitive)."""
        self.u._step_action(action)
        self.u._elapsed_steps += 1

    @staticmethod
    def _move_towards(cur, goal, max_step):
        delta = goal - cur
        dist = delta.norm(dim=1, keepdim=True)
        return cur + delta * torch.clamp(max_step / dist.clamp_min(1e-9), max=1.0)

    @torch.no_grad()
    def _go_home(self):
        """Move the (closed) gripper to the home pose. Used at reset."""
        u, cfg = self.u, self.cfg
        dt, home = u.control_timestep, self._home()
        # lift first so the gripper never sweeps through the workspace low
        tcp = u.agent.tcp.pose.p
        waypoints = [torch.cat([tcp[:, :2], torch.full_like(tcp[:, 2:], max(cfg.safe_height, cfg.home_position[2]))], 1), home]
        target = tcp.clone()
        for goal in waypoints:
            for _ in range(int(cfg.phase_timeout / dt)):
                target = self._move_towards(target, goal, cfg.move_speed * dt)
                self._sim_step(self._low_level_action(target, self._ee_euler0))
                if bool(((u.agent.tcp.pose.p - goal).norm(dim=1) < cfg.pos_tol).all()):
                    break
        for _ in range(9):  # let the arm settle so the first observation is static
            self._sim_step(self._low_level_action(home, self._ee_euler0))
        obs, _, _, _, info = u.step(self._low_level_action(home, self._ee_euler0))
        return obs, info

    # ------------------------------------------------------------ the primitive
    @torch.no_grad()
    def _execute_push(self, p_req, direction, force_des, duration):
        cfg, u = self.cfg, self.u
        n, dev, dt = u.num_envs, u.device, u.control_timestep
        d = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        z_push = p_req[:, 2].clamp(min=cfg.min_push_height)
        d3 = torch.cat([d, torch.zeros((n, 1), device=dev)], dim=1)
        self._set_push_orientation(d)

        # TCP sits pusher_radius behind the fingertips' front edge; start standoff before p
        pre_xy = p_req[:, :2] - (cfg.standoff + cfg.pusher_radius) * d
        tcp0 = u.agent.tcp.pose.p.clone()
        safe_z = torch.full((n,), cfg.safe_height, device=dev)
        home = self._home()

        phase = torch.full((n,), LIFT, dtype=torch.long, device=dev)
        phase_time = torch.zeros(n, device=dev)
        # LIFT only ever goes up (from home the gripper is already above safe height; the
        # arm can't reach safe height that close to its base anyway)
        goal = torch.cat([tcp0[:, :2], torch.maximum(tcp0[:, 2], safe_z)[:, None]], dim=1)
        target = tcp0.clone()
        line_origin = torch.cat([pre_xy, z_push[:, None]], dim=1)  # push line: line_origin + s * d
        approach_s = torch.zeros(n, device=dev)
        integ = torch.zeros(n, device=dev)
        push_time = torch.zeros(n, device=dev)
        lost_time = torch.zeros(n, device=dev)
        made_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        collided = torch.zeros(n, dtype=torch.bool, device=dev)
        contact_point = torch.full((n, 3), float("nan"), device=dev)
        timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
        block_p0 = u.obj.pose.p.clone()

        rec = {k: [] for k in ["phase", "tcp_pos", "target_pos", "obj_pose", "contact_force", "force_along_dir"]}
        frames = []
        max_steps = int((6 * cfg.phase_timeout + cfg.max_duration + 2) / dt)
        for _ in range(max_steps):
            if bool((phase == DONE).all()):
                break
            tcp = u.agent.tcp.pose.p
            f_vec = u.pusher_contact_force
            f_along = (f_vec * d3).sum(dim=1)  # force on the block along the push direction
            in_contact = f_along > cfg.contact_threshold

            # ---- waypoint phases: speed-limited straight moves of the commanded target
            wp = torch.zeros(n, dtype=torch.bool, device=dev)
            for ph in _WAYPOINT_PHASES:
                wp |= phase == ph
            speed = torch.where(phase == DESCEND, cfg.descend_speed, cfg.move_speed)
            target = torch.where(wp[:, None], self._move_towards(target, goal, (speed * dt)[:, None]), target)
            reached = wp & ((tcp - goal).norm(dim=1) < cfg.pos_tol)
            expired = wp & (phase_time > cfg.phase_timeout)
            timed_out |= expired & ~reached
            advance = reached | expired

            # ---- APPROACH: creep along the push line until contact
            ap = phase == APPROACH
            approach_s = torch.where(ap, approach_s + cfg.approach_speed * dt, approach_s)
            target = torch.where(ap[:, None], line_origin + approach_s[:, None] * d3, target)
            touched = ap & in_contact
            missed = ap & ~in_contact & (approach_s > cfg.standoff + cfg.approach_extra)
            made_contact |= touched
            # where contact actually happened: the fingertips' front edge at first touch
            contact_point = torch.where(touched[:, None], tcp + cfg.pusher_radius * d3, contact_point)

            # ---- bumping into the block while moving into position: stop and lift away
            moving_in = (phase == TRAVEL) | (phase == DESCEND)
            bumped = moving_in & (f_vec.norm(dim=1) > cfg.contact_threshold)
            collided |= bumped

            # ---- PUSH: PI force control on how far the target leads the TCP along d
            pu = phase == PUSH
            err = force_des - f_along
            integ = torch.where(pu, (integ + err * dt).clamp(0, cfg.max_lead / cfg.ki), integ)
            v_along = (u.agent.tcp.linear_velocity * d3).sum(dim=1)
            lead = (cfg.kp * err + cfg.ki * integ + self._kv * v_along.clamp_min(0)).clamp(0, cfg.max_lead)
            s_tcp = ((tcp - line_origin) * d3).sum(dim=1)
            target = torch.where(pu[:, None], line_origin + (s_tcp + lead)[:, None] * d3, target)
            push_time = torch.where(pu, push_time + dt, push_time)
            lost_time = torch.where(pu & ~in_contact, lost_time + dt, torch.zeros_like(lost_time))
            push_over = pu & ((push_time >= duration) | (lost_time > cfg.contact_loss_grace))

            # ---- phase transitions (each env moves at most one phase per step)
            new_phase = phase.clone()
            new_phase[advance] = phase[advance] + 1
            new_phase[touched] = PUSH
            new_phase[missed] = BACKOFF
            new_phase[push_over] = BACKOFF
            # a zero-duration push skips straight to backoff once contact is made
            new_phase[touched & (duration <= 0)] = BACKOFF
            new_phase[bumped] = RAISE
            changed = new_phase != phase
            # goals for the phases we just entered; stop where we are when backing off / bumping
            stop_here = changed & ((new_phase == BACKOFF) | bumped)
            target = torch.where(stop_here[:, None], tcp, target)
            goal_backoff = torch.cat([tcp[:, :2] - cfg.retreat_distance * d, tcp[:, 2:3]], dim=1)
            goal_by_phase = {
                TRAVEL: torch.cat([pre_xy, safe_z[:, None]], dim=1),
                DESCEND: torch.cat([pre_xy, z_push[:, None]], dim=1),
                BACKOFF: goal_backoff,
                RAISE: torch.cat([target[:, :2], safe_z[:, None]], dim=1),
                HOME: home,
            }
            for ph, g in goal_by_phase.items():
                m = changed & (new_phase == ph)
                goal = torch.where(m[:, None], g, goal)
            phase = new_phase
            phase_time = torch.where(changed, torch.zeros_like(phase_time), phase_time + dt)

            # untwist the gripper on the way home so every observation shows the same arm pose
            homing = (phase == HOME) | (phase == DONE)
            euler = torch.where(homing[:, None], self._ee_euler0, self._ee_euler)
            self._sim_step(self._low_level_action(target, euler))

            rec["phase"].append(phase.clone())
            rec["tcp_pos"].append(u.agent.tcp.pose.p.clone())
            rec["target_pos"].append(target.clone())
            rec["obj_pose"].append(u.obj.pose.raw_pose.clone())
            rec["contact_force"].append(u.pusher_contact_force.clone())
            rec["force_along_dir"].append((u.pusher_contact_force * d3).sum(dim=1))
            if cfg.record_frames:
                frames.append(u.render()[: cfg.record_envs].cpu())

        # one full env step (holding the home pose) to compute the observation, reward and info
        obs, reward, terminated, truncated, info = u.step(self._low_level_action(target, self._ee_euler0))

        push_info = {k: torch.stack(v) if v else None for k, v in rec.items()}  # (steps, num_envs, ...)
        phases = push_info["phase"]
        pushing = phases == PUSH
        f = push_info["force_along_dir"]
        push_info.update(
            requested_contact=p_req,  # p_contact as given
            contact_point=contact_point,  # where contact actually happened (NaN if no contact)
            collided=collided,  # bumped into the block while moving into position
            direction=d,
            desired_force=force_des,
            duration=duration,
            made_contact=made_contact,
            push_time=push_time,  # actual time spent in PUSH (s)
            mean_push_force=(f * pushing).sum(0) / pushing.sum(0).clamp_min(1),
            block_displacement=u.obj.pose.p - block_p0,
            timed_out=timed_out,  # some waypoint was not reached in time
            finished=phase == DONE,
            dt=dt,
        )
        if cfg.record_frames:
            push_info["frames"] = torch.stack(frames) if frames else None
        return obs, reward, terminated, truncated, info, push_info


def make_push_env(
    num_envs: int = 1,
    physics=None,
    push_config: Optional[PushPrimitiveConfig] = None,
    control_freq: int = 50,
    sim_freq: int = 100,
    obs_mode: str = "rgbd",
    **kwargs,
) -> PushPrimitiveWrapper:
    """Create PushBlock-v1 with the push-primitive action space.

    obs_mode: "rgbd" (default), "pointcloud", "state", "state_dict", "rgb", ...
    Extra kwargs go to gym.make (e.g. sim_backend, render_mode, sensor_configs).
    """
    import env as _env  # noqa: F401  registers PushBlock-v1

    base = gym.make(
        "PushBlock-v1",
        num_envs=num_envs,
        physics=physics,
        obs_mode=obs_mode,
        control_mode="pd_ee_pose",
        track_contact_force=True,
        sim_config=dict(sim_freq=sim_freq, control_freq=control_freq),
        **kwargs,
    )
    return PushPrimitiveWrapper(base, push_config)
