"""Push-primitive action space for PushTPhysics-v1 (panda_stick robot).

Each action is one push with the stick, the same parameterization as the block env's
primitive:

    a = (p_contact, d_push, F_push, T)

  p_contact  (3,)  the intended contact point: where the stick's side should touch the T, in
                   world coordinates (the frame of the point cloud and object poses). Any 3D
                   point; it need not be an observed one. The controller uses no knowledge of
                   the T: the stick tip goes to standoff behind p_contact (along -d_push) at
                   height p_contact.z and creeps along d_push until it feels contact, at most
                   approach_extra past p_contact. Then:
                     - wrong-but-close p: contact happens a bit earlier/later along the line;
                       info["push"]["contact_point"] is where it actually happened
                     - p beside the T: nothing is touched, the stick backs off (made_contact False)
                     - p inside the T: the stick lands on / bumps into the T while moving into
                       position; it stops and lifts away (collided True). Whatever the bump did
                       to the T is part of the outcome.
                   p.z is clamped to [min_push_height, max_push_height]: the stick is a vertical
                   cylinder (10 cm long, radius 8 mm, tip at the TCP), so any tip height between
                   the table and the T's top (4 cm) gives the same side contact.
  d_push     (2,)  push direction in the world xy plane (normalised internally).
  F_push     ()    desired pushing force in N, in [0, max_force].
  T          ()    push duration in s, in [0, max_duration].

As a flat Box action this is [px, py, pz, dx, dy, F, T] with shape (num_envs, 7). A dict with
keys "contact", "direction", "force" and "duration" also works.

One step() runs the whole primitive with the arm's end-effector pose controller, for all
parallel envs at once, each with its own phase. The stick keeps its reset orientation
(pointing down) throughout; a cylinder needs no yaw.

  1. LIFT     go up to safe height (only ever up)
  2. TRAVEL   move above the pre-contact point  p_contact - (standoff + stick radius) * d_push
  3. DESCEND  lower to p_contact.z
              (touching the T during TRAVEL/DESCEND -> stop, skip to RAISE, collided=True)
  4. APPROACH creep along d_push until the stick touches the T
  5. PUSH     force control for T seconds: the commanded tip target leads the actual tip along
              d_push by a PI function of (F_push - F_measured). Ends early if contact is lost
              for longer than contact_loss_grace.
  6. BACKOFF  retreat along -d_push
  7. RAISE    lift to safe height
  8. HOME     go to the home pose, behind the T's workspace as seen from the camera, so the
              next observation shows the T unoccluded. reset() also ends at the home pose.

The full per-control-step trajectory (phase, tip position, T pose, measured contact force, ...)
is returned in info["push"].
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
class PushTPrimitiveConfig:
    # action bounds
    # N. Sliding takes mu * m * g: ~11 N for the stock T, ~3-26 N over PHYSICS_RANGES. Asking for
    # more than that once the T slides only accelerates it (the force can't build up).
    max_force: float = 30.0
    max_duration: float = 2.0  # s
    contact_low: tuple = (-0.55, -0.5, 0.0)  # m, world-frame box for p_contact
    contact_high: tuple = (0.25, 0.5, 0.15)
    # geometry
    home_position: tuple = (-0.45, 0.30, 0.25)  # m, stick tip between pushes (behind the workspace)
    safe_height: float = 0.10  # m, tip height for free-space moves (the T is 4 cm tall)
    standoff: float = 0.03  # m, pre-contact gap between the stick's surface and p_contact
    pusher_radius: float = 0.008  # m, the stick's radius (TCP is on its axis)
    min_push_height: float = 0.01  # m, lowest tip height (keeps the tip off the table)
    max_push_height: float = 0.035  # m, highest tip height that still touches the 4 cm T
    approach_extra: float = 0.03  # m past p_contact before giving up
    retreat_distance: float = 0.03  # m
    # motion
    move_speed: float = 0.25  # m/s, free-space moves
    descend_speed: float = 0.1  # m/s, final descent (a landing on the T is caught early)
    approach_speed: float = 0.04  # m/s
    pos_tol: float = 0.005  # m, waypoint reached tolerance
    reach_tol: float = 0.02  # m, a TRAVEL/DESCEND that times out further than this is unreachable
    # s, a waypoint phase gives up after (its distance / speed) + phase_timeout (TRAVEL/DESCEND:
    # the push is abandoned and info["push"]["unreachable"] is set)
    phase_timeout: float = 1.0
    # force control (PI on the tip target lead distance)
    contact_threshold: float = 0.1  # N, force along d_push that counts as contact
    contact_loss_grace: float = 0.1  # s without contact before the push is aborted
    kp: float = 3e-4  # m / N
    ki: float = 1e-2  # m / (N s)
    # Extra lead = velocity_feedforward * (tip speed along d_push). The joint PD controller damps
    # motion towards a static target, so without this the applied force drops by ~D*v while the
    # T slides. None = use the arm's damping / stiffness ratio (0.1 s for panda_stick).
    velocity_feedforward: Optional[float] = None  # s
    max_lead: float = 0.08  # m
    # Height hold during APPROACH/PUSH: each control step the commanded tip height is raised by
    # height_hold_gain * (sag below p_contact.z), so the stick doesn't slide down the T's face
    # (with the T's high friction that drags the T into the table and pins it).
    height_hold_gain: float = 0.3
    max_height_offset: float = 0.03  # m
    # episode
    max_pushes: int = 10  # truncate after this many primitives
    record_frames: bool = False  # store env.render() for every control step in info["push"]["frames"]
    record_envs: Optional[int] = None  # only keep the first record_envs envs' frames (None = all)


def _yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class PushTPrimitiveWrapper(gym.Wrapper):
    """Wraps a PushTPhysics-v1 env created with control_mode="pd_ee_pose" and
    track_contact_force=True. Prefer make_push_t_env(), which sets those up."""

    def __init__(self, env: gym.Env, config: Optional[PushTPrimitiveConfig] = None):
        super().__init__(env)
        self.cfg = config or PushTPrimitiveConfig()
        u = self.env.unwrapped
        assert u.control_mode == "pd_ee_pose", "PushTPrimitiveWrapper needs control_mode='pd_ee_pose'"
        assert u.track_contact_force, "PushTPrimitiveWrapper needs track_contact_force=True"
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
        # stick orientation at reset (pointing down), kept for every move
        q = (self.u.agent.robot.pose.inv() * self.u.agent.tcp.pose).q
        self._ee_euler = matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")
        return self._go_home()

    def step(self, action):
        contact, direction, force, duration = self._parse_action(action)
        obs, reward, terminated, truncated, info, push_info = self._execute_push(contact, direction, force, duration)
        self._push_count += 1
        info = dict(info)
        info["push"] = push_info
        terminated = info["success"].clone()
        truncated = torch.full_like(terminated, self._push_count >= self.cfg.max_pushes)
        return obs, reward, terminated, truncated, info

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

    def _low_level_action(self, target_world):
        base = self.u.agent.robot.pose
        p_base = (base.inv() * Pose.create_from_pq(p=target_world)).p
        return torch.cat([p_base, self._ee_euler], dim=1)

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
        """Move the stick to the home pose. Used at reset."""
        u, cfg = self.u, self.cfg
        dt, home = u.control_timestep, self._home()
        # lift first so the stick never sweeps through the workspace low
        tcp = u.agent.tcp.pose.p
        waypoints = [torch.cat([tcp[:, :2], torch.full_like(tcp[:, 2:], max(cfg.safe_height, cfg.home_position[2]))], 1), home]
        target = tcp.clone()
        for goal in waypoints:
            for _ in range(int(cfg.phase_timeout / dt)):
                target = self._move_towards(target, goal, cfg.move_speed * dt)
                self._sim_step(self._low_level_action(target))
                if bool(((u.agent.tcp.pose.p - goal).norm(dim=1) < cfg.pos_tol).all()):
                    break
        for _ in range(9):  # let the arm settle so the first observation is static
            self._sim_step(self._low_level_action(home))
        obs, _, _, _, info = u.step(self._low_level_action(home))
        return obs, info

    # ------------------------------------------------------------ the primitive
    @torch.no_grad()
    def _execute_push(self, p_req, direction, force_des, duration):
        cfg, u = self.cfg, self.u
        n, dev, dt = u.num_envs, u.device, u.control_timestep
        d = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        z_push = p_req[:, 2].clamp(cfg.min_push_height, cfg.max_push_height)
        d3 = torch.cat([d, torch.zeros((n, 1), device=dev)], dim=1)

        # the tip (TCP) is on the stick's axis, pusher_radius behind its surface; start standoff before p
        pre_xy = p_req[:, :2] - (cfg.standoff + cfg.pusher_radius) * d
        tcp0 = u.agent.tcp.pose.p.clone()
        safe_z = torch.full((n,), cfg.safe_height, device=dev)
        home = self._home()

        phase = torch.full((n,), LIFT, dtype=torch.long, device=dev)
        phase_time = torch.zeros(n, device=dev)
        goal = torch.cat([tcp0[:, :2], torch.maximum(tcp0[:, 2], safe_z)[:, None]], dim=1)  # LIFT only goes up
        target = tcp0.clone()
        # time allowed for the current waypoint phase: its distance at move speed + phase_timeout
        phase_budget = (goal - target).norm(dim=1) / cfg.move_speed + cfg.phase_timeout
        line_origin = torch.cat([pre_xy, z_push[:, None]], dim=1)  # push line: line_origin + s * d
        approach_s = torch.zeros(n, device=dev)
        integ = torch.zeros(n, device=dev)
        z_offset = torch.zeros(n, device=dev)  # height-hold correction of the commanded tip height
        push_time = torch.zeros(n, device=dev)
        lost_time = torch.zeros(n, device=dev)
        made_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        collided = torch.zeros(n, dtype=torch.bool, device=dev)
        contact_point = torch.full((n, 3), float("nan"), device=dev)
        timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
        unreachable = torch.zeros(n, dtype=torch.bool, device=dev)
        tee_p0, tee_yaw0 = u.tee.pose.p.clone(), _yaw(u.tee.pose.q)

        rec = {k: [] for k in ["phase", "tcp_pos", "target_pos", "obj_pose", "contact_force", "force_along_dir"]}
        frames = []
        max_steps = int(30.0 / dt)  # safety cap; the loop ends as soon as every env is DONE
        for _ in range(max_steps):
            if bool((phase == DONE).all()):
                break
            tcp = u.agent.tcp.pose.p
            f_vec = u.pusher_contact_force
            f_along = (f_vec * d3).sum(dim=1)  # force on the T along the push direction
            in_contact = f_along > cfg.contact_threshold

            # ---- waypoint phases: speed-limited straight moves of the commanded target
            wp = torch.zeros(n, dtype=torch.bool, device=dev)
            for ph in _WAYPOINT_PHASES:
                wp |= phase == ph
            speed = torch.where(phase == DESCEND, cfg.descend_speed, cfg.move_speed)
            target = torch.where(wp[:, None], self._move_towards(target, goal, (speed * dt)[:, None]), target)
            reached = wp & ((tcp - goal).norm(dim=1) < cfg.pos_tol)
            expired = wp & (phase_time > phase_budget)
            timed_out |= expired & ~reached
            advance = reached | expired
            # the pre-contact point can't be reached (out of the arm's reach): don't descend or
            # push from wherever the stick ended up; lift away and go home
            # (near the arm's base the last few mm converge slowly: within reach_tol is close enough)
            far = (tcp - goal).norm(dim=1) > cfg.reach_tol
            cant_reach = expired & far & ((phase == TRAVEL) | (phase == DESCEND))
            unreachable |= cant_reach

            # ---- APPROACH: creep along the push line until contact
            ap = phase == APPROACH
            approach_s = torch.where(ap, approach_s + cfg.approach_speed * dt, approach_s)
            target = torch.where(ap[:, None], line_origin + approach_s[:, None] * d3, target)
            touched = ap & in_contact
            missed = ap & ~in_contact & (approach_s > cfg.standoff + cfg.approach_extra)
            made_contact |= touched
            # where contact actually happened: the stick's surface at first touch
            contact_point = torch.where(touched[:, None], tcp + cfg.pusher_radius * d3, contact_point)

            # ---- bumping into the T while moving into position: stop and lift away
            moving_in = (phase == TRAVEL) | (phase == DESCEND)
            bumped = moving_in & (f_vec.norm(dim=1) > cfg.contact_threshold)
            collided |= bumped

            # ---- PUSH: PI force control on how far the target leads the tip along d
            pu = phase == PUSH
            err = force_des - f_along
            integ = torch.where(pu, (integ + err * dt).clamp(0, cfg.max_lead / cfg.ki), integ)
            v_along = (u.agent.tcp.linear_velocity * d3).sum(dim=1)
            lead = (cfg.kp * err + cfg.ki * integ + self._kv * v_along.clamp_min(0)).clamp(0, cfg.max_lead)
            s_tcp = ((tcp - line_origin) * d3).sum(dim=1)
            target = torch.where(pu[:, None], line_origin + (s_tcp + lead)[:, None] * d3, target)
            # ---- hold the tip height in contact: under the pushing load the arm sags, and a stick
            # sliding down the T's face drags it into the table with friction (up to mu * push
            # force), which can pin it. Integral correction of the commanded height.
            contact_ph = ap | pu
            z_offset = torch.where(contact_ph, (z_offset + cfg.height_hold_gain * (z_push - tcp[:, 2])).clamp(
                -cfg.max_height_offset, cfg.max_height_offset), z_offset)
            target = torch.where(contact_ph[:, None], torch.cat([target[:, :2], (z_push + z_offset)[:, None]], 1), target)
            push_time = torch.where(pu, push_time + dt, push_time)
            lost_time = torch.where(pu & ~in_contact, lost_time + dt, torch.zeros_like(lost_time))
            push_over = pu & ((push_time >= duration) | (lost_time > cfg.contact_loss_grace))

            # ---- phase transitions (each env moves at most one phase per step)
            new_phase = phase.clone()
            new_phase[advance] = phase[advance] + 1
            new_phase[touched] = PUSH
            new_phase[missed] = BACKOFF
            new_phase[push_over] = BACKOFF
            new_phase[touched & (duration <= 0)] = BACKOFF  # zero-duration push: back off at contact
            new_phase[bumped] = RAISE
            new_phase[cant_reach] = RAISE
            changed = new_phase != phase
            # goals for the phases we just entered; stop where we are when backing off / bumping
            stop_here = changed & ((new_phase == BACKOFF) | bumped | cant_reach)
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
            new_speed = torch.where(phase == DESCEND, cfg.descend_speed, cfg.move_speed)
            phase_budget = torch.where(changed, (goal - target).norm(dim=1) / new_speed + cfg.phase_timeout, phase_budget)

            self._sim_step(self._low_level_action(target))

            rec["phase"].append(phase.clone())
            rec["tcp_pos"].append(u.agent.tcp.pose.p.clone())
            rec["target_pos"].append(target.clone())
            rec["obj_pose"].append(u.tee.pose.raw_pose.clone())
            rec["contact_force"].append(u.pusher_contact_force.clone())
            rec["force_along_dir"].append((u.pusher_contact_force * d3).sum(dim=1))
            if cfg.record_frames:
                frames.append(u.render()[: cfg.record_envs].cpu())

        # one full env step (holding the target) to compute the observation, reward and info
        obs, reward, terminated, truncated, info = u.step(self._low_level_action(target))

        push_info = {k: torch.stack(v) if v else None for k, v in rec.items()}  # (steps, num_envs, ...)
        pushing = push_info["phase"] == PUSH
        f = push_info["force_along_dir"]
        dyaw = torch.remainder(_yaw(u.tee.pose.q) - tee_yaw0 + np.pi, 2 * np.pi) - np.pi
        push_info.update(
            requested_contact=p_req,  # p_contact as given
            contact_point=contact_point,  # where contact actually happened (NaN if no contact)
            collided=collided,  # bumped into the T while moving into position
            direction=d,
            desired_force=force_des,
            duration=duration,
            made_contact=made_contact,
            push_time=push_time,  # actual time spent in PUSH (s)
            mean_push_force=(f * pushing).sum(0) / pushing.sum(0).clamp_min(1),
            tee_displacement=u.tee.pose.p - tee_p0,
            tee_rotation=dyaw,  # change of the T's yaw (rad, in [-pi, pi))
            timed_out=timed_out,  # some waypoint was not reached in time
            unreachable=unreachable,  # the pre-contact point was out of reach: no push was made
            finished=phase == DONE,
            dt=dt,
        )
        if cfg.record_frames:
            push_info["frames"] = torch.stack(frames) if frames else None
        return obs, reward, terminated, truncated, info, push_info


def make_push_t_env(
    num_envs: int = 1,
    physics=None,
    push_config: Optional[PushTPrimitiveConfig] = None,
    control_freq: int = 50,
    sim_freq: int = 100,
    obs_mode: str = "rgbd",
    **kwargs,
) -> PushTPrimitiveWrapper:
    """Create PushTPhysics-v1 with the push-primitive action space.

    obs_mode: "rgbd" (default), "pointcloud", "state", "state_dict", "rgb", ...
    Extra kwargs go to gym.make (e.g. sim_backend, render_mode, sensor_configs).
    """
    from . import env as _env  # noqa: F401  registers PushTPhysics-v1

    base = gym.make(
        "PushTPhysics-v1",
        num_envs=num_envs,
        physics=physics,
        obs_mode=obs_mode,
        control_mode="pd_ee_pose",
        track_contact_force=True,
        sim_config=dict(sim_freq=sim_freq, control_freq=control_freq),
        **kwargs,
    )
    return PushTPrimitiveWrapper(base, push_config)
