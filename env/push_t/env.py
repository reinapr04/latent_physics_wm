"""PushTPhysics-v1: ManiSkill's PushT-v1 with configurable T physics.

PushTPhysicsEnv extends mani_skill's PushTEnv (same panda_stick robot, white table, T shape,
goal T, end-zone marker, 90%-overlap success, observations and dense reward) and changes only
what is needed:
  - the T's physics (mass, T-table friction, restitution, damping) are configurable, as
    fixed values or (low, high) ranges sampled per parallel env, or set exactly per env with
    set_physics(); the realised values are in info["physics"]. Defaults are stock PushT's
    actual values (0.7 kg, T-table friction 1.65, angular damping 0.05). The stick-T contact
    is kept smooth (friction 0.25) in every env; see PushTPhysicsConfig.
  - a workaround for a SAPIEN GPU material bug (see _init_material_pool)
  - optional tracking of the contact force the stick applies to the T
  - a fixed camera framing the T workspace (for point clouds)
  - reset options to place the T and the goal explicitly ((x, y, yaw) per env); stock PushT
    has one fixed goal pose, so the goal is made per-env (success and reward use it)

How the physics is applied: PushTEnv._load_scene builds the T (a single material and mass
shared by all envs) inside a nested function, so it can't be swapped out. Instead this env
calls it unchanged and then, before the simulation starts, gives each env's T body its own
material, mass (inertia scaled with it) and damping, and each env's table its own material.

Important: PhysX fixes these properties when the scene is built, so they are applied during
reconfiguration, not on every reset. New values are sampled from ranges only when you call
``env.reset(options={"reconfigure": True})`` (or set ``reconfiguration_freq``).

Example:
    import gymnasium as gym
    import env.push_t  # registers PushTPhysics-v1
    from env.push_t import PushTPhysicsConfig, PHYSICS_RANGES

    e = gym.make("PushTPhysics-v1", num_envs=4, physics=PushTPhysicsConfig(**PHYSICS_RANGES))
    obs, info = e.reset(seed=0)
    info["physics"]  # dict of (num_envs,) tensors with the actual values per env
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import PandaStick
from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

# A physics parameter is either a fixed value or a (low, high) range sampled uniformly.
Param = Union[float, tuple[float, float]]

# Stock PushT's T as it actually is in the simulator. PushTEnv sets T_mass = 0.8 via
# builder._mass, but SAPIEN ignores that and computes the mass from the shape at the default
# density (700 cm^3 * 1000 kg/m^3 = 0.7 kg); angular damping is SAPIEN's default 0.05.
STOCK_MASS = 0.7
STOCK_ANGULAR_DAMPING = 0.05
# Stock T-table friction: PhysX averages the two materials, (T 3.0 + table 0.3) / 2.
STOCK_TABLE_FRICTION = (3.0 + 0.3) / 2

# Ranges centred on stock PushT's values (mass 0.7 kg, T-table friction 1.65), e.g. for
# collecting dynamics-model data: PushTPhysicsConfig(**PHYSICS_RANGES).
PHYSICS_RANGES = dict(mass=(0.35, 1.05), static_friction=(0.8, 2.5), dynamic_friction=(0.8, 2.5))

TEE_HALF_THICKNESS = 0.02  # m, stock PushT's T is 4 cm thick

# Point-cloud crop for this task (world frame, m; dynamics_model.geometry.observed_cloud's
# `workspace`): the table area the T can be on. The T spawns as far back as x = -0.36 (and can
# be pushed further); the robot's base starts at x = -0.53; z_max drops the parked stick.
WORKSPACE = ((-0.45, -0.5, -0.01), (0.5, 0.5, 0.15))


@dataclass
class PushTPhysicsConfig:
    """Physical properties of the T and its contact with the table.

    Each sampled field can be a float (fixed) or a (low, high) tuple (sampled uniformly per
    parallel env at each reconfiguration). Defaults are stock PushT's actual values.

    static_friction / dynamic_friction are the T-TABLE friction coefficients, exactly. PhysX
    averages the friction of the two touching materials, so they are realised with a fixed,
    low-friction T material (tee_material_friction) and a per-env table material
    (2 * friction - tee_material_friction). This keeps the stick-T contact smooth and the
    same in every env ((stick 0.3 + 0.2) / 2 = 0.25): with stock materials the T's high
    friction also applies to the stick, which can drag the T into the table and pin it.
    """

    mass: Param = STOCK_MASS  # kg
    static_friction: Param = STOCK_TABLE_FRICTION  # T-table
    dynamic_friction: Param = STOCK_TABLE_FRICTION  # T-table
    restitution: Param = 0.0  # bounciness, in [0, 1]
    linear_damping: Param = 0.0
    angular_damping: Param = STOCK_ANGULAR_DAMPING

    # Friction of the T's own material (fixed; not sampled). Sets the stick-T friction.
    tee_material_friction: float = 0.2

    # Names of the fields that get sampled per env and reported in info["physics"].
    @staticmethod
    def tee_fields() -> list[str]:
        return [f.name for f in fields(PushTPhysicsConfig) if f.name != "tee_material_friction"]

    @classmethod
    def from_dict(cls, d: dict) -> "PushTPhysicsConfig":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        out = {}
        for name in self.tee_fields():
            v = getattr(self, name)
            out[name] = float(rng.uniform(v[0], v[1])) if isinstance(v, tuple) else float(v)
        return out


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """z rotations (b,) -> quaternions (b, 4) wxyz."""
    q = torch.zeros((len(yaw), 4), device=yaw.device)
    q[:, 0] = (yaw / 2).cos()
    q[:, 3] = (yaw / 2).sin()
    return q


@register_env("PushTPhysics-v1", max_episode_steps=100)
class PushTPhysicsEnv(PushTEnv):
    """
    **Task Description:** PushT-v1 (push the T onto the goal T), with configurable T physics.

    **Randomizations:**
    - as PushT-v1: T position in a spawn box near the goal, uniform z rotation; goal T fixed.
      Reset options "tee_pose" / "goal_pose" ((num_envs, 3) of x, y, yaw) override them
    - T physics as specified by `physics` (fixed values or per-env ranges)

    **Success Conditions:** as PushT-v1 (the T covers 90% of the goal T's area), with the goal
    of each env
    """

    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    def __init__(
        self,
        *args,
        physics: Union[PushTPhysicsConfig, dict, None] = None,
        track_contact_force: bool = False,
        **kwargs,
    ):
        if physics is None:
            physics = PushTPhysicsConfig()
        elif isinstance(physics, dict):
            physics = PushTPhysicsConfig.from_dict(physics)
        self.physics = physics
        self._physics_override = None
        # If True, the contact force the stick applies to the T is averaged over every physics
        # sub-step of a control step and exposed as self.pusher_contact_force /
        # info["pusher_contact_force"] (num_envs, 3), in N, world frame.
        self.track_contact_force = track_contact_force
        super().__init__(*args, **kwargs)

    def set_physics(self, per_env: Optional[list[dict]]):
        """Use these exact T physics (one dict per parallel env, e.g. {"mass": 0.6,
        "static_friction": 2.0}) from the next reconfiguration on, instead of sampling from
        `physics`; unspecified fields are still sampled. Takes effect at
        env.reset(options={"reconfigure": True}). None = sample again."""
        if per_env is not None:
            assert len(per_env) == self.num_envs, f"need {self.num_envs} dicts, got {len(per_env)}"
            unknown = set().union(*per_env) - set(PushTPhysicsConfig.tee_fields())
            assert not unknown, f"unknown physics fields: {unknown}"
        self._physics_override = per_env

    @property
    def _default_sensor_configs(self):
        # Fixed camera on the far side of the table, looking down at the T's workspace (spawn
        # box and goal). Override with gym.make(..., sensor_configs=dict(width=..., pose=...)).
        pose = sapien_utils.look_at(eye=[0.3, 0.2, 0.6], target=[-0.15, -0.05, 0.0])
        return [CameraConfig("base_camera", pose=pose, width=256, height=256, fov=1.0, near=0.01, far=10)]

    # ------------------------------------------------------------------ scene
    def _init_material_pool(self):
        """Create all PhysX materials once and reuse them across reconfigurations.

        Workaround for a GPU-sim bug (SAPIEN 3.0.3): a newly created PhysxMaterial that
        reuses the slot of a freed one keeps stale GPU data (it acts as if friction were
        ~0). Materials are freed when the scene is rebuilt, so creating fresh ones on
        every reconfigure silently breaks one env's friction. Instead we create a
        throwaway material (absorbs a slot freed during env setup), then one T material and
        one table material per env, and only mutate their values afterwards.
        """
        if hasattr(self, "_tee_materials"):
            return
        self._sacrificial_material = sapien.physx.PhysxMaterial(0.5, 0.5, 0.0)
        self._tee_materials = [sapien.physx.PhysxMaterial(0.5, 0.5, 0.0) for _ in range(self.num_envs)]
        self._table_materials = [sapien.physx.PhysxMaterial(0.5, 0.5, 0.0) for _ in range(self.num_envs)]

    def _load_scene(self, options: dict):
        self._init_material_pool()
        super()._load_scene(options)  # stock scene: table, T, goal T, end-zone marker, overlap setup
        self._apply_tee_physics()

        # stock PushT has one fixed goal (a single world->goal transform); make it per env
        self.goal_yaw = torch.full((self.num_envs,), float(self.goal_z_rot), device=self.device)
        self.world_to_goal_trans = self.world_to_goal_trans.expand(self.num_envs, 3, 3).clone()

        # the stick's collision shape is on panda_hand (the tcp link has no collision shape)
        self._pusher_links = [sapien_utils.get_obj_by_name(self.agent.robot.get_links(), "panda_hand")]
        self._reset_contact_accumulator()
        self.pusher_contact_force = torch.zeros((self.num_envs, 3), device=self.device)

    def _apply_tee_physics(self):
        """Give each env's T its own mass, damping and restitution, and each env's table the
        material that makes the T-table friction the configured value (before the sim starts)."""
        names = PushTPhysicsConfig.tee_fields()
        per_env = [self.physics.sample(self._batched_episode_rng[i]) for i in range(self.num_envs)]
        if self._physics_override is not None:
            # explicit per-env values (set_physics); unspecified fields keep the sampled value
            per_env = [{**sampled, **given} for sampled, given in zip(per_env, self._physics_override)]
        mu_tee = self.physics.tee_material_friction
        for p in per_env:
            assert min(p["static_friction"], p["dynamic_friction"]) >= mu_tee / 2, \
                f"T-table friction must be >= tee_material_friction / 2 = {mu_tee / 2}"
        # table: PhysX uses (T material + table material) / 2, so table = 2 * target - T material
        for body, i in zip(self.table_scene.table._bodies, self.table_scene.table._scene_idxs.tolist()):
            p = per_env[i]
            material = self._table_materials[i]
            material.static_friction = 2 * p["static_friction"] - mu_tee
            material.dynamic_friction = 2 * p["dynamic_friction"] - mu_tee
            material.restitution = body.collision_shapes[0].physical_material.restitution
            for shape in body.collision_shapes:
                shape.set_physical_material(material)
        for body, i in zip(self.tee._bodies, self.tee._scene_idxs.tolist()):
            p = per_env[i]
            material = self._tee_materials[i]
            material.static_friction = mu_tee
            material.dynamic_friction = mu_tee
            material.restitution = p["restitution"]
            for shape in body.collision_shapes:
                shape.set_physical_material(material)
            # same shape, so the inertia scales with the mass
            body.inertia = body.inertia * (p["mass"] / body.mass)
            body.mass = p["mass"]
            body.linear_damping = p["linear_damping"]
            body.angular_damping = p["angular_damping"]
        # (num_envs,) tensors of the realised physics values, reported in info["physics"].
        self.tee_physics = {
            k: torch.tensor([p[k] for p in per_env], dtype=torch.float32, device=self.device) for k in names
        }

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # PushTEnv places the goal T (fixed pose) and the T (random, near the goal)
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # options["tee_pose"] / options["goal_pose"] ((num_envs, 3): x, y, yaw) place the
            # T / goal explicitly (e.g. random goals, or the same start for several runs)
            if options.get("tee_pose") is not None:
                xyy = torch.as_tensor(options["tee_pose"], dtype=torch.float32).to(self.device)[env_idx]
                p = torch.cat([xyy[:, :2], torch.full((b, 1), TEE_HALF_THICKNESS + 1e-3)], 1)
                self.tee.set_pose(Pose.create_from_pq(p=p, q=_yaw_quat(xyy[:, 2])))
            goal_yaw = torch.full((b,), float(self.goal_z_rot))
            goal_xy = self.goal_offset.to(self.device).expand(b, 2)
            if options.get("goal_pose") is not None:
                xyy = torch.as_tensor(options["goal_pose"], dtype=torch.float32).to(self.device)[env_idx]
                goal_xy, goal_yaw = xyy[:, :2], xyy[:, 2]
                p = torch.cat([goal_xy, torch.full((b, 1), 1e-3)], 1)
                self.goal_tee.set_pose(Pose.create_from_pq(p=p, q=_yaw_quat(goal_yaw)))
            self.goal_yaw[env_idx] = goal_yaw % (2 * np.pi)
            goal_to_world = torch.eye(3).repeat(b, 1, 1)
            goal_to_world[:, :2, :2] = torch.stack(
                [torch.stack([goal_yaw.cos(), -goal_yaw.sin()], -1), torch.stack([goal_yaw.sin(), goal_yaw.cos()], -1)], -2)
            goal_to_world[:, :2, 2] = goal_xy
            self.world_to_goal_trans[env_idx] = torch.linalg.inv(goal_to_world)

            self.tee.set_linear_velocity(torch.zeros((b, 3)))
            self.tee.set_angular_velocity(torch.zeros((b, 3)))
            self.pusher_contact_force[env_idx] = 0

    # ------------------------------------------------------- contact sensing
    def _reset_contact_accumulator(self):
        self._contact_force_sum = torch.zeros((self.num_envs, 3), device=self.device)
        self._contact_force_count = 0

    def _after_simulation_step(self):
        if not self.track_contact_force:
            return
        for link in self._pusher_links:
            # force exerted on the T by this link
            self._contact_force_sum += self.scene.get_pairwise_contact_forces(self.tee, link).to(self.device)
        self._contact_force_count += 1

    def _after_control_step(self):
        if not self.track_contact_force:
            return
        self.pusher_contact_force = self._contact_force_sum / max(self._contact_force_count, 1)
        self._reset_contact_accumulator()

    # ------------------------------------------------------------- task logic
    def evaluate(self):
        # PushTEnv's success (overlap with this env's goal T), plus the physics (and contact force)
        info = super().evaluate()
        info["physics"] = self.tee_physics
        if self.track_contact_force:
            info["pusher_contact_force"] = self.pusher_contact_force
        return info

    def compute_dense_reward(self, obs, action, info):
        # PushTEnv's reward compares the T's rotation with the class-level goal rotation;
        # use each env's goal rotation instead (everything else is the stock reward)
        stock = self.goal_z_rot
        self.goal_z_rot = self.goal_yaw
        try:
            return super().compute_dense_reward(obs, action, info)
        finally:
            self.goal_z_rot = stock
