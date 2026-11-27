"""PushBlock-v1: ManiSkill's PushCube-v1 with configurable block physics.

PushBlockEnv extends mani_skill's PushCubeEnv (same robot, table, 4 cm cube, goal target,
success condition, observations and dense reward) and changes only what is needed:
  - the block's physics (mass, friction, restitution, damping) and the table friction are
    configurable, as fixed values or (low, high) ranges sampled per parallel env, or set
    exactly per env with set_physics(); the realised values are in info["physics"]
  - a workaround for a SAPIEN GPU material bug (see _init_material_pool)
  - optional tracking of the contact force the gripper applies to the block
  - a fixed camera framing the workspace (for point clouds)
  - reset options to place the block and the goal explicitly

Important: PhysX fixes these properties when the scene is built, so they are applied during
reconfiguration, not on every reset. New values are sampled from ranges only when you call
``env.reset(options={"reconfigure": True})`` (or set ``reconfiguration_freq``).

Example:
    import gymnasium as gym
    import env  # registers PushBlock-v1
    from env import BlockPhysicsConfig

    e = gym.make(
        "PushBlock-v1",
        num_envs=4,
        physics=BlockPhysicsConfig(mass=0.5, static_friction=(0.2, 1.0), dynamic_friction=(0.2, 1.0)),
    )
    obs, info = e.reset(seed=0)
    info["physics"]  # dict of (num_envs,) tensors with the actual values per env
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose

# A physics parameter is either a fixed value or a (low, high) range sampled uniformly.
Param = Union[float, tuple[float, float]]


@dataclass
class BlockPhysicsConfig:
    """Physical properties of the pushed block (and the table surface it slides on).

    Each block field can be a float (fixed) or a (low, high) tuple (sampled uniformly
    per parallel env at each reconfiguration).
    """

    mass: Param = 0.1  # kg
    static_friction: Param = 0.5
    dynamic_friction: Param = 0.5
    restitution: Param = 0.0  # bounciness, in [0, 1]
    linear_damping: Param = 0.0
    angular_damping: Param = 0.0
    # centre of mass offset from the geometric centre, in the block's own frame (m). 0 = uniform block
    com_x: Param = 0.0
    com_y: Param = 0.0

    # Table surface friction. None keeps ManiSkill's default. PhysX averages the
    # friction of the two touching materials, so the block-table friction is
    # (block_friction + table_friction) / 2.
    table_static_friction: Optional[float] = None
    table_dynamic_friction: Optional[float] = None

    # Names of the block fields that get sampled per env and reported in info["physics"].
    @staticmethod
    def block_fields() -> list[str]:
        return [f.name for f in fields(BlockPhysicsConfig) if not f.name.startswith("table_")]

    @classmethod
    def from_dict(cls, d: dict) -> "BlockPhysicsConfig":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        out = {}
        for name in self.block_fields():
            v = getattr(self, name)
            out[name] = float(rng.uniform(v[0], v[1])) if isinstance(v, tuple) else float(v)
        return out


def _box_inertia(mass: float, half_size: float) -> np.ndarray:
    # Solid cube with edge 2h: I = m * (2h)^2 / 6 on each principal axis.
    return np.full(3, mass * (2 * half_size) ** 2 / 6.0)


@register_env("PushBlock-v1", max_episode_steps=50)
class PushBlockEnv(PushCubeEnv):
    """
    **Task Description:** PushCube-v1 (push the block into the goal region in front of it),
    with configurable block physics.

    **Randomizations:**
    - as PushCube-v1: block xy uniformly in [-0.1, 0.1]^2, placed flat; goal at block xy +
      [0.1 + goal_radius, 0]. Reset options "block_xy" / "goal_xy" ((num_envs, 2)) override them
    - block physics as specified by `physics` (fixed values or per-env ranges)

    **Success Conditions:** as PushCube-v1 (block xy within goal_radius of the goal, on the table)
    """

    SUPPORTED_ROBOTS = ["panda"]  # contact tracking uses the Panda gripper's links
    agent: Panda

    def __init__(
        self,
        *args,
        physics: Union[BlockPhysicsConfig, dict, None] = None,
        track_contact_force: bool = False,
        **kwargs,
    ):
        if physics is None:
            physics = BlockPhysicsConfig()
        elif isinstance(physics, dict):
            physics = BlockPhysicsConfig.from_dict(physics)
        self.physics = physics
        self._physics_override = None
        # If True, the contact force the gripper (fingers + hand) applies to the block is
        # averaged over every physics sub-step of a control step and exposed as
        # self.pusher_contact_force / info["pusher_contact_force"] (num_envs, 3), in N, world frame.
        self.track_contact_force = track_contact_force
        super().__init__(*args, **kwargs)

    def set_physics(self, per_env: Optional[list[dict]]):
        """Use these exact block physics (one dict per parallel env, e.g. {"mass": 0.3,
        "static_friction": 0.6}) from the next reconfiguration on, instead of sampling from
        `physics`. Takes effect at env.reset(options={"reconfigure": True}). None = sample again."""
        if per_env is not None:
            assert len(per_env) == self.num_envs, f"need {self.num_envs} dicts, got {len(per_env)}"
            unknown = set().union(*per_env) - set(BlockPhysicsConfig.block_fields())
            assert not unknown, f"unknown physics fields: {unknown}"
        self._physics_override = per_env

    @property
    def _default_sensor_configs(self):
        # Fixed third-person camera on the far side of the table from the robot, framing the
        # workspace around the block (~0.8 m across), so a 4 cm block is ~12 px wide. Override
        # with gym.make(..., sensor_configs=dict(width=..., height=..., pose=...)).
        pose = sapien_utils.look_at(eye=[0.6, 0.25, 0.6], target=[0.05, 0.0, 0.0])
        return [CameraConfig("base_camera", pose=pose, width=256, height=256, fov=1.0, near=0.01, far=10)]

    # ------------------------------------------------------------------ scene
    def _init_material_pool(self):
        """Create all PhysX materials once and reuse them across reconfigurations.

        Workaround for a GPU-sim bug (SAPIEN 3.0.3): a newly created PhysxMaterial that
        reuses the slot of a freed one keeps stale GPU data (it acts as if friction were
        ~0). Materials are freed when the scene is rebuilt, so creating fresh ones on
        every reconfigure silently breaks one env's friction. Instead we create a
        throwaway material (absorbs a slot freed during env setup), then one material
        per env plus one for the table, and only mutate their values afterwards.
        """
        if hasattr(self, "_block_materials"):
            return
        self._sacrificial_material = sapien.physx.PhysxMaterial(0.5, 0.5, 0.0)
        self._block_materials = [sapien.physx.PhysxMaterial(0.5, 0.5, 0.0) for _ in range(self.num_envs)]
        self._table_material = sapien.physx.PhysxMaterial(0.5, 0.5, 0.0)

    def _load_scene(self, options: dict):
        # Replaces PushCubeEnv._load_scene, which builds one cube with a fixed material shared
        # by all envs: here every env gets its own block with its own physics.
        self._init_material_pool()

        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._apply_table_friction()

        # Build one block per parallel env so each can have its own physics, then merge
        # them into a single Actor view for batched pose/velocity access.
        names = BlockPhysicsConfig.block_fields()
        per_env = [self.physics.sample(self._batched_episode_rng[i]) for i in range(self.num_envs)]
        if self._physics_override is not None:
            # explicit per-env values (set_physics); unspecified fields keep the sampled value
            per_env = [{**sampled, **given} for sampled, given in zip(per_env, self._physics_override)]
        blocks = []
        h = self.cube_half_size
        for i, p in enumerate(per_env):
            material = self._block_materials[i]
            material.static_friction = p["static_friction"]
            material.dynamic_friction = p["dynamic_friction"]
            material.restitution = p["restitution"]
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[h] * 3, material=material)
            builder.add_box_visual(
                half_size=[h] * 3,
                material=sapien.render.RenderMaterial(base_color=np.array([12, 42, 160, 255]) / 255),
            )
            builder.set_mass_and_inertia(p["mass"], sapien.Pose(p=[p["com_x"], p["com_y"], 0.0]), _box_inertia(p["mass"], h))
            builder.set_scene_idxs([i])
            builder.set_initial_pose(sapien.Pose(p=[0, 0, h]))
            block = builder.build(name=f"block-{i}")
            block.linear_damping = p["linear_damping"]
            block.angular_damping = p["angular_damping"]
            blocks.append(block)
        self.obj: Actor = Actor.merge(blocks, name="block")

        # (num_envs,) tensors of the realised physics values, reported in info["physics"].
        self.block_physics = {
            k: torch.tensor([p[k] for p in per_env], dtype=torch.float32, device=self.device) for k in names
        }
        self.block_half_size = torch.full((self.num_envs,), h, dtype=torch.float32, device=self.device)

        hand = sapien_utils.get_obj_by_name(self.agent.robot.get_links(), "panda_hand")
        self._pusher_links = [self.agent.finger1_link, self.agent.finger2_link, hand]
        self._reset_contact_accumulator()
        self.pusher_contact_force = torch.zeros((self.num_envs, 3), device=self.device)

        # as in PushCubeEnv
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _apply_table_friction(self):
        sf, df = self.physics.table_static_friction, self.physics.table_dynamic_friction
        if sf is None and df is None:
            return
        mat = self._table_material
        for body in self.table_scene.table._bodies:
            for shape in body.collision_shapes:
                old = shape.physical_material
                mat.static_friction = old.static_friction if sf is None else sf
                mat.dynamic_friction = old.dynamic_friction if df is None else df
                mat.restitution = old.restitution
                shape.set_physical_material(mat)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # PushCubeEnv places the block randomly and the goal in front of it
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # options["block_xy"] / options["goal_xy"] ((num_envs, 2)) place the block / goal
            # explicitly (e.g. the same start for several evaluation runs)
            if options.get("block_xy") is not None or options.get("goal_xy") is not None:
                xyz = self.obj.pose.p[env_idx].clone()
                if options.get("block_xy") is not None:
                    xyz[..., :2] = torch.as_tensor(options["block_xy"], dtype=torch.float32).to(self.device)[env_idx]
                    self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
                goal_xyz = self.goal_region.pose.p[env_idx].clone()
                goal_xyz[..., :2] = xyz[..., :2] + torch.tensor([0.1 + self.goal_radius, 0])
                if options.get("goal_xy") is not None:
                    goal_xyz[..., :2] = torch.as_tensor(options["goal_xy"], dtype=torch.float32).to(self.device)[env_idx]
                self.goal_region.set_pose(Pose.create_from_pq(p=goal_xyz, q=self.goal_region.pose.q[env_idx]))
            self.obj.set_linear_velocity(torch.zeros((b, 3)))
            self.obj.set_angular_velocity(torch.zeros((b, 3)))
            self.pusher_contact_force[env_idx] = 0

    # ------------------------------------------------------- contact sensing
    def _reset_contact_accumulator(self):
        self._contact_force_sum = torch.zeros((self.num_envs, 3), device=self.device)
        self._contact_force_count = 0

    def _after_simulation_step(self):
        if not self.track_contact_force:
            return
        for link in self._pusher_links:
            # force exerted on the block by this link
            self._contact_force_sum += self.scene.get_pairwise_contact_forces(self.obj, link).to(self.device)
        self._contact_force_count += 1

    def _after_control_step(self):
        if not self.track_contact_force:
            return
        self.pusher_contact_force = self._contact_force_sum / max(self._contact_force_count, 1)
        self._reset_contact_accumulator()

    # ------------------------------------------------------------- task logic
    def evaluate(self):
        # PushCubeEnv's success, plus the physics (and contact force)
        info = super().evaluate()
        info["physics"] = self.block_physics
        if self.track_contact_force:
            info["pusher_contact_force"] = self.pusher_contact_force
        return info

    # observations and the dense reward are PushCubeEnv's (see reward.py for the planner copy)
