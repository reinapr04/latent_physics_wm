"""PushBlock-v1: a ManiSkill push task with configurable block physics.

A Panda arm must push a block into a goal region in front of it. Unlike the stock
PushCube-v1, the block's physical properties (mass, friction, restitution, damping) and
the table friction are configurable, either as fixed values or as (low, high) ranges that
are sampled independently for each parallel env. The block's size is fixed: PushCube's
default 4 cm cube (cube_half_size = 0.02 m).

Important: PhysX fixes these properties when the scene is built, so they are applied
during reconfiguration, not on every reset. New values are sampled from ranges only
when you call ``env.reset(options={"reconfigure": True})`` (or set ``reconfiguration_freq``).

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

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .reward import block_reward
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

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
class PushBlockEnv(BaseEnv):
    """
    **Task Description:**
    Push a block into a goal region in front of it.

    **Randomizations:**
    - block xy position uniformly in [-0.1, 0.1]^2 on the table, placed flat
    - goal is at block xy + [0.1 + goal_radius, 0]
    - block physics as specified by `physics` (fixed values or per-env ranges)

    **Success Conditions:**
    - block xy is within goal_radius of the goal xy and the block is still on the table
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    goal_radius = 0.1
    cube_half_size = 0.02  # m, same block size as PushCube-v1
    # Generic names used by the push primitive, the planner and env.roll_ball (a subclass that
    # only swaps the object): the object's half size (= its center height when resting) and name.
    obj_half_size = cube_half_size
    OBJECT_NAME = "block"

    def __init__(
        self,
        *args,
        physics: Union[BlockPhysicsConfig, dict, None] = None,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        track_contact_force: bool = False,
        **kwargs,
    ):
        if physics is None:
            physics = BlockPhysicsConfig()
        elif isinstance(physics, dict):
            physics = BlockPhysicsConfig.from_dict(physics)
        self.physics = physics
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # If True, the contact force the gripper (fingers + hand) applies to the block is
        # averaged over every physics sub-step of a control step and exposed as
        # self.pusher_contact_force / info["pusher_contact_force"] (num_envs, 3), in N, world frame.
        self.track_contact_force = track_contact_force
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # Fixed third-person camera on the far side of the table from the robot, framing the
        # workspace around the block (~0.8 m across), so a 4 cm block is ~12 px wide. Override
        # with gym.make(..., sensor_configs=dict(width=..., height=..., pose=...)).
        pose = sapien_utils.look_at(eye=[0.6, 0.25, 0.6], target=[0.05, 0.0, 0.0])
        return [CameraConfig("base_camera", pose=pose, width=256, height=256, fov=1.0, near=0.01, far=10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

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
        self._init_material_pool()

        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._apply_table_friction()

        # Build one block per parallel env so each can have its own physics, then merge
        # them into a single Actor view for batched pose/velocity access.
        names = BlockPhysicsConfig.block_fields()
        per_env = [self.physics.sample(self._batched_episode_rng[i]) for i in range(self.num_envs)]
        if getattr(self, "_physics_override", None) is not None:
            # explicit per-env values (set_physics); unspecified fields keep the sampled value
            per_env = [{**sampled, **given} for sampled, given in zip(per_env, self._physics_override)]
        blocks = []
        h = self.obj_half_size
        for i, p in enumerate(per_env):
            material = self._block_materials[i]
            material.static_friction = p["static_friction"]
            material.dynamic_friction = p["dynamic_friction"]
            material.restitution = p["restitution"]
            builder = self.scene.create_actor_builder()
            self._add_object_shape(builder, h, material, p["mass"])
            builder.set_scene_idxs([i])
            builder.set_initial_pose(sapien.Pose(p=[0, 0, h]))
            block = builder.build(name=f"{self.OBJECT_NAME}-{i}")
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

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

    def _add_object_shape(self, builder, h, material, mass):
        """Collision + visual shape and inertia of the pushed object (a cube of half edge h).
        Subclasses override this to push something else (env.roll_ball: a sphere of radius h)."""
        builder.add_box_collision(half_size=[h] * 3, material=material)
        builder.add_box_visual(
            half_size=[h] * 3,
            material=sapien.render.RenderMaterial(base_color=np.array([12, 42, 160, 255]) / 255),
        )
        builder.set_mass_and_inertia(mass, sapien.Pose(), _box_inertia(mass, h))

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
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # options["block_xy"] / options["goal_xy"] ((num_envs, 2)) place the block / goal
            # explicitly (e.g. the same start for several evaluation runs); default: random
            # block, goal 0.2 m in front of it
            xyz = torch.zeros((b, 3))
            if options.get("block_xy") is not None:
                xyz[..., :2] = torch.as_tensor(options["block_xy"], dtype=torch.float32).to(self.device)[env_idx]
            else:
                xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.block_half_size[env_idx]
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
            self.obj.set_linear_velocity(torch.zeros((b, 3)))
            self.obj.set_angular_velocity(torch.zeros((b, 3)))
            self.pusher_contact_force[env_idx] = 0

            goal_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
            if options.get("goal_xy") is not None:
                goal_xyz[..., :2] = torch.as_tensor(options["goal_xy"], dtype=torch.float32).to(self.device)[env_idx]
            goal_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(Pose.create_from_pq(p=goal_xyz, q=euler2quat(0, np.pi / 2, 0)))

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
    def _block_reward(self):
        return block_reward(self.obj.pose.p, self.goal_region.pose.p[..., :2], self.block_half_size, self.goal_radius)

    def evaluate(self):
        info = {
            "success": self._block_reward()[1],
            "physics": self.block_physics,
        }
        if self.track_contact_force:
            info["pusher_contact_force"] = self.pusher_contact_force
        return info

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(goal_pos=self.goal_region.pose.p, obj_pose=self.obj.pose.raw_pose)
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        h = self.block_half_size
        push_offset = torch.zeros_like(self.obj.pose.p)
        push_offset[:, 0] = -h - 0.005
        tcp_to_push_dist = torch.linalg.norm(self.obj.pose.p + push_offset - self.agent.tcp.pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_push_dist)

        reached = tcp_to_push_dist < 0.01
        reward += self._block_reward()[0] * reached

        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
