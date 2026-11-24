"""Task specifications: everything task-dependent that the data collection, model, planner and
evaluation need to agree on, in one place.

    block   PushBlock-v1     4 cm cube (PushCube's block), rigid 6-DoF motion
    ball    RollBallPhys-v1  ManiSkill RollBall-v1's 3.5 cm-radius ball, translation only

Select with --task in train_dynamics.py; the eval scripts read it from the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_id: str
    half_size: float  # m: cube half edge / ball radius (object center height when resting)
    shape: str  # "box" | "sphere": how the samplers aim pushes at the observed object
    # The model predicts a rigid motion (rotation + translation). A sphere's rotation is
    # invisible in a point cloud and huge when it rolls, so for the ball only the translation is
    # learned (the true rotation is replaced by the identity when the data is stored).
    rotation: bool
    # push ranges for the random policy and the MPC proposals
    force: tuple  # N
    duration: tuple  # s
    push_height: tuple  # m, p_z of random/MPC pushes
    probe_height: tuple  # m, p_z of the probing pushes used to fit a new env's latent
    # physics pool ranges (train_dynamics.make_physics_pool): mass is log-uniform, the rest uniform
    mass: tuple
    static_friction: tuple
    dynamic_ratio: tuple  # mu_d = mu_s * U(ratio)
    linear_damping: tuple = (0.0, 0.0)
    angular_damping: tuple = (0.0, 0.0)
    # after a push, keep simulating (arm at home) until the object is slower than settle_speed
    # (m/s) or settle_timeout (s) has passed, so the next observation shows it at rest
    settle_speed: float = 0.0  # 0 = no wait (the block stops by itself while the arm goes home)
    settle_timeout: float = 0.0
    goal_dist: tuple = (0.10, 0.25)  # m, start-to-goal distance in eval_mpc.py
    # the gripper can't hit the ball squarely at its equator from below without the hand
    # clipping the top, nor push a cube above its top face; these are the physical limits
    physics_fields: tuple = field(default=("mass", "static_friction", "dynamic_friction", "linear_damping", "angular_damping"))


TASKS = {
    "block": TaskSpec(
        name="block",
        env_id="PushBlock-v1",
        half_size=0.02,
        shape="box",
        rotation=True,
        force=(0.2, 5.0),
        duration=(0.3, 0.8),
        push_height=(0.01, 0.05),  # the block is 4 cm tall: the top of the range misses it
        probe_height=(0.015, 0.035),
        mass=(0.05, 1.0),
        static_friction=(0.2, 1.0),
        dynamic_ratio=(0.5, 1.0),
    ),
    "ball": TaskSpec(
        name="ball",
        env_id="RollBallPhys-v1",
        half_size=0.035,
        shape="sphere",
        rotation=False,
        # a ball runs away from the gripper, so the measured push force is ~1/3 of the requested
        # one; above 4 N most pushes roll it > 30 cm (scripts/push_stats.py)
        force=(0.5, 4.0),
        duration=(0.3, 0.8),
        push_height=(0.02, 0.055),  # the ball is 7 cm tall
        probe_height=(0.025, 0.045),
        mass=(0.05, 1.0),
        static_friction=(0.2, 1.0),
        dynamic_ratio=(0.5, 1.0),
        # rolling resistance: how far the ball rolls after it leaves the gripper. With these
        # ranges a random push rolls it 7 cm (median) to 40 cm (90th percentile), see README
        linear_damping=(0.1, 1.0),
        angular_damping=(2.0, 8.0),
        settle_speed=0.01,
        settle_timeout=3.0,
        goal_dist=(0.10, 0.30),
    ),
}


def get_task(name) -> TaskSpec:
    """Task name -> TaskSpec (a TaskSpec passes through)."""
    if isinstance(name, TaskSpec):
        return name
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; choose from {list(TASKS)}")
    return TASKS[name]
