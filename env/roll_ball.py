"""RollBallPhys-v1: ManiSkill's RollBall-v1 ball, in the PushBlock workspace, with configurable physics.

The object is the same ball as RollBall-v1 (radius 3.5 cm, blue); the table, robot, camera, goal
and reset distribution are PushBlock-v1's, so both tasks share the push primitive, the point
cloud pipeline, the planner and the evaluation scripts, and only the object changes.

What makes a ball different from a block for a learned dynamics model:
  - it keeps rolling after the gripper lets go: the push primitive waits until it stops
    (PushPrimitiveConfig.settle_speed), so a transition is still "object at rest -> at rest";
  - how far it rolls is set by its rolling resistance (linear/angular damping) and mass, not
    only by friction; these are the sampled physics (see env.tasks.TASKS["ball"]);
  - its rotation is unobservable (and large), so the model learns the translation only
    (TaskSpec.rotation = False).
"""

from __future__ import annotations

import numpy as np
import sapien

from mani_skill.utils.registration import register_env

from .push_block import PushBlockEnv


@register_env("RollBallPhys-v1", max_episode_steps=50)
class RollBallEnv(PushBlockEnv):
    """
    **Task Description:**
    Push/roll a ball into a goal region in front of it.

    **Randomizations:**
    - ball xy position uniformly in [-0.1, 0.1]^2 on the table
    - goal is at ball xy + [0.1 + goal_radius, 0]
    - ball physics (mass, friction, restitution, linear/angular damping) as specified by `physics`

    **Success Conditions:**
    - ball xy is within goal_radius of the goal xy and the ball is still on the table
    """

    OBJECT_NAME = "ball"
    obj_half_size = 0.035  # m, RollBall-v1's ball_radius

    def _add_object_shape(self, builder, r, material, mass):
        builder.add_sphere_collision(radius=r, material=material)
        builder.add_sphere_visual(
            radius=r, material=sapien.render.RenderMaterial(base_color=np.array([0.0, 0.2, 0.8, 1.0]))
        )
        # solid sphere: I = 2/5 m r^2
        builder.set_mass_and_inertia(mass, sapien.Pose(), np.full(3, 0.4 * mass * r * r))
