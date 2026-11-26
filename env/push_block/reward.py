"""PushBlock-v1's block reward as a plain function of the block position, for planners.

The env's reward is PushCubeEnv.compute_dense_reward (inherited unchanged), which needs the
simulator's state. Planners score predicted block positions instead, with this function: the
block terms of that reward, written as a function of a position. Keep it in sync with
mani_skill/envs/tasks/tabletop/push_cube.py (place_reward, z_reward, success).
"""

import torch


def block_reward(block_pos, goal_xy, half_size, goal_radius):
    """block_pos (..., 3) block center, goal_xy (..., 2) -> (reward (...), success (...)).

    reward = place * (1 + upright), with
      place   = 1 - tanh(5 * xy distance block -> goal)
      upright = 1 - tanh(5 * |block z - half_size|)   (the block rests on the table)
    success = within goal_radius of the goal and still on the table.
    """
    dist = torch.linalg.norm(block_pos[..., :2] - goal_xy, dim=-1)
    place = 1 - torch.tanh(5 * dist)
    upright = 1 - torch.tanh(5 * (block_pos[..., 2] - half_size).abs())
    on_table = block_pos[..., 2] < half_size + 5e-3
    return place * (1 + upright), (dist < goal_radius) & on_table
