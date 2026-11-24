"""Observations that carry both the point cloud and the camera image.

ManiSkill's "pointcloud" obs mode converts the camera images to a point cloud and drops them.
We ask for the raw textures instead (OBS_MODE) and do the conversion ourselves with ManiSkill's
own function, keeping the RGB image for the image-feature models (dynamics_model.model
IMAGE_ENCODERS, scripts/dino_features.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mani_skill.envs.utils.observations import sensor_data_to_pointcloud

OBS_MODE = "rgb+position+segmentation"  # what the point cloud is built from, plus rgb
IMAGE_SIZE = 224  # stored image resolution (DINOv2 ViT-S/14 input: 16 x 16 patches)


def split_obs(obs, env):
    """obs from an env created with obs_mode=OBS_MODE -> (obs with obs["pointcloud"] as in the
    "pointcloud" obs mode, RGB images (B, IMAGE_SIZE, IMAGE_SIZE, 3) uint8 on the env's device).
    Also accepts an obs that already has a point cloud (then the image is None)."""
    if "pointcloud" in obs:
        return obs, None
    rgb = obs["sensor_data"]["base_camera"]["rgb"][..., :3]
    if rgb.shape[1] != IMAGE_SIZE:
        x = rgb.permute(0, 3, 1, 2).float()
        rgb = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="area").round().clamp(0, 255).to(torch.uint8)
        rgb = rgb.permute(0, 2, 3, 1).contiguous()
    o = {**obs, "sensor_data": dict(obs["sensor_data"])}  # the conversion deletes the images
    o = sensor_data_to_pointcloud(o, env.unwrapped._sensors)
    return o, rgb
