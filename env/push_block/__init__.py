"""PushBlock-v1: ManiSkill's PushCube-v1 extended with configurable block physics."""

from .env import BlockPhysicsConfig, PushBlockEnv  # importing registers "PushBlock-v1"
from .reward import block_reward

__all__ = ["BlockPhysicsConfig", "PushBlockEnv", "block_reward"]
