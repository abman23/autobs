import itertools
import json
import os
from typing import Any, SupportsFloat

# import gym
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, Dict, Box
from gymnasium.utils import seeding

RANDOM_SEED: int | None = None  # manually set random seed


class BaseEnvironment(gym.Env):
    """MDP environment of single-BS, version 1.11.
    Old dataset, capacity reward.

    """

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.
        """
        self.version = "v1122"
        
        map_size = 256 
        self.action_space_size = 32
        self.action_space: Discrete = Discrete(self.action_space_size ** 2)
        
        
        self.observation_space = gym.spaces.Dict(
            {
                "observations": Box(low=0., high=1., shape=(map_size ** 2,), dtype=np.float32),
                "action_mask": Box(low=0., high=1., shape=(self.action_space.n,), dtype=np.int8)
            }
        )


    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        return self.observation_space.sample(), {}


    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        return self.observation_space.sample(), 0, False, {}