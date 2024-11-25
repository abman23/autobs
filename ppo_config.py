import yaml
import os

import torch
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

from rl_module.action_mask_rlm import PPOActionMaskRLM
from env.utils_v1 import dict_update, ROOT_DIR


def get_ppo_config(BaseEnvironment):
    config= yaml.safe_load(open(os.path.join(ROOT_DIR, f'config.yaml'), 'r'))
    algo_name = 'ppo'
    env_config = dict_update(config.get("env"), {"algo_name": algo_name})
    
    ppo_config = (
        PPOConfig()
        .environment(env=BaseEnvironment, env_config=env_config, disable_env_checking=True)
        .framework("torch")
        .resources(
            num_gpus=torch.cuda.device_count(),
        )
        .exploration(
            explore=True,
            exploration_config=config["explore"].get("exploration_config", {})
        )
        .training(
            model=config["train"].get("model", {})
        )
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=True,  
        )
    )
    if not config["env"].get("no_masking", True):
        ppo_config = ppo_config.rl_module(
            rl_module_spec=SingleAgentRLModuleSpec(module_class=PPOActionMaskRLM),
        )
    agent_config = ppo_config
    agent = agent_config.build()
    return agent