import torch
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.examples.rl_module.action_masking_rlm import TorchActionMaskRLM

from multi_agent.async_agent_original import Agent
from rl_module.action_mask_rlm import PPOActionMaskRLM
from env.utils_v1 import dict_update
from env.callbacks import AsyncActionCallbacks


class AsyncPPO(Agent):
    def __init__(self, config: dict, log_file: str, version: str) -> None:
        """
        Initialize the AsyncPPO agent.

        Args:
            config (dict): Configuration dictionary containing settings for environment, rollouts, training, etc.
            log_file (str): Path to the log file.
            version (str): Version identifier for the agent.
        """
        super().__init__(config, log_file, version)

        # Set the algorithm name to 'ppo'
        self.algo_name = 'ppo'
        
        # Update environment configuration with algorithm name
        env_config = dict_update(config.get("env"), {"algo_name": self.algo_name})

        # Create a PPO configuration using the provided config dictionary
        ppo_config = (
            PPOConfig()
            .environment(env=self.env_class, env_config=env_config)
            .framework("torch")  # Specify the deep learning framework as PyTorch
            .rollouts(
                num_rollout_workers=config["rollout"].get("num_rollout_workers", 1),
                num_envs_per_worker=config["rollout"].get("num_envs_per_worker", 1),
                rollout_fragment_length="auto",
                batch_mode=config["rollout"].get("batch_mode", "truncate_episodes"),
                remote_worker_envs=False,
            )
            .resources(
                num_gpus=torch.cuda.device_count(),  # Dynamically set number of GPUs available
            )
            .exploration(
                explore=True,
                exploration_config=config["explore"].get("exploration_config", {})
            )
            .training(
                train_batch_size=config["train"].get("train_batch_size", 4000),
                lr=config["train"].get("lr", 5e-5),
                gamma=config["train"].get("gamma", 0.9),
                grad_clip=config["train"].get("grad_clip", None),
                sgd_minibatch_size=config["train"].get("sgd_minibatch_size", 128),
                num_sgd_iter=config["train"].get("num_sgd_iter", 30),
                model=config["train"].get("model", {})
            )
            .evaluation(
                evaluation_interval=config["eval"].get("evaluation_interval", 1),
                evaluation_duration=config["eval"].get("num_maps_per_eval", 3),
                evaluation_config=config["eval"].get("evaluation_config", {}),
                evaluation_num_workers=config["eval"].get("evaluation_num_workers", 3),
                # enable_async_evaluation=True,
            )
            .reporting(
                min_sample_timesteps_per_iteration=config["report"].get("min_sample_timesteps_per_iteration", 1000)
            )
            .callbacks(AsyncActionCallbacks)  # Define the callback functions for async actions
            .experimental(
                _enable_new_api_stack=True,  # Enable new API stack for RLlib
                _disable_preprocessor_api=True,  # Disable observation preprocessing
            )
        )
        
        # Check if action masking is enabled in the environment configuration
        if not config["env"].get("no_masking", True):
            # Set RL module specification for action masking if masking is enabled
            ppo_config = ppo_config.rl_module(
                rl_module_spec=SingleAgentRLModuleSpec(module_class=PPOActionMaskRLM),
            )
        
        # Store the finalized agent configuration
        self.agent_config = ppo_config
        
        # Build the agent with the provided configuration
        self.agent = self.agent_config.build()
