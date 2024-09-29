import json
import logging
import os.path
import time
from logging import Logger
import itertools
from tqdm import tqdm
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt
from ray import train, tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.tune.logger import pretty_print
import cv2


import torch

from env.utils_v2 import draw_map_with_tx, plot_coverage
from env.utils_v1 import ROOT_DIR, dict_update, load_map_normalized

from dataset_builder.pmnet_v3 import PMNet


class Agent(object):
    """Base class for a Deep Reinforcement Learning (DRL) agent under the asynchronous MDP setting.

    Attributes:
        agent_config (AlgorithmConfig): Configuration specific to the algorithm used by the agent.
        agent (Algorithm): The DRL algorithm instance used for training.
        algo_name (str): The name of the algorithm.
    """

    agent_config: AlgorithmConfig  # Initialized by a specific agent
    agent: Algorithm
    algo_name: str

    def __init__(self, config: dict, log_file: str, version: str = "v21") -> None:
        """
        Initializes the Agent class with configuration, logging, and environment setup.

        Args:
            config (dict): A dictionary containing configuration parameters.
            log_file (str): The path to the log file for saving logs.
            version (str, optional): The version of the environment to use. Defaults to "v21".
        """
        self.config: dict = config
        
        # Set up a logger for the agent
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Set up the environment class based on the specified version
        if version == "v21":
            from env.env_v21 import BaseEnvironment
        elif version == 'v30':
            from env.env_v30 import BaseEnvironment
        elif version == 'v19':
            from env.env_v19 import BaseEnvironment
        elif version == 'v192':
            from env.env_v192 import BaseEnvironment
        elif version == 'v110':
            from env.env_v110 import BaseEnvironment
        elif version == 'v111':
            from env.env_v111 import BaseEnvironment
        elif version == 'v112':
            from env.env_v1122 import BaseEnvironment
        elif version == 'v1122':
            from env.env_v1122 import BaseEnvironment
        elif version == 'v32':
            from env.env_v32 import BaseEnvironment
        elif version == 'v33':
            from env.env_v33 import BaseEnvironment
        elif version == 'v34':
            from env.env_v34 import BaseEnvironment
        elif version == 'v35':
            from env.env_v35 import BaseEnvironment
        else:
            from env.env_v31 import BaseEnvironment
        
        self.env_class = BaseEnvironment
        self.version: str = version

    def train_and_eval(self, log: bool = True, **kwargs):
        """Train and evaluate the agent, plotting the average training/evaluation reward per environment step vs. training step.

        Args:
            log (bool, optional): Whether to log the training and evaluation process. Defaults to True.
            **kwargs: Additional keyword arguments, expected to include 'timestamp'.
        """
        num_episode = self.config["stop"].get("training_iteration", 10)
        eval_interval = self.config["eval"].get("evaluation_interval", 5)
        data_saving_interval = self.config["agent"].get("data_saving_interval", 10)

        # Initialize evaluation data
        if eval_interval is None:
            ep_eval = np.arange(0)
            ep_reward_mean = np.empty(0, dtype=float)
            ep_reward_std = np.empty(0, dtype=float)
        else:
            ep_eval = np.arange(0, num_episode, eval_interval) + eval_interval
            ep_reward_mean = np.empty(num_episode // eval_interval, dtype=float)
            ep_reward_std = np.empty(num_episode // eval_interval, dtype=float)

        # Initialize training data
        ep_train = np.arange(num_episode) + 1
        ep_reward_mean_train = np.empty(num_episode, dtype=float)
        ep_reward_std_train = np.empty(num_episode, dtype=float)

        timestamp = kwargs["timestamp"]
        start_info = f"==========={self.algo_name.upper()} train and eval started at {timestamp} for {self.version}==========="
        
        # Log and print start info
        if log:
            self.logger.info(start_info)
        print(start_info)

        max_val_reward: float = -np.inf
        for i in range(num_episode):
            # Perform one training step (may include multiple environment episodes)
            result = self.agent.train()

            print("\n")
            print(f"================TRAINING # {i + 1}================")
            print(f"timesteps_total: {result['timesteps_total']}")
            print(f"time_total_s: {result['time_total_s']}")
            
            if eval_interval is not None and (i + 1) % eval_interval == 0:
                print(f"================EVALUATION AT # {i + 1}================")

            print(f'Max reward per episode: {result["episode_reward_max"]}\nMin reward per episode: {result["episode_reward_min"]}\nMean reward per episode: {result["episode_reward_mean"]}')
            
            if i == num_episode - 1:
                if log:
                    # Save the final result and checkpoint
                    self.logger.debug(self.config)
                    self.logger.info("=============TRAINING ENDED=============")
                else:
                    print("=============TRAINING ENDED=============")
                    print(self.config)
            
            # Update training rewards
            reward_per_round_mean = result["custom_metrics"]["reward_per_round_mean"]
            reward_per_round_std = result["custom_metrics"]["reward_per_round_std"]
            ep_reward_mean_train[i] = reward_per_round_mean
            ep_reward_std_train[i] = reward_per_round_std

            if eval_interval is not None and (i + 1) % eval_interval == 0:
                # Calculate and log evaluation mean reward per step
                ep_r_mean = result['evaluation']["custom_metrics"]["reward_per_round_mean"]
                ep_r_std = result['evaluation']["custom_metrics"]["reward_per_round_std"]
                idx = (i + 1) // eval_interval - 1
                ep_reward_mean[idx] = ep_r_mean
                ep_reward_std[idx] = ep_r_std
                
                # Save the agent with the greatest validation reward
                if log and ep_r_mean > max_val_reward:
                    max_val_reward = ep_r_mean
                    self.agent.save(f"./checkpoint/{self.version}_{self.algo_name}_{timestamp}")
            
            if log and ((i + 1) % data_saving_interval == 0 or i == num_episode - 1):
                # Save the training and evaluation data periodically
                data = {
                    "ep_train": ep_train.tolist(),
                    "ep_reward_mean_train": ep_reward_mean_train.tolist(),
                    "ep_reward_std_train": ep_reward_std_train.tolist(),
                    "ep_eval": ep_eval.tolist(),
                    "ep_reward_std": ep_reward_std.tolist(),
                    "ep_reward_mean": ep_reward_mean.tolist(),
                }
                json.dump(data,
                          open(os.path.join(ROOT_DIR, f"data/{self.version}_{self.algo_name}_{timestamp}.json"), 'w'))

        # Plot mean reward over episodes
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.plot(ep_train, ep_reward_mean_train, color='red', label='train')

        if eval_interval is not None:
            # Plot the mean reward in evaluation
            ax.plot(ep_eval, ep_reward_mean, color="blue", label='eval')
        
        ax.set(xlabel="episode", ylabel="average reward",
               title=f"{self.algo_name.upper()} Training Results")
        ax.grid()
        ax.legend()

        if log:
            fig.savefig(os.path.join(ROOT_DIR, f"figures/{self.version}_{self.algo_name}_{timestamp}.png"))
        
        plt.show()

    def test(self, timestamp: str, duration: int = 25, log: bool = True, suffix: str = 'after',
             test_on_trained: bool = False):
        """
        Test the trained agent on both the training maps and test maps.

        Args:
            timestamp (str): The timestamp when the model was trained, used to locate the model.
            duration (int, optional): Number of test iterations to run for each map. Defaults to 25.
            log (bool, optional): If True, logs the test information. Defaults to True.
            suffix (str, optional): Suffix used in logging and file naming. Defaults to 'after'.
            test_on_trained (bool, optional): If True, test on both training and test maps. 
                                              Otherwise, test only on test maps. Defaults to False.
        """
        # Log the start of the test process
        msg = f"\n=============Test for {self.algo_name.upper()} {suffix.upper()}============="
        print(msg)
        if log:
            self.logger.info(msg)

        # Load environment configurations
        env_config_train: dict = json.loads(json.dumps(self.config.get("env")))
        env_config_test = dict_update(env_config_train, self.config['eval']['evaluation_config']['env_config'])
        if test_on_trained:
            env_configs = {'training': env_config_train, 'test': env_config_test}
        else:
            env_configs = {'test': env_config_test}

        # Reload the best model based on the provided timestamp
        model_path = os.path.join(ROOT_DIR, f'checkpoint/{self.version}_{self.algo_name}_{timestamp}')
        self.agent.restore(model_path)

        start_time = time.time()
        for env_type, env_config in env_configs.items():
            # Initialize the environment for evaluation
            env_eval = self.env_class(config=env_config)
            env_eval.evaluation = True  # Ensure that the map is selected in sequence at each reset
            
            # Initialize cumulative statistics for reporting
            coverage_test_mean = 0.0
            capacity_test_mean = 0.0
            reward_mean_overall = 0.0
            reward_opt_mean = 0.0
            num_roi_mean = 0

            for i in range(duration):
                before_reset = time.time()
                obs, info_dict = env_eval.reset()
                after_reset = time.time()
                
                # Calculate the number of RoI pixels
                num_roi = np.sum(env_eval.pixel_map == env_eval.non_building_pixel)
                num_roi_mean += num_roi / duration

                # Calculate optimal locations and rewards
                locs_opt, reward_opt = env_eval.calc_optimal_locations()
                after_calc_opt = time.time()
                locs = []

                # Take actions and compute rewards
                before_action = time.time()
                for _ in range(env_eval.n_bs):
                    action = self.agent.compute_single_action(obs)
                    row, col = env_eval.calc_upsampling_loc(action)
                    locs.append((row, col))
                    obs, _, _, _, info_dict = env_eval.step(action)
                after_action = time.time()
                
                # Calculate coverage and capacity rewards
                _, coverage_test = env_eval.calc_coverage_reward(locs)
                _, capacity_test = env_eval.calc_capacity_reward(locs)
                accumulated_reward = info_dict['accumulated_reward']
                
                # Accumulate mean rewards for statistics
                coverage_test_mean += coverage_test / duration
                capacity_test_mean += capacity_test / duration
                reward_mean_overall += accumulated_reward / duration
                reward_opt_mean += reward_opt / duration

                # Log and print information at intervals
                info = (
                    f"Stats for {env_type} map {i} with index {env_eval.map_idx}\n"
                    f"percentage of coverage: {coverage_test * 100}\n"
                    f"capacity: {capacity_test}\n"
                    f"reward: {accumulated_reward}\n"
                    f"optimal reward: {reward_opt}\n"
                    f"ratio: {accumulated_reward / reward_opt}, num_roi: {num_roi}\n\n"
                )
                time_info = (
                    f"one map total {time.time() - before_reset:.4f}s, env reset {after_reset - before_reset:.4f}s, "
                    f"calc opt action {after_calc_opt - after_reset:.4f}s, inference {after_action - before_action:.4f}s")

                if i % 10 == 0:
                    print(f"{time.time() - start_time:.4f}s so far")
                    print(time_info)
                    print(info)
                    if log:
                        self.logger.info(time_info)
                        self.logger.info(info)

                # Save coverage/capacity maps if needed
                if log and (i == 0 or i == duration - 1):
                    # (overlap of power maps corresponding to multiple TX locations)
                    if env_eval.reward_type == 'coverage':
                        # Plot coverage area 
                        coverage_map, _ = env_eval.calc_coverage(locs)
                        coverage_map_opt, _ = env_eval.calc_coverage(locs_opt)
                        overall_rewards = env_eval.calc_rewards_for_all_locations()
                        coverage_map_dir = os.path.join(ROOT_DIR, f'figures/coverage_map/{self.version}_{timestamp}_{self.algo_name}')
                        os.makedirs(coverage_map_dir, exist_ok=True)
                        coverage_map_path = os.path.join(coverage_map_dir,
                                                         f'{env_eval.map_idx}_{suffix}.png')
                        plot_coverage(filepath=coverage_map_path, pixel_map=env_eval.pixel_map,
                                      coverage_curr=coverage_map,
                                      coverage_opt=coverage_map_opt, tx_locs=locs, opt_tx_locs=locs_opt, rewards=overall_rewards, save=True)
                    elif env_eval.reward_type == 'capacity':
                        # Plot capacity map
                        capacity_map, _ = env_eval.calc_capacity(locs)
                        capacity_map_opt, _ = env_eval.calc_capacity(locs_opt)
                        overall_rewards = env_eval.calc_rewards_for_all_locations()
                        capacity_map_dir = os.path.join(ROOT_DIR, f'figures/capacity_map/{self.version}_{timestamp}_{self.algo_name}')
                        os.makedirs(capacity_map_dir, exist_ok=True)
                        capacity_map_path = os.path.join(capacity_map_dir,
                                                         f'{env_eval.map_idx}.png')
                        plot_coverage(filepath=capacity_map_path, pixel_map=env_eval.pixel_map,
                                      coverage_curr=capacity_map,
                                      coverage_opt=capacity_map_opt, tx_locs=locs, opt_tx_locs=locs_opt, rewards=overall_rewards, save=True)

            # Log summary statistics for the test
            info1 = (
                    f"Stats for {env_type} maps\n"
                    f"average percentage of coverage: {coverage_test_mean * 100}\n"
                    f"average capacity: {capacity_test_mean}\n"
                    f"average reward: {reward_mean_overall}\n"
                    f"optimal reward: {reward_opt_mean}\n"
                    f"ratio: {reward_mean_overall / reward_opt_mean}, num_roi: {num_roi}\n\n"
            )

            if log:
                self.logger.info(info1)
            print(info1)