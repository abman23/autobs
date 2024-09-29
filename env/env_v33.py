import itertools
import json
import os
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Discrete, Dict, Box
from gymnasium.utils import seeding

from env.utils_v1 import ROOT_DIR, load_map_normalized

RANDOM_SEED: int | None = None  # manually set random seed


class BaseEnvironment(gym.Env):
    """MDP environment of multi-BS, version 3.3.
    Old dataset, capacity reward.

    """
    pixel_map: np.ndarray  # current building map
    version: str = "v33"
    steps: int
    map_idx: int  # index of the current building map
    accumulated_reward: int  # accumulated reward in one episode
    tx_locs: list  # Deployed TX locations
    r_prev: float  # reward before deploying the current TX

    def __init__(self, config: dict) -> None:
        """Initialize the base MDP environment.

        """
        self.version = "v33"
        # directory contains building maps, training and test power maps
        self.dataset_dir = config.get("dataset_dir", "resource/new_usc")
        evaluation_mode = config.get("evaluation", False)
        self.evaluation = evaluation_mode
        # training or test env
        self.map_suffix = "test" if evaluation_mode else "train"
        # Generate indices from 0 to 1209
        all_indices = np.arange(0, 1210)
        
        # Generate evaluation indices by taking every 110th index
        eval_indices = all_indices[::11]
        
        if not evaluation_mode:
            # Select indices not in eval_indices
            self.map_indices = np.setdiff1d(all_indices, eval_indices)
        else:
            # Select evaluation indices
            self.map_indices = eval_indices
        self.coverage_thresholds = np.array([0.6078, 0.627, 0.647])
        self.coverage_masks = np.array([0.5, 0.75, 1.])
        self.coverage_threshold: float = 0.647
        self.non_building_pixel: float = config["non_building_pixel"]
        self.reward_type: str = config.get('reward_type', 'coverage')

        self.n_bs: int = config.get("n_bs", 2)
        self.n_maps: int = config.get("n_maps", 1)
        self.n_steps_truncate: int = config.get("n_steps_truncate", 10)
        # count the number of used cropped maps
        self.n_trained_maps: int = 0

        map_size = config.get("map_size", 256)
        # number of pixels in one row or column of the cropped map
        self.map_size: int = map_size
        action_space_size = config.get("action_space_size", 64)
        assert map_size % action_space_size == 0, f"map_size {map_size} must be divisible by action_space_size {action_space_size}"
        # used for calculating the location corresponding to action in the reduced action space
        self.upsampling_factor = map_size // action_space_size
        self.action_space_size: int = action_space_size
        # action mask has the same shape as the action
        self.mask: np.ndarray = np.empty(action_space_size ** 2, dtype=np.int8)
        self.no_masking: bool = config.get("no_masking", False)

        self.action_space: Discrete = Discrete(action_space_size ** 2)
        if self.no_masking:
            self.observation_space: Box = Box(low=0., high=1., shape=(map_size ** 2 * 2,), dtype=np.float32)
        else:
            self.observation_space: Dict = Dict(
                {
                    "observations": Box(low=0., high=1., shape=(map_size ** 2 * 2,), dtype=np.float32),
                    "action_mask": Box(low=0., high=1., shape=(self.action_space.n,), dtype=np.int8)
                }
            )


        # fix a random seed
        self._np_random, seed = seeding.np_random(RANDOM_SEED)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to its initial state for a new episode.

        Args:
            seed: Optional seed for random number generation.
            options: Optional dictionary containing reset options.

        Returns:
            A tuple containing the initial observation and a dictionary of additional info.
        """
        super().reset(seed=seed, options=options)

        # Reset step counter and accumulated reward
        self.steps = 0
        self.accumulated_reward = 0

        # Choose the map index based on whether in evaluation or training mode
        # During evaluation, maps are chosen sequentially
        # During training, maps are chosen randomly, with indices shuffled after all maps have been trained
        if not self.evaluation and self.n_trained_maps % self.n_maps == 0:
            self.map_indices = np.random.permutation(self.map_indices)
        self.map_idx = int(self.map_indices[self.n_trained_maps % self.n_maps])
        self.n_trained_maps += 1

        # update map and action mask
        map_path = os.path.join(self.dataset_dir, 'map', str(self.map_idx) + '.png')
        self.pixel_map = load_map_normalized(map_path)
        # 1 - free space
        self.mask = self._calc_action_mask()

        # initial state: no TX (transmitter) deployed
        self.tx_locs = []
        self.r_prev = 0.

        # observe capacity map as building map
        obs = np.tile(self.pixel_map.reshape(-1), 2)

        # Determine whether to include action masking in the observation
        if self.no_masking:
            observation = obs
        else:
            observation = {
                "observations": obs,
                "action_mask": self.mask
            }

        # Additional information to be returned with the observation
        info_dict = {
            "n_trained_maps": self.n_trained_maps,
            "map_suffix": self.map_suffix,
            "map_index": self.map_idx,
            "accumulated_reward": self.accumulated_reward,
        }

        return observation, info_dict

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Performs a step in the environment based on the given action.

        Args:
            action: The action to be taken by the agent, typically indicating where to place a transmitter.

        Returns:
            A tuple containing:
                - observation: The new observation after the action.
                - reward: The reward obtained from taking the action.
                - done: A boolean indicating if the episode is finished (always False in this case).
                - trunc: A boolean indicating if the episode is truncated (e.g., step limit reached).
                - info_dict: A dictionary with additional information about the step.
        """
        # Calculate the upsampling location based on the action
        row, col = self.calc_upsampling_loc(action)

        # Append the new TX (transmitter) location and calculate the new reward
        self.tx_locs.append((row, col))
        if self.reward_type == 'coverage':
            output_map, r_new = self.calc_coverage(self.tx_locs)
        else:  # capacity
            output_map, r_new = self.calc_capacity(self.tx_locs)
        
        # Calculate the incremental reward and update accumulated reward
        r = r_new - self.r_prev
        self.accumulated_reward += r
        self.r_prev = r_new

        # Increment the step count and check for truncation condition
        self.steps += 1
        trunc = self.steps >= self.n_steps_truncate  # truncate if reach the step limit

        # reset TX locations and reward if all TXs are deployed
        if len(self.tx_locs) == self.n_bs:
            self.tx_locs = []
            self.r_prev = 0.
            output_map = self.pixel_map.copy()

        # Prepare additional information to return with the observation
        info_dict = {
            "steps": self.steps,
            "loc": [row, col],
            "reward": r,
            "n_bs": self.n_bs,
            "accumulated_reward": self.accumulated_reward,
        }

        # update observation by combining the pixel map and output map
        obs = np.concatenate((self.pixel_map, output_map), axis=None)

        if self.no_masking:
            observation = obs
        else:
            observation = {
                "observations": obs,
                "action_mask": self.mask
            }

        return observation, r, False, trunc, info_dict
        

    def calc_rewards_for_all_locations(self) -> dict:
        """Calculate the reward for all possible TX locations on the current map.

        This method computes rewards for all potential transmitter (TX) locations
        on the current map. If a precomputed reward file exists, it loads the rewards
        from that file; otherwise, it calculates the rewards, stores them in a dictionary,
        and saves the results to a file for future use.

        Returns:
            A dictionary where keys are location coordinates (as "row,col") and
            values are the corresponding rewards.
        """
        rewards = {}
        
        # Define the directory and filename for saving/loading precomputed rewards
        data_dir = os.path.join(self.dataset_dir, f'overall_{self.version}_{self.reward_type}_{self.n_bs}')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir) # Create the directory if it doesn't exist
        filename = os.path.join(data_dir, f"overall_{self.reward_type}_{self.n_bs}_{self.map_idx}.json")
        
        if not os.path.exists(filename): # Check if the reward file already exists
            all_actions = itertools.combinations_with_replacement(range(self.action_space_size ** 2), 1)
    
            for actions in all_actions:
                tx_locs = []
                flag = False

                for action in actions:
                    row, col = self.calc_upsampling_loc(action)

                    if self.pixel_map[row, col] == self.non_building_pixel:
                        # skip non-building pixel; no reward calculation for these
                        flag = True
                        break
                    tx_locs.append((row, col))

                if flag:
                    continue # Skip to the next action if a non-building pixel was encountered
                
                # Calculate the reward based on the specified reward type
                if self.reward_type == 'coverage':
                    _, reward = self.calc_coverage(tx_locs)
                else:  # capacity
                    _, reward = self.calc_capacity(tx_locs)
                
                rewards[f'{row},{col}'] = reward # Store the reward with location as key

            # Save the computed rewards to a file for future use
            json.dump(rewards, open(filename, 'w'))
        else:
            # Load precomputed rewards from the file 
            rewards = json.load(open(filename))
    
        return rewards


    def calc_optimal_locations(self) -> tuple[list, int]:
        """Calculate the optimal TX locations that maximize the reward.

        This function computes the optimal transmitter (TX) locations on the current map
        that yield the maximum possible reward. If a precomputed optimal reward file exists,
        it loads the optimal locations and reward from that file. Otherwise, it calculates
        them, stores the results, and returns the optimal locations and the associated reward.

        Returns:
            tuple: A tuple containing:
                - locs_opt (list): The list of optimal TX locations as (row, col) tuples.
                - reward_opt (int): The maximum reward achieved at the optimal locations.
        """
        locs_opt, reward_opt = [(-1, -1)], -1 # Initialize with default values

        # Define the directory and filename for saving/loading the optimal results
        data_dir = os.path.join(self.dataset_dir, f'optimal_{self.version}_{self.reward_type}_{self.n_bs}')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir) # Create the directory if it doesn't exist
        filename = os.path.join(data_dir, f"optimal_{self.reward_type}_{self.n_bs}_{self.map_idx}.json")
        
        if not os.path.exists(filename): # Check if the optimal results file already exists
        # Generate all possible combinations of actions (TX locations)
            all_actions = itertools.combinations_with_replacement(range(self.action_space_size ** 2), self.n_bs)

            for actions in all_actions:
                tx_locs = []
                flag = False
                for action in actions:
                    row, col = self.calc_upsampling_loc(action)
                    if self.pixel_map[row, col] == self.non_building_pixel:
                        # Skip non-building pixels; no reward calculation for these
                        flag = True
                        break
                    tx_locs.append((row, col))
                if flag:
                    continue # Skip to the next action if a non-building pixel was encountered
                
                # Calculate the reward based on the specified reward type
                if self.reward_type == 'coverage':
                    _, reward = self.calc_coverage(tx_locs)
                else:  # capacity
                    _, reward = self.calc_capacity(tx_locs)
                if reward > reward_opt:
                    reward_opt = reward
                    locs_opt = tx_locs

            # save result to avoid repeatedly computation
            result = {"locs_opt": locs_opt, "reward_opt": reward_opt}
            json.dump(result, open(filename, 'w'))
        else:
            # Load precomputed optimal locations and reward from the file if it exists
            result = json.load(open(filename))
            locs_opt, reward_opt = result["locs_opt"], result["reward_opt"]

        return locs_opt, reward_opt
    
    def calc_coverage(self, tx_locs: list) -> tuple[np.ndarray, int]:
        """Calculate the overall coverage reward of multiple TXs, given their locations.

        Args:
            tx_locs: tx location tuples

        Returns:
            (Coverage map, number of covered pixels in the map)

        """
        overall_coverage = np.zeros_like(self.pixel_map, dtype=int)
        _, avg_capacity = self.calc_capacity(tx_locs)
        for tx_loc in tx_locs:
            # Find index fpr tx location
            row, col = tx_loc[0], tx_loc[1]
            loc_idx = row * self.map_size + col

            # Retrieve image
            pmap_filename = f'pmap_{self.map_idx}_{loc_idx}.png'
            pmap_dir = os.path.join(self.dataset_dir, 'power_map', f'{self.map_idx}')
            pmap_path = os.path.join(pmap_dir, pmap_filename)
            power_map = load_map_normalized(pmap_path)
            # compute pixels covered by one TX from power map
            for i in range(len(self.coverage_thresholds)):
                covered = np.where(power_map >= self.coverage_thresholds[i], self.coverage_masks[i], 0)
                # update overall covered pixels
                overall_coverage = np.maximum(overall_coverage, covered)

        return overall_coverage, float(avg_capacity)

    def calc_capacity(self, tx_locs: list[tuple]) -> tuple[np.ndarray, float]:
        """Calculate the capacity reward of TXs (between 0 and 1), given their locations.

        Args:
            tx_locs: [(row_1, col_1), (row_2, col_2) ...].

        Returns:
            (Capacity map, average max grayscale value of each RoI pixel given different TX locations)

        """
        # only consider RoI while RoI are white pixels (value=1) in the power map
        num_roi = np.sum(self.pixel_map == self.non_building_pixel)

        capacity_map = np.zeros_like(self.pixel_map, dtype=float)
        for row, col in tx_locs:
            # Find index fpr tx location
            loc_idx = row * self.map_size + col

            # Retrieve image
            pmap_filename = f'pmap_{self.map_idx}_{loc_idx}.png'
            pmap_dir = os.path.join(self.dataset_dir, 'power_map', f'{self.map_idx}')
            pmap_path = os.path.join(pmap_dir, pmap_filename)
            power_map = load_map_normalized(pmap_path)
            capacity_map = np.maximum(capacity_map, power_map)

        avg_capacity = capacity_map.sum() / num_roi
        
        transformed_avg_capacity = np.exp(avg_capacity * 2) / np.exp(2)
        return capacity_map, float(transformed_avg_capacity)
        
    def calc_coverage_reward(self, tx_locs: list) -> tuple[np.ndarray, int]:
        """Calculate the overall coverage reward of multiple TXs, given their locations.

        Args:
            tx_locs: tx location tuples

        Returns:
            (Coverage map, number of covered pixels in the map)

        """
        num_roi = np.sum(self.pixel_map == self.non_building_pixel)
        overall_coverage = np.zeros_like(self.pixel_map, dtype=int)
        for tx_loc in tx_locs:
            # Find index for tx location
            row, col = tx_loc[0], tx_loc[1]
            loc_idx = row * self.map_size + col

            # Retrieve image
            pmap_filename = f'pmap_{self.map_idx}_{loc_idx}.png'
            pmap_dir = os.path.join(self.dataset_dir, 'power_map', f'{self.map_idx}')
            pmap_path = os.path.join(pmap_dir, pmap_filename)
            power_map = load_map_normalized(pmap_path)
            power_map[self.pixel_map != self.non_building_pixel] = 0
            
            # compute pixels covered by one TX from power map
            for i in range(len(self.coverage_thresholds)):
                covered = np.where(power_map >= self.coverage_thresholds[i], self.coverage_masks[i], 0)
                # update overall covered pixels
                overall_coverage = np.maximum(overall_coverage, covered)
            
        avg_coverage = int(overall_coverage[overall_coverage==1].sum()) / num_roi

        return overall_coverage, avg_coverage
    
    def calc_capacity_reward(self, tx_locs: list[tuple]) -> tuple[np.ndarray, float]:
        """Calculate the capacity reward of TXs (between 0 and 1), given their locations.

        Args:
            tx_locs: [(row_1, col_1), (row_2, col_2) ...].

        Returns:
            (Capacity map, average max grayscale value of each RoI pixel given different TX locations)

        """
        # only consider RoI while RoI are black pixels (value=0) in the power map
        num_roi = np.sum(self.pixel_map == self.non_building_pixel)
        capacity_map = np.zeros_like(self.pixel_map, dtype=float)
        for row, col in tx_locs:
            # Find index fpr tx location
            loc_idx = row * self.map_size + col

            # Retrieve image
            pmap_filename = f'pmap_{self.map_idx}_{loc_idx}.png'
            pmap_dir = os.path.join(self.dataset_dir, 'power_map', f'{self.map_idx}')
            pmap_path = os.path.join(pmap_dir, pmap_filename)
            power_map = load_map_normalized(pmap_path)
            power_map[self.pixel_map != self.non_building_pixel] = 0
            capacity_map = np.maximum(capacity_map, power_map)

        avg_capacity = capacity_map.sum() / num_roi
        
        return capacity_map, float(avg_capacity)

    def calc_upsampling_loc(self, action: int) -> tuple:
        """Calculate the location corresponding to a 'space-reduced' action by upsampling.

        Args:
            action: action in the reduced action space.

        Returns:
            Coordinate of the location - (row, col).

        """
        row_r, col_r = divmod(action, self.action_space_size)
        row = row_r * self.upsampling_factor + (self.upsampling_factor - 1) // 2
        col = col_r * self.upsampling_factor + (self.upsampling_factor - 1) // 2
        return row, col

    def _calc_action_mask(self) -> np.ndarray:
        """Calculate the action mask in the reduced action space.

        Returns:
            A 0-1 flatten array of the action mask.

        """
        idx = np.arange((self.upsampling_factor - 1) // 2, self.map_size, self.upsampling_factor)
        # filter out non-building pixel
        action_pixels = np.where(self.pixel_map[idx][:, idx] != self.non_building_pixel, 1, 0)
        return action_pixels.reshape(-1).astype(np.int8)
