import sys
sys.path.insert(0,'/content/drive/MyDrive/Arjun/Sionna/Inference')

import os
os.environ['PYTHONPATH'] = "/content/drive/MyDrive/Arjun/Sionna/Inference"

import cv2
import numpy as np
from tqdm import tqdm
import torch
import argparse
parser = argparse.ArgumentParser(description="Process some files and numbers.")

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from ppo_config import get_ppo_config
from env.utils_v1 import find_top_left, calc_action_mask, calc_upsampling_loc, get_powermap, get_stats, calc_optimal_locations, calc_random_locations
from visualize.sionna_vis import Sionna_Visualizer 
from visualize.pmnet_v3 import PMNet

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', help='single or multi', default='single')
parser.add_argument('-c', '--crop_id', type=int, help='enter number between 0-15', default=3)
parser.add_argument('-r', '--reward_type', help="'coverage' or 'capacity' for exhaustive search", default='coverage')

args = parser.parse_args()
version = args.version
# Find the top and left values for a given crop_id
sionna_crop_id = args.crop_id
reward_type = args.reward_type

if version == "single":
    from env.env_v1122 import BaseEnvironment 
else:
    from env.env_v35 import BaseEnvironment 

pmnet = PMNet(
      n_blocks=[3, 3, 27, 3],
      atrous_rates=[6, 12, 18],
      multi_grids=[1, 2, 4],
      output_stride=8,)
pmnet.cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pmnet.load_state_dict(torch.load("pmnet_checkpoint/model_0.00008.pt"))
pmnet.to(device)

coverage_masks = np.array([0.5, 0.75, 1.])
coverage_thresholds = np.array([0.6078, 0.627, 0.647])

optimal_coverages = []
optimal_capacities = []
coverages = []
capacities = []

if __name__ == "__main__":
    agent = get_ppo_config(BaseEnvironment)
    
    agent.restore(f"checkpoints/{version}_ppo")

    city_map = cv2.imread(f"visualize//USC_city_map.png")[:, :, 0]
    sionna_vis = Sionna_Visualizer(city_map)
    
    for crop_id in tqdm(range(5)):
        top, left = find_top_left(crop_id)
        pixel_map = cv2.resize(city_map[top:top+512, left:left+512], (256, 256)) /255
    
        if version == 'single':
            mask = calc_action_mask(pixel_map)
            obs = pixel_map.reshape(-1)
            
            observation = {
                            "observations": obs,
                            "action_mask": mask
            }
            tx_locs = []
            
            action = agent.compute_single_action(observation)
            row, col = calc_upsampling_loc(action)
            tx_locs.append((row, col))
            tx_layer = np.zeros_like(pixel_map, dtype=int)
            tx_layer[row-3:row+3, col-3:col+3] = 1
        
        else: 
            mask = calc_action_mask(pixel_map)
            obs = np.tile(pixel_map.reshape(-1), 2)
            
            observation = {
                            "observations": obs,
                            "action_mask": mask
            }
            tx_locs = []
            action = agent.compute_single_action(observation)
            row, col = calc_upsampling_loc(action)
            tx_locs.append((row, col))
            tx_layer = np.zeros_like(pixel_map, dtype=int)
            tx_layer[row-3:row+3, col-3:col+3] = 1
            
            power_map = get_powermap(pixel_map, tx_layer)
            
            next_obs = np.zeros_like(power_map)
            for i in range(len(coverage_thresholds)):
                covered = np.where(power_map >= coverage_thresholds[i], coverage_masks[i], 0)
                next_obs = np.maximum(next_obs, covered)
            
            obs = np.concatenate((pixel_map, next_obs), axis=None)
            observation = {
                            "observations": obs,
                            "action_mask": mask
            }
            
            action = agent.compute_single_action(observation)
            row, col = calc_upsampling_loc(action)
            tx_locs.append((row, col))
            tx_layer[row-3:row+3, col-3:col+3] = 1
        
        pathgain, coverage, capacity = get_stats(pixel_map, tx_locs, pmnet=pmnet)
        cv2.imwrite(f"visualize/output/{version}_{crop_id}_ppo.png", 255 * (pathgain + tx_layer))
        coverages.append(coverage)
        capacities.append(capacities)

        optimal_locs, optimal_coverage, optimal_capacity = calc_optimal_locations(pixel_map, crop_id, version, reward_type=reward_type, pmnet=pmnet)
        optimal_coverages.append(optimal_coverage)
        optimal_capacities.append(optimal_capacity)
        
        random_locs, random_coverage, random_capacity = calc_random_locations(pixel_map, crop_id, version, reward_type=reward_type, pmnet=pmnet)

        if crop_id == sionna_crop_id:
            sionna_vis.move_camera(sionna_crop_id, top, left)
            # PPO
            sionna_vis.deploy_tx(tx_locs)
            sionna_vis.render_coverage()
            sionna_vis.remove_all_tx()
            # Exhaustive
            sionna_vis.deploy_tx(optimal_locs)
            sionna_vis.render_coverage(baseline="Exhaustive")
            sionna_vis.remove_all_tx()
            # Heuristic
            sionna_vis.deploy_tx(random_locs)
            sionna_vis.render_coverage(baseline="Heuristic")
            sionna_vis.remove_all_tx()
    
    
    mean_optimal_coverage = np.mean(optimal_coverages) 
    mean_optimal_capacity = np.mean(optimal_capacities) 
    mean_coverage = np.mean(coverages)
    mean_capacity = np.mean(capacities)   
    print(f"Mean Optimal Coverage: {mean_optimal_coverage}")
    print(f"Mean Optimal Capacity: {mean_optimal_capacity}")
    print(f"Mean Coverage: {mean_coverage}")
    print(f"Mean Capacity: {mean_capacity}")
    print(f"Coverage Ratio: {mean_coverage/mean_optimal_coverage}")
    print(f"Capacity Ratio: {mean_capacity/mean_optimal_capacity}")