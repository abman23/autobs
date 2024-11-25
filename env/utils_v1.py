from datetime import datetime
import json
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml
from tqdm import tqdm
import torch
import itertools
import cv2

from visualize.pmnet_v3 import get_output

# project root directory
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config= yaml.safe_load(open(os.path.join(ROOT_DIR, f'config.yaml'), 'r'))

action_space_size = config.get("action_space_size", 32)
map_size = config.get("map_size", 256)
upsampling_factor = map_size // action_space_size
non_building_pixel = config.get("non_building_pixel", 1.0)

coverage_thresholds = np.array([0.6078, 0.627, 0.647])


def dbm_to_mW(dbm):
    return 10 ** (dbm/10)

def mW_to_dbm(mW):
    return 10 * np.log10(mW)

sigma_sq = dbm_to_mW((coverage_thresholds[-1] * 255 - 255) - 6)
SNR_building = dbm_to_mW(- 255)/sigma_sq

def load_map_normalized(filepath: str) -> np.ndarray:
    """Convert map image to array (pixel value normalized to the range [0,1]).

    """
    image = Image.open(filepath).convert('L')
    image_arr = np.array(image, dtype=np.float32) / 255

    return image_arr


def dict_update(old_dict: dict, new_dict: dict) -> dict:
    """Updates the old dictionary with new key-value pairs in the new dictionary.

    Returns:
        The updated dictionary (not in-placed).
    """
    returned_dict = json.loads(json.dumps(old_dict))
    for key, value in new_dict.items():
        returned_dict[key] = value

    return returned_dict

def find_top_left(crop_id: int, crop_size: int = 512, stride: int = 100, height: int = 900, width: int = 900):
    crops_per_row = (width - crop_size) // stride + 1
    row = crop_id // crops_per_row
    col = crop_id % crops_per_row
    top = row * stride
    left = col * stride
    return top, left

def calc_action_mask(pixel_map: np.ndarray) -> np.ndarray:
    """Calculate the action mask in the reduced action space.

    Returns:
        A 0-1 flatten array of the action mask.

    """
    idx = np.arange((upsampling_factor - 1) // 2, map_size, upsampling_factor)
    # filter out non-building pixel
    action_pixels = np.where(pixel_map[idx][:, idx] != non_building_pixel, 1, 0)
    return action_pixels.reshape(-1).astype(np.int8)

def calc_upsampling_loc(action: int) -> tuple:
    """Calculate the location corresponding to a 'space-reduced' action by upsampling.

    Args:
        action: action in the reduced action space.

    Returns:
        Coordinate of the location - (row, col).

    """
    row_r, col_r = divmod(action, action_space_size)
    row = row_r * upsampling_factor + (upsampling_factor - 1) // 2
    col = col_r * upsampling_factor + (upsampling_factor - 1) // 2
    return row, col

def get_powermap(pixel_map, tx_layer, pmnet):
    inputs=np.stack([pixel_map, tx_layer], axis=2)
    power_map = get_output(pmnet, inputs)
    power_map[pixel_map != 1] = 0
    return power_map

def get_stats(city_map: np.ndarray, tx_locs: list[tuple], pmnet) -> tuple[np.ndarray, float, float]:
    
    num_roi = np.sum(city_map == non_building_pixel)
    pathgain_map_mW = np.zeros_like(city_map, dtype=float)
    tx_layer = np.zeros_like(city_map, dtype=float)
    
    for tx_loc in tx_locs:
        row, col = tx_loc[0], tx_loc[1]
        loc_idx = row * map_size + col
        
        tx_layer[row-3:row+3, col-3:col+3] = 1
        power_map_mW = dbm_to_mW(get_powermap(city_map, tx_layer, pmnet) * 255 - 255)
        
        pathgain_map_mW = np.add(power_map_mW, pathgain_map_mW)
        
    pathgain_map_db = mW_to_dbm(pathgain_map_mW)
    pathgain = (pathgain_map_db + 255)/255
    
    coverage_reward = calc_coverage(city_map, pathgain) 
    
    capacity_reward = calc_capacity(pathgain_map_mW) 
    
    return pathgain, coverage_reward, capacity_reward

def calc_coverage(city_map, pathgain_map: np.ndarray) -> float:
    """Calculate the overall coverage reward given the pathgain_map (dbm).

    Args:
        pathgain_map: np_array of pathgain in dbm

    Returns:
        (ratio of pixels covered)

    """
    num_roi = np.sum(city_map == non_building_pixel)
  
    covered = np.where(pathgain_map >= coverage_thresholds[-1], 1, 0)

    avg_coverage = 100 * (int(covered[covered==1].sum()) / num_roi)

    return avg_coverage

def calc_capacity(pathgain_map: np.ndarray) -> float:
    """Calculate the capacity reward reward given the pathgain_map (mW).

    Args:
        pathgain_map: np_array of pathgain in mW

    Returns:
        (capacity reward)

    """
    SNR_matrix = pathgain_map/sigma_sq
    
    SNR_matrix_roi = SNR_matrix[SNR_matrix > SNR_building]
    
    capacity_rewards = np.minimum(np.log2(1 + SNR_matrix_roi), 10)
    
    mean_capacity = np.mean(capacity_rewards)
    
    return float(mean_capacity)

def calc_optimal_locations(city_map, crop_id, version, reward_type, pmnet) -> tuple[np.ndarray, float, float]:

    locs_opt, reward_opt, pathgain_opt, coverage_opt, capacity_opt = [(-1, -1)], -1, np.zeros_like(city_map), 0, 0
    dataset_dir = "optimal_locs"
    data_dir = os.path.join(dataset_dir, f'optimal_{version}_{reward_type}')
    n_bs = 1 if version == 'single' else 2
    
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    filename = os.path.join(data_dir, f"optimal_{crop_id}.json")
    if not os.path.exists(filename):
        
        all_actions = itertools.combinations_with_replacement(range(32 ** 2), n_bs)

        for actions in all_actions:
            tx_locs = []
            flag = False
            for action in actions:
                row, col = calc_upsampling_loc(action)
                if city_map[row, col] == non_building_pixel:
                    # skip non-building pixel
                    flag = True
                    break
                tx_locs.append((row, col))
            if flag:
                continue
            
            pathgain, coverage_reward, capacity_reward = get_stats(city_map, tx_locs, pmnet)
        
            reward = coverage_reward if reward_type == 'coverage' else capacity_reward
            
            if reward > reward_opt:
                reward_opt = reward
                locs_opt = tx_locs
                pathgain_opt = pathgain
                coverage_opt = coverage_reward
                capacity_opt = capacity_reward
        
        opt_tx_layer = np.zeros_like(city_map)
    
        for tx_loc in locs_opt:
            row, col = tx_loc[0], tx_loc[1]
            opt_tx_layer[row-3:row+3, col-3:col+3] = 1
        
        cv2.imwrite(f"visualize/output/{version}_{crop_id}_exhaustive.png", 255 * (pathgain_opt + opt_tx_layer))

        # save result to avoid repeatedly computation
        result = {"locs_opt": locs_opt, 'coverage_opt': coverage_opt, 'capacity_opt': capacity_opt}
        json.dump(result, open(filename, 'w'))
    else:
        result = json.load(open(filename))
        locs_opt, coverage_opt, capacity_opt = result["locs_opt"], result["coverage_opt"], result["capacity_opt"]

    return locs_opt, coverage_opt, capacity_opt

def calc_random_locations(city_map, crop_id, version, reward_type, pmnet):
    n_bs = 1 if version == 'single' else 2
    rand_locs = []
    rand_tx_layer = np.zeros_like(city_map, dtype=int)
    indices = np.where(city_map != non_building_pixel)
    
    for _ in range(n_bs):
        random_index = np.random.randint(0, len(indices[0]))
        rand_row, rand_col = indices[0][random_index], indices[1][random_index]
        rand_locs.append((rand_row, rand_col))
        rand_tx_layer[rand_row-3:rand_row+3, rand_col-3:rand_col+3] = 1
     
    pathgain, rand_coverage, rand_capacity = get_stats(city_map, rand_locs, pmnet)
    
    cv2.imwrite(f"visualize/output/{version}_{crop_id}_heuristic.png", 255 * (pathgain + rand_tx_layer))
    
    return rand_locs, rand_coverage, rand_capacity