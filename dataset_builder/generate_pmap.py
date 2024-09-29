import json
# Power Map Generation
import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io

from dataset_builder.pmnet_v3 import PMNet
from env.utils_v1 import ROOT_DIR, calc_coverages_and_save, load_map_normalized
from concurrent.futures import ProcessPoolExecutor


# Dataset preparation
def load_maps(dir_base: str = "usc", index: int = 1) -> np.ndarray:
    """Load pixel maps as np.ndarray from images.

    Args:
        dir_base (str): Base directory name where the maps are stored.
        index (int): Index of the specific map to load.

    Returns:
        np.ndarray: Normalized 256x256 grayscale map array with values in [0,1].
    """
    dir_maps = os.path.join(ROOT_DIR, dir_base, "map")
    filename = os.path.join(dir_maps, str(index)) + ".png"
    arr_map = np.asarray(io.imread(filename))

    return arr_map / 255  # 256 x 256 matrix with value in [0,1] (grayscale)


def generate_tx_layer(arr_map: np.ndarray, tx_size: int = 5, upsampling_factor: int = 4,
                      non_building_pixel_value: float = 0.) -> dict[int, np.ndarray]:
    """Generate TX layers corresponding to all valid TX locations on a map.
    
    Args:
        arr_map (np.ndarray): The input map array.
        tx_size (int): Size of the TX (transmitter) layer.
        upsampling_factor (int): Factor by which TX locations are upsampled.
        non_building_pixel_value (float): Pixel value representing non-building areas.

    Returns:
        dict[int, np.ndarray]: Dictionary mapping 1D indices to TX layer arrays.
    """
    tx_layers = {}
    map_size = arr_map.shape[0]  # 256
    n_steps = map_size // upsampling_factor # Calculate number of TX locations
    for row in range(n_steps):
        for col in range(n_steps):
            # only generate upsampled TX location corresponding to the action in auto BS
            y, x = row * upsampling_factor + (upsampling_factor - 1) // 2, col * upsampling_factor + (
                    upsampling_factor - 1) // 2
            if arr_map[y, x] != non_building_pixel_value:
                arr_tx = np.zeros((map_size, map_size), dtype=np.uint8)  # black background
                y_top, y_bottom = max(0, y - tx_size // 2), min(map_size, y + tx_size // 2 + 1),
                x_left, x_right = max(0, x - tx_size // 2), min(map_size, x + tx_size // 2 + 1)
                arr_tx[y_top: y_bottom, x_left: x_right] = 1.  # Mark TX location as white

                idx = map_size * y + x  # Compute 1D index for the TX location
                tx_layers[idx] = arr_tx  # Store TX layer in the dictionary

    return tx_layers


def create_dataset(input_dir_base: str = "usc", index: int = 1, tx_size: int = 1, upsampling_factor: int = 4,
                   non_building_pixel_value: float = 0., device: str = "cpu") -> tuple[list[str], torch.Tensor, dict[int, np.ndarray]]:
    """Create dataset for PMNet (cropped maps + TX locations).
    
    Args:
        input_dir_base (str): Base directory for input maps.
        index (int): Map index to load.
        tx_size (int): Size of the TX layer.
        upsampling_factor (int): Factor by which TX locations are upsampled.
        non_building_pixel_value (float): Pixel value representing non-building areas.
        device (str): Device to store tensors ("cpu" or "cuda").

    Returns:
        tuple[list[str], torch.Tensor, dict[int, np.ndarray]]: Tuple containing the list of map indices,
        the stacked tensor of maps and TX locations, and the dictionary of TX layers.
    """
    arr_map = load_maps(input_dir_base, index)
    idx_map_tx, tensors = [], []  # index (map index + tx index), tensor ([map, tx], ch=2)
    tx_layers = generate_tx_layer(arr_map, tx_size, upsampling_factor, non_building_pixel_value)
    for idx_tx, tx_layer in tx_layers.items():
        idx_data = str(index) + '_' + str(idx_tx) # Combine map index and TX index
        idx_map_tx.append(idx_data)
        arr_input = np.stack([arr_map, tx_layer], axis=0, dtype=np.float32) # Stack map and TX layers
        tensor_input = torch.from_numpy(arr_input) # Convert to torch tensor
        tensors.append(tensor_input)

    tensors = torch.stack(tensors, dim=0).to(device) # Stack tensors and move to specified device

    return idx_map_tx, tensors, tx_layers


def inference_and_save(model: nn.Module, idx: list[str], tensors: torch.Tensor, batch_size: int = 256,
                       dir_base: str = "usc", dir_img: str = "pmap"):
    """Use PMNet to generate power maps from the given dataset and save them.
    
    Args:
        model (nn.Module): The trained PMNet model.
        idx (list[str]): List of indices for the dataset.
        tensors (torch.Tensor): The input tensors for inference.
        batch_size (int): Number of samples per batch during inference.
        dir_base (str): Base directory for saving the power maps.
        dir_img (str): Subdirectory for saving the images.
    """
    assert len(idx) == tensors.size(dim=0) # Ensure the number of indices matches the number of tensors
    
    model.eval() # Set model to evaluation mode

    n_batches = len(idx) // batch_size + 1 if len(idx) % batch_size != 0 else len(idx) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(idx))

            batch_idx = idx[start: end]
            batch_tensors = tensors[start: end]

            preds = model(batch_tensors)
            preds = torch.clip(preds, 0, 1) # Clip predictions to valid range

            for j in range(len(preds)):
                file_name = 'pmap_' + batch_idx[j] + '.png'
                file_path = os.path.join(ROOT_DIR, dir_base, dir_img, file_name)
                arr = preds[j, 0].cpu().numpy()
                plt.imsave(file_path, arr, cmap='gray') # Save the predicted power map as a grayscale image


def inference(model: nn.Module, idx: list[str], tensors: torch.Tensor, tx_layers: dict, batch_size: int = 256, save: bool = False,
              **kwargs) -> dict[int, np.ndarray]:
    """Use PMNet to generate power maps from the given dataset.
    
    Args:
        model (nn.Module): The trained PMNet model.
        idx (list[str]): List of indices for the dataset.
        tensors (torch.Tensor): The input tensors for inference.
        tx_layers (dict): Dictionary of TX layers.
        batch_size (int): Number of samples per batch during inference.
        save (bool): Whether to save the generated power maps.
        **kwargs: Additional keyword arguments for saving options.

    Returns:
        dict[int, np.ndarray]: Dictionary mapping TX indices to generated power maps.
    """
    assert len(idx) == tensors.size(dim=0) # Ensure the number of indices matches the number of tensors
    model.eval() # Set model to evaluation mode

    n_batches = len(idx) // batch_size + 1 if len(idx) % batch_size != 0 else len(idx) // batch_size
    power_maps = {}

    with torch.no_grad():
        for i in (range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(idx))

            batch_idx = idx[start: end]
            batch_tensors = tensors[start: end]

            preds = model(batch_tensors)
            preds = torch.clip(preds, 0, 1) # Clip predictions to valid range

            for j in range(len(preds)):
                pmap_idx = int(batch_idx[j].split('_')[-1])
                arr = preds[j, 0].cpu().numpy()
                if True: #if mark_tx
                    # mark the tx location
                    tx_layer = tx_layers[pmap_idx]
                    arr_w_tx = np.where(tx_layer > arr, tx_layer, arr)
                power_maps[pmap_idx] = arr
                if save: # Save the generated power maps if required
                    map_idx, loc_idx = batch_idx[j].split('_')[0], batch_idx[j].split('_')[1]
                    img_dir = os.path.join(ROOT_DIR, kwargs['dir_base'], "debug", map_idx)
                    os.makedirs(img_dir, exist_ok=True)
                    file_path = os.path.join(img_dir, f'pmap_{map_idx}_{loc_idx}.png')
                    plt.imsave(file_path, arr, cmap='gray', vmin=0, vmax=1)

    return power_maps


def generate_pmaps(map_idx: int, upsampling_factor: int, batch_size: int, save: bool,
                   non_building_pixel: float, dir_base: str, dir_img: str) -> dict[int, np.ndarray]:
    """Generate all path loss maps corresponding to some TX locations given a building map.
    
    Args:
        map_idx (int): Index of the map to process.
        upsampling_factor (int): Factor by which TX locations are upsampled.
        batch_size (int): Number of samples per batch during inference.
        save (bool): Whether to save the generated power maps.
        non_building_pixel (float): Pixel value representing non-building areas.
        dir_base (str): Base directory for input maps.
        dir_img (str): Subdirectory for saving the images.

    Returns:
        dict[int, np.ndarray]: Dictionary mapping TX indices to generated power maps.
    """
    # Load PMNet Model Parameters
    pretrained_model = os.path.join(ROOT_DIR, 'dataset_builder/checkpoints/model_0.00008.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PMNet(n_blocks=[3, 3, 27, 3],
                  atrous_rates=[6, 12, 18],
                  multi_grids=[1, 2, 4],
                  output_stride=8, )
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    model = model.to(device)
    # Generate power maps using PMNet
    idx, tensors, tx_layers = create_dataset(input_dir_base=dir_base, index=map_idx, tx_size=5,
                                             upsampling_factor=upsampling_factor,
                                             non_building_pixel_value=non_building_pixel, device=device)
    power_maps = inference(model=model, idx=idx, tensors=tensors, tx_layers=tx_layers, batch_size=batch_size, save=save, dir_base=dir_base,
                           dir_img=dir_img)

    return power_maps


if __name__ == '__main__':

    num_data = 1296 # length of dataset
    for i in tqdm(np.arange(0, num_data)):
        generate_pmaps(i, 8, batch_size=32, save=True, non_building_pixel=1.,
                      dir_base='resource/usc_old_sparse', dir_img='power_map')
