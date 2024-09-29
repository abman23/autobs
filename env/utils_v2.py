import numpy as np
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from env.utils_v1 import load_map_normalized


def reward_to_color(reward, min_reward=0.47, max_reward=0.71): # Min reward and max reward needs to be updated wrt environment 
    # Normalize reward to range [0, 1]
    reward = np.clip(reward, min_reward, max_reward)
    normalized_reward = (reward - min_reward) / (max_reward - min_reward)
    threshold = 0.5
    
    # Map reward to RGB color
    if normalized_reward < threshold:
        # Low rewards mapped to blue
        return (0, int(255 * (2 * normalized_reward)), 255)
    else:
        # High rewards mapped to red
        return (int(255 * (2 * (normalized_reward - threshold))), 255, 0)

def rgb_to_heatmap(rgb_array):
    # Normalize the RGB values to the range [0, 1]
    rgb_normalized = rgb_array / 255.0
    
    # Convert RGB to grayscale using the luminosity method
    grayscale = 0.2989 * rgb_normalized[..., 0] + 0.5870 * rgb_normalized[..., 1] + 0.1140 * rgb_normalized[..., 2]
    
    # Apply a colormap to the grayscale image to create a heatmap
    heatmap = cm.viridis(grayscale)
    
    # Remove the alpha channel
    heatmap = heatmap[..., :3]
    
    return (heatmap * 255).astype(np.uint8)
        
def draw_map_with_tx(filepath: str, pixel_map: np.ndarray, mark_size: int,
                     target_locs: list, curr_locs: list, rewards: dict, save: bool = True) -> Image:
    """Save building map array as a black-white image and mark TX locations.

    Args:
        filepath: Path of the image.
        pixel_map: Building map array.
        mark_size: Size of the marker.
        target_locs: Coordinates of target markers.
        curr_locs: Coordinates of current markers.
        save: Save the image or not.

    Returns:

    """
    # convert the binary array to an image
    image_from_array = Image.fromarray((255 * pixel_map).astype(np.uint8), mode='L')
    image_from_array = image_from_array.convert('RGB')
            
    # Create an ImageDraw object to draw on the image
    draw = ImageDraw.Draw(image_from_array)
    
    for action, reward in rewards.items():
        y, x = np.array(action.split(',')).astype(np.int16)
        map_size = pixel_map.shape[0]
        top_left = (max(0, x - (mark_size - 1) // 2), max(0, y - (mark_size - 1) // 2))
        bottom_right = (min(map_size, x + mark_size // 2 + 1), min(map_size, y + mark_size // 2 + 1))
        processed_reward = int(255*reward)
        
        color = reward_to_color(reward)
        # Draw the red point
        draw.rectangle((top_left, bottom_right), fill=color)
    

    for loc in target_locs:
        x, y = loc[1], loc[0]
        map_size = pixel_map.shape[0]
        top_left = (max(0, x - (mark_size - 1) // 2), max(0, y - (mark_size - 1) // 2))
        bottom_right = (min(map_size, x + mark_size // 2 + 1), min(map_size, y + mark_size // 2 + 1))
        # Draw the red point
        draw.rectangle((top_left, bottom_right), fill="red")

    for loc in curr_locs:
        x, y = loc[1], loc[0]
        map_size = pixel_map.shape[0]
        top_left = (max(0, x - (mark_size - 1) // 2), max(0, y - (mark_size - 1) // 2))
        bottom_right = (min(map_size, x + mark_size // 2 + 1), min(map_size, y + mark_size // 2 + 1))
        # draw.point(xy, fill='blue')
        draw.rectangle((top_left, bottom_right), fill="blue")

    if save:
        image_from_array.save(filepath)
    return np.array(image_from_array)

def plot_reward_heatmap(rewards, grid_size):
    """Plot a heatmap of rewards for all possible locations.

    Args:
        rewards (dict): Dictionary containing rewards for all possible locations.
        grid_size (int): Size of the grid (assuming a square grid).
    """
    reward_matrix = np.zeros((grid_size, grid_size))
    
    # Storing rewards in reward matrix (excluding roi region)
    for action, reward in rewards.items():
        for loc in action:
            row, col = loc // grid_size, loc % grid_size
            reward_matrix[row, col] = max(reward_matrix[row, col], reward)

    # Plotting reward for all possible locations
    plt.figure(figsize=(8, 6))
    plt.imshow(reward_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Reward')
    plt.title('Reward Heatmap for All Possible Locations')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def plot_coverage(filepath: str, pixel_map: np.ndarray, coverage_curr: np.ndarray, coverage_opt: np.ndarray,
                  tx_locs: list, opt_tx_locs: list, rewards: dict, save: bool = True) -> None:
    """Plot the coverage map of current TX locations vs. optimal TX locations, plus the pixel map.

    Args:
        filepath (str): The file path where the plot will be saved.
        pixel_map (np.ndarray): The base pixel map, typically showing building vs. non-building areas.
        coverage_curr (np.ndarray): The current coverage map based on the current TX locations.
        coverage_opt (np.ndarray): The optimal coverage map based on the best TX locations.
        tx_locs (list): List of current TX locations.
        opt_tx_locs (list): List of optimal TX locations.
        rewards (dict): A dictionary containing the rewards for each location.
        save (bool): Whether to save the plot to a file or just display it.

    Returns:
        None
    """
    # Set up the plot with three subplots in a single row
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Create a copy of the pixel map to highlight regions of interest (buildings)
    roi_map = pixel_map.copy()
    
    # Identify non-building pixels (assuming buildings are marked with 1)
    building_pixels = pixel_map != 1
    
    # Set non-building regions to 0 (black) in the region of interest map
    roi_map[building_pixels] = 0
    
    # Plot the base pixel map with transmitter locations marked
    pixel_map_with_tx = draw_map_with_tx(filepath='', pixel_map=roi_map, mark_size=3,
                                         target_locs=[], curr_locs=[], rewards=rewards, save=False)
    axes[0].imshow(pixel_map_with_tx)  # Show the base pixel map in the first subplot

    # Update the current coverage map to highlight non-building regions
    coverage_curr[building_pixels] = pixel_map[building_pixels]
    coverage_img = draw_map_with_tx(filepath='', pixel_map=coverage_curr, mark_size=3,
                                    target_locs=[], curr_locs=tx_locs, rewards=dict({}), save=False)
    current_heatmap = rgb_to_heatmap(coverage_img)  # Convert RGB image to a heatmap for better visualization
    current_heatmap[building_pixels] = coverage_img[building_pixels]  # Keep building regions intact
    axes[1].imshow(current_heatmap)  # Show the current TX coverage in the second subplot

    # Update the optimal coverage map similarly to highlight non-building regions
    coverage_opt[building_pixels] = pixel_map[building_pixels]
    coverage_img_opt = draw_map_with_tx(filepath='', pixel_map=coverage_opt, mark_size=3,
                                        target_locs=opt_tx_locs, curr_locs=[], rewards=dict({}), save=False)
    target_heatmap = rgb_to_heatmap(coverage_img_opt)  # Convert RGB image to a heatmap
    target_heatmap[building_pixels] = coverage_img_opt[building_pixels]  # Keep building regions intact
    axes[2].imshow(target_heatmap)  # Show the optimal TX coverage in the third subplot

    # Set titles for each subplot
    axes[0].set_title('Building Map')
    axes[1].set_title('Deployed TXs\' Coverage')
    axes[2].set_title('Optimal TXs\' Coverage')

    # Save the plot to a file if required
    if save:
        fig.savefig(filepath, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    draw_map_with_tx('./test.png', pixel_map=load_map_normalized('../resource/usc_old_sparse/map/1.png'), mark_size=5,
                     target_locs=[[12, 120], [35, 37]], curr_locs=[[90, 90], [200, 200]])