import numpy as np

from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

class Sionna_Visualizer():
    def __init__(self, city_map):
        self.scene = load_scene("visualize/USC.xml")

        self.city_map = city_map

        self.scene.tx_array = PlanarArray(
            num_rows = 4,
            num_cols = 4,
            vertical_spacing = 0.5,
            horizontal_spacing = 0.5,
            pattern = "iso",
            polarization="V"
        )

        self.scene.rx_array = PlanarArray(
            num_rows = 1,
            num_cols = 1,
            vertical_spacing = 0.5,
            horizontal_spacing = 0.5,
            pattern = "hw_dipole",
            polarization="V"
        )

        for obj_name in self.scene.objects:
            obj = self.scene.get(obj_name)
            if self.scene.get(obj_name).radio_material.name == "itu_concrete.002":
                obj.radio_material = "itu_concrete"
        self.tx_num = 0
        
    def move_camera(self, crop_id, top, left):
        self.crop_id, self.top, self.left = crop_id, top, left
        cam_view = np.array((top, left)) + 256
        self.cam = Camera("cam{i}", (cam_view[1]-400-75, 500-cam_view[0]-125, 650), orientation=(np.pi/2, np.pi/2, 0))
        
    def deploy_tx(self, tx_locs: list[tuple]):
        for row, col in tx_locs:
            image_pos = np.array((col, row)) * (-2, 2) + (-self.left, self.top)
            tx_position = (-500, 400) - image_pos
            z = self.city_map[image_pos[1], -image_pos[0]]
            z = z if z < 255 else 0

            self.tx_num += 1
            tx = Transmitter(f"tx_pred_{self.tx_num}", (tx_position[0], tx_position[1], z+2), [0.0, 0.0, 0.0])
            self.scene.add(tx)
            
    def remove_all_tx(self):
        for i in range(self.tx_num, 0, -1):
            self.scene.remove(f"tx_pred_{i}")
        self.tx_num = 0
    
    def render_coverage(self, baseline=None):
        cm = self.scene.coverage_map(
            cm_cell_size=(1.0, 1.0), los=True,
            diffraction=False, scattering=False, edge_diffraction=False, max_depth = 8, num_samples = 7.99*(10**6)
        )
        if self.tx_num > 1: version = "multi"
        else: version = "single"
        
        file_name = f"visualize/sionna_output/{version}_{self.crop_id}"
        if baseline: file_name = file_name + f'_{baseline}'
        else: file_name = file_name + '_ppo'
        self.scene.render_to_file(self.cam, file_name+'.png', coverage_map=cm, resolution=(500,500))