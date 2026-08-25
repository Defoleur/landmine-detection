import os
import carb
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

class ImageMapHelper:
    @staticmethod
    def load_probability_map(file_name):
        """
        Loads a probability map image once and keeps it in memory.
        Returns the PIL Image object or None if loading fails.
        """
        try:
            # Construct absolute path assuming the script is in the same dir as images
            # Or adjust based on your project root
            map_path = os.path.abspath(file_name)
            
            if not os.path.exists(map_path):
                carb.log_warn(f"Probability map not found: {map_path}")
                return None
                
            img = Image.open(map_path).convert("L") # Convert to grayscale immediately
            return img
        except Exception as e:
            carb.log_warn(f"Failed to load image {file_name}: {e}")
            return None
        
    @staticmethod
    def build_probability_cdf(map_array, map_bounds, spawn_bounds):
        """
        Pre-calculates the Cumulative Distribution Function (CDF) for valid pixels.
        Returns: (cdf_array, valid_pixel_indices_u, valid_pixel_indices_v, crop_offset)
        """
        if map_array is None:
            return None

        height, width = map_array.shape

        # Unpack Bounds
        map_min_x, map_min_y = map_bounds[0][0], map_bounds[0][1]
        map_max_x, map_max_y = map_bounds[1][0], map_bounds[1][1]
        map_range_x = map_max_x - map_min_x
        map_range_y = map_max_y - map_min_y

        spawn_min_x, spawn_min_y = spawn_bounds[0][0], spawn_bounds[0][1]
        spawn_max_x, spawn_max_y = spawn_bounds[1][0], spawn_bounds[1][1]

        # 1. Convert Spawn Bounds (World) -> Image Crop Bounds (Pixels)
        # We only care about pixels that are INSIDE the spawn area.
        # Logic: World X -> Image V (Height), World Y -> Image U (Width)
        
        # Norm factors
        norm_min_x = (spawn_min_x - map_min_x) / map_range_x
        norm_max_x = (spawn_max_x - map_min_x) / map_range_x
        norm_min_y = (spawn_min_y - map_min_y) / map_range_y
        norm_max_y = (spawn_max_y - map_min_y) / map_range_y

        # Map to pixels (V=Row, U=Col)
        v_min = int(norm_min_x * (height - 1))
        v_max = int(norm_max_x * (height - 1))
        u_min = int(norm_min_y * (width - 1))
        u_max = int(norm_max_y * (width - 1))

        # Ensure correct ordering and clipping
        v_start, v_end = sorted((v_min, v_max))
        u_start, u_end = sorted((u_min, u_max))
        
        v_start = max(0, v_start)
        v_end = min(height, v_end + 1) # +1 for slicing inclusive
        u_start = max(0, u_start)
        u_end = min(width, u_end + 1)

        # 2. Crop the Map Array
        # This drastically reduces loop/calculation time
        crop = map_array[v_start:v_end, u_start:u_end]
        
        if crop.size == 0:
            return None

        # 3. Filter Valid Pixels (Vectorized)
        # Valid = (Weight > 0)
        # We flatten the crop to 1D arrays for CDF calculation
        valid_mask = crop > 0
        if not np.any(valid_mask):
            return None
            
        weights = crop[valid_mask]
        
        # Get coordinates of valid pixels relative to the crop
        # v_local (rows), u_local (cols)
        v_local, u_local = np.where(valid_mask)

        # 4. Calculate CDF
        # Cumulative sum of brightness values
        cdf = np.cumsum(weights, dtype=np.float64)
        # Normalize to 0.0 - 1.0
        cdf /= cdf[-1]

        return {
            "cdf": cdf,
            "weights": weights,
            "u_local": u_local,
            "v_local": v_local,
            "offset_u": u_start,
            "offset_v": v_start,
            "map_dims": (width, height),
            "map_bounds": map_bounds
        }

    @staticmethod
    def generate_position_from_cdf(cdf_data, z_bounds, rng=None):
        """
        O(log N) generation using the pre-calculated CDF.
        """
        if rng is None:
            rng = random
        if cdf_data is None:
            return None # Fallback needed

        cdf = cdf_data["cdf"]
        
        # 1. Generate Random Value (0-1)
        r = rng.random()
        
        # 2. Find Index in CDF (Binary Search)
        idx = np.searchsorted(cdf, r)
        
        # 3. Retrieve Pixel Coordinates
        u_local = cdf_data["u_local"][idx]
        v_local = cdf_data["v_local"][idx]
        chosen_weight = cdf_data["weights"][idx]
        
        # Add Global Offset
        pixel_u = u_local + cdf_data["offset_u"]
        pixel_v = v_local + cdf_data["offset_v"]

        # 4. Convert Pixel -> World with Jitter
        # Add uniform jitter (-0.5 to 0.5) to simulate continuous position within the pixel
        # otherwise all objects align to a strict grid
        pixel_u_jit = pixel_u + rng.uniform(-0.5, 0.5)
        pixel_v_jit = pixel_v + rng.uniform(-0.5, 0.5)

        width, height = cdf_data["map_dims"]
        map_bounds = cdf_data["map_bounds"]
        
        min_x, min_y = map_bounds[0][0], map_bounds[0][1]
        max_x, max_y = map_bounds[1][0], map_bounds[1][1]
        range_x = max_x - min_x
        range_y = max_y - min_y

        # Inverse Logic: 
        # Pixel U (Width) -> World Y
        # Pixel V (Height) -> World X
        
        norm_y = pixel_u_jit / (width - 1)
        norm_x = pixel_v_jit / (height - 1)
        
        world_y = norm_y * range_y + min_y
        world_x = norm_x * range_x + min_x
        # world_z = random.uniform(z_bounds[0], z_bounds[1])

        if chosen_weight >= 255:
            world_z = 0.15  # On the white area
        else:
            world_z = 0.17

        return [world_x, world_y, world_z]

    @staticmethod
    def save_spawn_visualization(positions, map_image, map_bounds, output_filename):
        """
        Saves a debug image. Positions are mapped relative to map_bounds.
        """
        if map_image is None or not positions:
            return

        try:
            width, height = map_image.size
            min_x, min_y = map_bounds[0][0], map_bounds[0][1]
            max_x, max_y = map_bounds[1][0], map_bounds[1][1]
            range_x = max_x - min_x
            range_y = max_y - min_y

            u_coords = []
            v_coords = []

            for pos in positions:
                wx, wy = pos[0], pos[1]
                
                # Check if point is actually on the map for visualization
                if not (min_x <= wx <= max_x and min_y <= wy <= max_y):
                    continue

                norm_x = (wx - min_x) / range_x if range_x > 0 else 0
                norm_y = (wy - min_y) / range_y if range_y > 0 else 0
                
                pixel_u = int(norm_y * (width - 1))
                pixel_v = int(norm_x * (height - 1))
                
                u_coords.append(pixel_u)
                v_coords.append(pixel_v)

            plt.figure(figsize=(10, 10))
            plt.imshow(map_image, cmap='gray')
            plt.scatter(u_coords, v_coords, c='red', s=10, alpha=0.7, label='Spawned')
            plt.title(f"Spawn Visualization: {output_filename.split('/')[-1]}")
            plt.axis('off')
            
            save_path = os.path.abspath(output_filename)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            carb.log_info(f"Saved spawn debug image to: {save_path}")

        except Exception as e:
            carb.log_warn(f"Failed to save spawn visualization: {e}")
