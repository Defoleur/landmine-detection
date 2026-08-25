import omni.isaac.sensor
from omni.isaac.sensor import Camera
import omni.replicator.core as rep  # Required for accessing the mapping
import numpy as np
import carb
import math
from pxr import Gf
import omni.usd


from omni.isaac.core.prims import XFormPrim
from pxr import UsdGeom, Usd
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.prims import XFormPrim

import os
import json

class CameraWrapper:
    def __init__(self, prim_path, resolution, 
                 frequency=30, local_pose=None, 
                 focal_length=3.0, clip_range=(0.01, 10000),
                 mines_detection_enabled=False,
                 camera_mode='ground'):

        self.prim_path = prim_path
        self.resolution = resolution
        self.width, self.height = resolution
        self.frequency = frequency
        self.local_pose = local_pose if local_pose is not None else np.array([0.0, 0.0, 0.0])
        self.focal_length = focal_length
        self.clip_range = clip_range
        self._camera_mode = camera_mode
        
        self._cam = None
        self._annotator = None  # We will keep a reference to the Replicator annotator
        self._depth_annotator = None  # ADDED: Reference for the depth annotator
        self._mine_position_cache = {}
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self.mines_json_path = os.path.join(self.script_dir, "../../../../pegasus.simulator/pegasus/simulator/config/mines_positions.json")
        self.mines_detection_enabled = mines_detection_enabled

    def initialize(self):
        if self.is_initialized():
            return

        try:
            # 1. Create the Isaac Sensor Camera
            self._cam = Camera(
                prim_path=self.prim_path,
                frequency=self.frequency,
                resolution=self.resolution
            )
            if self._camera_mode == 'front':
                self._cam.set_local_pose(self.local_pose, np.array([1.0, 0.0, 0.0, 0.0]))
            else:
                self._cam.set_local_pose(self.local_pose, np.array([0.7071068, 0.0, 0.7071068, 0.0]))
            self._cam.set_focal_length(self.focal_length)
            self._cam.set_clipping_range(self.clip_range[0], self.clip_range[1])
            
            # cx = self.width / 2.0
            # cy = self.height / 2.0
            
            # # Mock focal length in pixels (standard for a ~90 degree FOV)
            # fx = 458.1245681803812
            # fy = 458.1245681803812
            
            # # Dramatic distortion coefficients (k1, k2, k3, k4)
            # test_distortion = [0.0, 0.0, 0.0, 0.0] 
            
            # # Apply the OpenCV Fisheye math to the USD Prim
            # self._cam.set_opencv_fisheye_properties(
            #     cx=cx, cy=cy, fx=fx, fy=fy, fisheye=test_distortion
            # )

            self._cam.initialize()

            # 2. Get the Render Product Path created by this camera
            rp_path = self._cam.get_render_product_path()

            # 3. Attach a Replicator Annotator to this Render Product
            # "instance_segmentation" provides both the pixel map AND the ID mapping
            # self._annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
            if self.mines_detection_enabled:
                self._annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
                self._annotator.attach([rp_path])

            self._depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self._depth_annotator.attach([rp_path])

            
        except Exception as e:
            carb.log_error(f"CameraWrapper: Failed to initialize: {e}")
            self._cam = None
            self._annotator = None
            self._depth_annotator = None

    def _load_mines_cache(self):
        """Loads mine positions from JSON into memory."""
        if not self.mines_json_path or not os.path.exists(self.mines_json_path):
            carb.log_warn(f"CameraWrapper: Mines JSON not found at {self.mines_json_path}")
            return

        try:
            with open(self.mines_json_path, 'r') as f:
                data = json.load(f)
            self._mine_position_cache = {}
            count = 0
            for path, pos_list in data.items():
                # Store as numpy array for fast distance calc
                self._mine_position_cache[path] = np.array(pos_list)
                count += 1
                
            carb.log_info(f"CameraWrapper: Loaded {count} mine positions from JSON.")
            
        except Exception as e:
            carb.log_error(f"CameraWrapper: Error loading mines JSON: {e}")

    def is_initialized(self) -> bool:
        # return self._cam is not None and self._annotator is not None and self._depth_annotator is not None
        return self._cam is not None and self._depth_annotator is not None

    def set_camera_mode(self, mode: str):
        if not self._cam: return
        self._camera_mode = mode
        if mode == 'front':
            self._cam.set_local_pose(self.local_pose, np.array([1.0, 0.0, 0.0, 0.0]))
        else:
            self._cam.set_local_pose(self.local_pose, np.array([0.7071068, 0.0, 0.7071068, 0.0]))
    
    def get_visible_prims(self, name_filter="mine", max_dist=20.0) -> list[str]:
        """
        Returns detailed data for visible mines.
        
        Returns:
            list[dict]: A list of dicts with keys:
                        - 'path': (str) USD Prim path
                        - 'distance': (float) Distance from camera in meters
                        - 'bbox': (list) [x_min, y_min, x_max, y_max] in pixels
        """
        if not self.is_initialized(): return []

        try:
            # 1. Get Annotator Data
            annotator_data = self._annotator.get_data()
            # print(annotator_data)
            if "data" not in annotator_data or "info" not in annotator_data:
                return []

            # Shape is (H, W, 1) -> Squeeze to (H, W)
            instance_pixels = annotator_data["data"].squeeze() 
            id_to_path = annotator_data["info"].get("idToLabels", {})
            if instance_pixels is None: return []

            cam_pos, _ = self.get_world_pose()
            
            # 2. Find Unique IDs
            visible_ids = np.unique(instance_pixels)
            results = []
            filter_lower = name_filter.lower()
            # for instance_id in visible_ids:
            #     if instance_id == 0: continue # Background
            #     # INFO: FOR instance_segmentation
            #     # str_id = str(instance_id)
            #     # INFO: FOR instance_segmentation_fast
            #     str_id = instance_id
            #     if str_id in id_to_path:
            #         prim_path = id_to_path[str_id]
            #         print("Prim_path:", prim_path)
            #         if filter_lower in prim_path.lower():
            #             try:
            #                 # --- A. Check Distance ---
            #                 prim_xform = XFormPrim(prim_path)
            #                 obj_pos, _ = prim_xform.get_world_pose()
            #                 dist = np.linalg.norm(obj_pos - cam_pos)
                            
            #                 if dist <= max_dist:
            #                     # --- B. Calculate Bounding Box from Mask ---
            #                     # Find where pixels equal this ID
            #                     # Note: np.where returns (row_indices, col_indices) -> (y, x)
            #                     y_indices, x_indices = np.where(instance_pixels == instance_id)
                                
            #                     if len(x_indices) > 0:
            #                         x_min = int(np.min(x_indices))
            #                         x_max = int(np.max(x_indices))
            #                         y_min = int(np.min(y_indices))
            #                         y_max = int(np.max(y_indices))
                                    
            #                         results.append({
            #                             "path": prim_path,
            #                             "distance": float(dist),
            #                             "bbox": [x_min, y_min, x_max, y_max] # [x_min, y_min, x_max, y_max]
            #                         })
            #             except Exception:
            #                 continue
            # return results
            for instance_id in visible_ids:
                if instance_id == 0: continue 
                
                # Check mapping (str or int depending on Replicator version)
                prim_path = id_to_path.get(str(instance_id)) or id_to_path.get(int(instance_id))
                
                if prim_path and filter_lower in prim_path.lower():
                    
                    # --- OPTIMIZED DISTANCE CHECK ---
                    # Look up position in cache instead of querying USD
                    obj_pos = self._mine_position_cache.get(prim_path)
                    
                    # If not in cache (e.g. spawned dynamically), try to fetch it once
                    if obj_pos is None:
                        try:
                            obj_pos, _ = XFormPrim(prim_path).get_world_pose()
                            self._mine_position_cache[prim_path] = obj_pos # Cache it for next time
                        except:
                            continue

                    dist = np.linalg.norm(obj_pos - cam_pos)
                    
                    if dist <= max_dist:
                        # --- Calculate Bounding Box ---
                        y_indices, x_indices = np.where(instance_pixels == instance_id)
                        
                        if len(x_indices) > 0:
                            x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                            y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                            
                            results.append({
                                "path": prim_path,
                                "distance": float(dist),
                                "bbox": [x_min, y_min, x_max, y_max]
                            })
            return results

        except Exception as e:
            carb.log_warn(f"CameraWrapper: Error processing frame: {e}")
            return []
    
    def get_world_pose(self):
        if self._cam: return self._cam.get_world_pose()
        return (np.array([0,0,0]), np.array([1,0,0,0]))
    # --- Standard Getters (Delegating to _cam) ---

    def get_rgba(self) -> np.ndarray | None:
        if not self._cam: return None
        data = self._cam.get_rgba()
        if data is None or data.ndim != 3:
            return None
        return data[:, :, :3]

    def get_depth(self) -> np.ndarray | None:
        """
        Returns a 2D numpy array (H, W) containing the depth map in meters/units.
        """
        if not self.is_initialized() or not self._depth_annotator:
            return None
        
        try:
            depth_data = self._depth_annotator.get_data()
            if depth_data is None:
                return None
            
            # Replicator sometimes wraps the array in a dict depending on the Omniverse version
            if isinstance(depth_data, dict):
                depth_data = depth_data.get("data", depth_data)
                
            # Squeeze removes empty dimensions (e.g., converting (H, W, 1) to (H, W))
            return np.squeeze(depth_data)
            
        except Exception as e:
            carb.log_warn(f"CameraWrapper: Error getting depth data: {e}")
            return None

    def get_intrinsics(self) -> dict | None:
        if not self._cam: return None
        try:
            f = self._cam.get_focal_length()
            h_ap = self._cam.get_horizontal_aperture()
            return {
                "focal_length": f,
                "horizontal_aperture": h_ap,
                "vertical_aperture": self._cam.get_vertical_aperture(),
                "clipping_range": self._cam.get_clipping_range(),
                "fov_horizontal": 2 * math.atan(h_ap / (2 * f))
            }
        except: return None

    def set_world_pose(self, pos, orient):
        if self._cam: self._cam.set_world_pose(pos, orient)

    def reset(self):
        self._cam = None
        if self._annotator:
            self._annotator.detach()
            self._annotator = None
            
    # --- Helper ---
    @staticmethod
    def _look_at(src, tgt):
        d = np.array(tgt) - np.array(src)
        n = np.linalg.norm(d)
        if n < 1e-6: return [1,0,0,0]
        d /= n
        y = math.atan2(d[1], d[0])
        p = math.asin(np.clip(d[2], -1, 1))
        # Z-up quaternion (w, x, y, z) -> Isaac (w, x, y, z)
        # Note: Gf.Quatf creates real, i, j, k. 
        # Isaac set_world_pose expects [w, x, y, z]
        q = Gf.Quatf(math.cos(y/2)*math.cos(p/2), 0, math.sin(p/2), math.sin(y/2))
        return [q.GetReal(), q.GetImaginary()[0], -q.GetImaginary()[1], q.GetImaginary()[2]]

    def update_chase_camera(self, target, offset=(-6,0,1), alpha=0.2):
        if not self._cam or target is None: return
        curr_p, _ = self.get_world_pose()
        des_p = np.array(target) + np.array(offset)
        new_p = curr_p * (1-alpha) + des_p * alpha
        self.set_world_pose(new_p, self._look_at(new_p, target))
    
    def update_stabilized(self, drone_pos, drone_quat, offset=(0.12, 0, -0.15)):
        if not self._cam: return
        w, x, y, z = drone_quat
        yaw = math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
        c, s = math.cos(yaw), math.sin(yaw)
        roff = np.array([offset[0]*c - offset[1]*s, offset[0]*s + offset[1]*c, offset[2]])
        world_pos = np.array(drone_pos) + roff
        qw, qz = math.cos(yaw/2), math.sin(yaw/2)
        q_yaw = np.array([qw, 0.0, 0.0, qz])
        q_nadir = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        w1, x1, y1, z1 = q_yaw
        w2, x2, y2, z2 = q_nadir
        orient = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        self._cam.set_world_pose(world_pos, orient)

    def log_intrinsics(self):
        """Logs calculated intrinsics from the camera."""
        intrinsics = self.get_intrinsics()
        if intrinsics:
            focal_length = intrinsics["focal_length"]
            v_ap = intrinsics["vertical_aperture"]
            h_ap = intrinsics["horizontal_aperture"]
            
            # Use self.height and self.width stored during __init__
            focal_x = self.height * focal_length / v_ap
            focal_y = self.width * focal_length / h_ap
            center_x = self.width * 0.5
            center_y = self.height * 0.5
            carb.log_warn(f"Intrinsics for {self.prim_path}: {intrinsics}")
            carb.log_warn(f"Focal X: {focal_x}, Focal Y: {focal_y}, Center X: {center_x}, Center Y: {center_y}")
        else:
            carb.log_warn(f"Camera {self.prim_path} not ready, cannot log intrinsics.")
