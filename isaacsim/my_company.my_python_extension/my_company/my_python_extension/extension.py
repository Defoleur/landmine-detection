# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import omni.ext
import carb.events
import omni.kit.app
import omni.timeline
import omni.usd
import numpy as np
import time
from pxr import Usd
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.debug_draw")
from isaacsim.util.debug_draw import _debug_draw
import numpy as np
from .helpers.velocity_reader import VelocityReader
from .helpers.camera_wrapper import CameraWrapper
from .helpers.trajectory_drawer import TrajectoryDrawer
from .helpers.scene_utils import SceneUtils
from .helpers.udp_helper import UDPHelper
from .helpers.thermal_camera import ThermalCamera
from .helpers.chase_camera import ChaseCamera
from .helpers.drone_trail import DroneTrail
from .helpers.dataset_recorder import DatasetRecorder
from .helpers.auto_record_manager import AutoRecordManager

UDP_IP = "127.0.0.1"
UDP_IMAGE_PORT = 5005
UDP_VELOCITY_PORT = 5006
UDP_TAG_PORT = 5007
UDP_POSITION_PORT = 5008
UDP_TRAJECTORY_PORT = 5010
UDP_TRAJECTORY_PORTS = [5010, 5011, 5012]
UDP_MINES_PORT = 5013
UDP_DEPTH_PORT = 5014
UDP_SECOND_IMAGE_PORT = 5015

WIDTH = 1280
HEIGHT = 960

# Any class derived from `omni.ext.IExt` in the top level module (defined in
# `python.modules` of `extension.toml`) will be instantiated when the extension
# gets enabled, and `on_startup(ext_id)` will be called. Later when the
# extension gets disabled on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    """This is a blank extension template."""
    # ext_id is the current extension id. It can be used with the extension
    # manager to query additional information, like where this extension is
    # located on the filesystem.

    def on_update(self, dt: float):
        s = carb.settings.get_settings()
        if self.is_simulation_active:
            if self._camera is None:
                self.stage = omni.usd.get_context().get_stage()

                self.body_prim = self.stage.GetPrimAtPath("/World/quadrotor/body")
                if not self.body_prim.IsValid():
                    carb.log_warn("Quadrotor body not found in stage - skipping camera init.")
                    return
                
                self._discover_drones()
                carb.log_warn(f"Prims: {self.all_drone_prims}")

                camera_path = s.get("/ext/my_extension/camera_path") or "/World/quadrotor/body/camera_022"
                cam_w = s.get_as_int("/ext/my_extension/camera_width") or 800
                cam_h = s.get_as_int("/ext/my_extension/camera_height") or 600
                mines_detection = s.get_as_bool("/ext/my_extension/mines_detection") or False
                chase_camera_enabled = s.get_as_bool("/ext/my_extension/chase_camera_enabled") or False
                thermal_camera_enabled = s.get_as_bool("/ext/my_extension/thermal_camera_enabled") or False
                follow_camera_path = s.get("/ext/my_extension/follow_camera_path") or "/World/camera_follow"
                camera_mode_idx = s.get_as_int("/ext/my_extension/camera_mode") or 0
                camera_mode = 'front' if camera_mode_idx == 0 else 'ground'
                
                self._camera = CameraWrapper(
                    prim_path=camera_path,
                    resolution=(cam_w, cam_h),
                    local_pose=np.array([0.12, 0, 0]),
                    frequency=30,
                    focal_length=1.5,
                    mines_detection_enabled=mines_detection,
                    camera_mode=camera_mode
                )
                
                self._camera.initialize()
                self._camera.log_intrinsics()
                self._last_camera_mode = camera_mode

                self.last_mine_check_time = time.time()

                if chase_camera_enabled:
                    self.chase_camera = ChaseCamera(
                        prim_path=follow_camera_path,
                        resolution=(cam_w, cam_h),
                        focal_length=1.5,
                        offset=(-0.5, 0.0, 0.5),
                        smoothness=5.0,
                        target_smoothness=5.0
                    )
                    self.chase_camera.initialize()

                if thermal_camera_enabled:
                    self.thermal_camera = ThermalCamera(self._camera)
            else:
                if self.tag_prim is None:
                    tag_path = s.get("/ext/my_extension/tag_path") or "/World/tag"
                    self.tag_prim = self.stage.GetPrimAtPath(tag_path)

                self.udp_helper.send_tag_pos(self.tag_prim)

                drone_positions = self.udp_helper.send_all_drone_positions(self.all_drone_prims, self.all_velo_readers)

                trajectory_draw = s.get_as_bool("/ext/my_extension/trajectory_draw") or False
                if trajectory_draw:
                    if not self._trajectory_listener_started:
                        self.udp_helper.start_trajectory_listener()
                        self._trajectory_listener_started = True
                    all_trajectories = self.udp_helper.get_all_trajectories()
                self.traj_drawer.clear()

                thermal_camera_enabled = s.get_as_bool("/ext/my_extension/thermal_camera_enabled") or False
                if thermal_camera_enabled:
                    self.thermal_camera.step_and_send()

                for drone_name, drone_pos in drone_positions.items():
                    port = self.drone_port_map.get(drone_name)
                    if port is None:
                        carb.log_warn(f"No trajectory port mapped for drone: {drone_name}")
                        continue

                    if trajectory_draw:
                        traj_data = all_trajectories.get(port)
                        if traj_data is None:
                            continue
                        received_positions, yaws = traj_data
                        self.traj_drawer.draw_path_with_target_yaw(
                            received_positions, yaws, drone_pos
                        )

                chase_camera_enabled = s.get_as_bool("/ext/my_extension/chase_camera_enabled") or False
                if chase_camera_enabled and self.chase_camera is not None:
                    primary_pos = drone_positions.get("quadrotor")
                    if primary_pos is not None:
                        self.chase_camera.update(np.array([primary_pos[0], primary_pos[1], primary_pos[2]]), dt)

                drone_trail_enabled = s.get_as_bool("/ext/my_extension/drone_trail_enabled") or False
                if drone_trail_enabled:
                    if self.drone_trail is None:
                        self.drone_trail = DroneTrail(self.debug_draw, thickness=2.0)
                    primary_pos = drone_positions.get("quadrotor")
                    if primary_pos is not None:
                        self.drone_trail.add_position([primary_pos[0], primary_pos[1], primary_pos[2]])
                    self.drone_trail.draw()

                camera_mode_idx = s.get_as_int("/ext/my_extension/camera_mode") or 0
                camera_mode = 'front' if camera_mode_idx == 0 else 'ground'
                if camera_mode != self._last_camera_mode:
                    self._last_camera_mode = camera_mode
                    self._camera.set_camera_mode(camera_mode)

                camera_stabilized = s.get_as_bool("/ext/my_extension/camera_stabilized") or False
                if self._camera is not None:
                    if camera_stabilized and camera_mode == 'ground':
                        if self._dc_art == _dynamic_control.INVALID_HANDLE:
                            self._dc_art = self._dc.get_articulation("/World/quadrotor")
                        if self._dc_art != _dynamic_control.INVALID_HANDLE:
                            if self._dc_body == _dynamic_control.INVALID_HANDLE:
                                self._dc_body = self._dc.get_rigid_body("/World/quadrotor/body")
                            if self._dc_body != _dynamic_control.INVALID_HANDLE:
                                pose = self._dc.get_rigid_body_pose(self._dc_body)
                                pos = [pose.p.x, pose.p.y, pose.p.z]
                                dq = [pose.r.w, pose.r.x, pose.r.y, pose.r.z]
                                self._camera.update_stabilized(pos, dq)
                    elif camera_stabilized != self._was_stabilized:
                        self._camera.set_camera_mode(camera_mode)
                self._was_stabilized = camera_stabilized

        else:
            self._camera = None
            self.chase_camera = None
            self.drone_trail = None
            self.all_drone_prims = {}
            for reader in self.all_velo_readers.values():
                reader.reset()
            self.all_velo_readers = {}
            self._dc_art = _dynamic_control.INVALID_HANDLE
            self._dc_body = _dynamic_control.INVALID_HANDLE

    def on_update_cameras(self, event):
        s = carb.settings.get_settings()

        try:
            self.auto_record_manager.tick(event)
        except Exception as e:
            carb.log_warn(f"Auto record error. {e}")

        if self.is_simulation_active:
            if self._camera is not None:
                try:
                    image = self._camera.get_rgba()
                    if image is not None:
                        self.udp_helper.send_image(image)

                    if s.get_as_bool("/ext/my_extension/mines_detection") or False:
                        if self.last_time_mines_retrieved is None or time.time() - self.last_time_mines_retrieved > 10:
                            self._camera._load_mines_cache()
                            self.last_time_mines_retrieved = time.time()
                        # INFO: TIME UPDATE
                        if time.time() - self.last_mine_check_time > 0.1:
                        # if time.time() - self.last_mine_check_time > 1:
                            visible_mines = self._camera.get_visible_prims(name_filter="mine")
                            self.udp_helper.send_mines_data(visible_mines)
                            self.last_mine_check_time = time.time()
                except Exception as e:
                    carb.log_warn(f"Failed to retrieve and send image. {e}")

                try:
                    self.dataset_recorder.tick()
                except Exception as e:
                    carb.log_warn(f"Dataset recorder error. {e}")

    def _discover_drones(self):
        """
        Finds all prims starting with 'quadrotor' and populates
        the drone prim and velocity reader dictionaries.
        """
        if not self.stage:
            carb.log_error("Stage not set, cannot discover drones.")
            return
            
        carb.log_info("Discovering all drones...")
        
        # Clear any old data
        self.all_drone_prims = {}
        self.all_velo_readers = {}
        
        # Iterate over all prims in the stage
        for prim in Usd.PrimRange.Stage(self.stage):
            # Check if name matches pattern
            if prim.GetName().startswith("quadrotor"):
                
                # Check if it's an Xform (root of an articulation)
                if prim.GetTypeName() != "Xform":
                    continue
                    
                drone_name = prim.GetName() # e.g., "quadrotor" or "quadrotor_01"
                art_path = prim.GetPath().pathString # e.g., "/World/quadrotor" or "/quadrotor_01"
                body_path = f"{art_path}/body"
                body_prim = self.stage.GetPrimAtPath(body_path)
                
                if body_prim.IsValid():
                    carb.log_warn(f"Found drone: {drone_name}")
                    # Use the drone_name as the unique key
                    self.all_drone_prims[drone_name] = body_prim
                    self.all_velo_readers[drone_name] = VelocityReader(art_path, body_path)
                else:
                    carb.log_warn(f"Found prim {drone_name} but could not find body at {body_path}")
        
        carb.log_info(f"Discovered {len(self.all_drone_prims)} drones.")


    @property
    def is_simulation_active(self):
        """
        Returns True if either the normal Omniverse Play button is pressed,
        OR if the Pegasus global setting indicates the 30Hz loop is running.
        """
        # 1. Check normal timeline
        is_timeline_playing = self.timeline.is_playing()
        
        # 2. Check the global setting. 
        # Using "or False" ensures it doesn't crash if Pegasus hasn't created the setting yet.
        settings = carb.settings.get_settings()
        is_custom_loop_playing = settings.get_as_bool("/ext/pegasus/custom_30hz_active") or False
        
        # 3. Return True if EITHER is active
        return is_timeline_playing or is_custom_loop_playing

    def on_startup(self, _ext_id):
        self.timeline = omni.timeline.get_timeline_interface()
        self._camera = None
        self.follow_camera = None
        self.chase_camera = None
        self.drone_trail = None
        self.body_prim = None
        self.additional_camera_path = "/World/quadrotor/body/camera_pegasus"
        self.tag = None
        self.drone_xform = None
        self.tag_prim = None
        self.drone_pos = None
        self.drone_rigid_body = None
        self.yaws = None
        self.last_mine_check_time = None

        self.all_drone_prims = {}
        self.all_velo_readers = {}
        self.last_time_mines_retrieved = None

        self.drone_port_map = {
            "quadrotor": 5010,
            "quadrotor_01": 5011,
            "quadrotor_02": 5012,
        }

        self.debug_draw = _debug_draw.acquire_debug_draw_interface()

        self.traj_drawer = TrajectoryDrawer(self.debug_draw)

        self.velo_reader = VelocityReader("/World/quadrotor", "/World/quadrotor/body")
      
        self.udp_helper = UDPHelper(
            ip=UDP_IP,
            image_port=UDP_IMAGE_PORT,
            vel_port=UDP_VELOCITY_PORT,
            tag_port=UDP_TAG_PORT,
            pos_port=UDP_POSITION_PORT,
            traj_ports=UDP_TRAJECTORY_PORTS, # Pass the list
            depth_port=UDP_DEPTH_PORT,
            mines_port=UDP_MINES_PORT,
            second_image_port=UDP_SECOND_IMAGE_PORT
        )

        self.thermal_camera = None
        self._trajectory_listener_started = False
        self.dataset_recorder = DatasetRecorder()
        self.auto_record_manager = AutoRecordManager()

        self.received_positions = []
        self._last_camera_mode = None
        self._was_stabilized = False
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._dc_art = _dynamic_control.INVALID_HANDLE
        self._dc_body = _dynamic_control.INVALID_HANDLE

        update_stream = omni.kit.app.get_app().get_update_event_stream()

        self.sub_cameras = update_stream.create_subscription_to_pop(self.on_update_cameras, name="Cameras Subscription")

        self._physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_update)

        print("[my_company.my_python_extension] Extension startup")

    def on_shutdown(self):
        """This is called every time the extension is deactivated. It is used
        to clean up the extension state."""
        self.dataset_recorder.stop()
        self.auto_record_manager.stop()

