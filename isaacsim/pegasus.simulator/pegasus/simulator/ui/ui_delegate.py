"""
| File: ui_delegate.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Definition of the UiDelegate which is an abstraction layer betweeen the extension UI and code logic features
"""
import math
# External packages
import os
import asyncio
import threading
from scipy.spatial.transform import Rotation
# Omniverse extensions
import carb
import omni.ui as ui
import omni.usd
from pxr import Usd, Gf, UsdGeom
from pxr import UsdPhysics, PhysxSchema
import json
import time
import numpy as np
import random
import omni.physx
import omni.kit
# Extension Configurations
from pegasus.simulator.params import ROBOTS, TAGS, MISSIONS, OBSTACLES, SIMULATION_ENVIRONMENTS, BACKENDS, WORLD_SETTINGS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
# Vehicle Manager to spawn Vehicles
from pegasus.simulator.logic.backends import Backend, BackendConfig, PX4MavlinkBackend, PX4MavlinkBackendConfig, ArduPilotMavlinkBackend, BetaflightBackend, ArduPilotMavlinkBackendConfig, BetaflightBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.vehicle_manager import VehicleManager
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.tags import Tag
from pegasus.simulator.logic.obstacles import Obstacle, obstacles_parameters
from pegasus.simulator.logic.vehicle_manager import VehicleManager
# Import Mission Helper
from pegasus.simulator.ui.mission_helper import MissionHelper
from isaacsim.core.utils.semantics import upgrade_prim_semantics_to_labels, add_labels

import omni.timeline
import subprocess
import psutil
import os
import signal
import time

try:
    from pegasus.simulator.logic.backends import ROS2Backend
    ROS2_available = True
except ImportError:
    ROS2_available = False
    carb.log_warn("ROS2 backend not available. Please install the ROS2 extension to use this feature.")


class UIDelegate:
    """
    Object that will interface between the logic/dynamic simulation part of the extension and the Widget UI
    """

    def __init__(self):
        # The window that will be bound to this delegate
        self._window = None

        self.wind_enabled = False
        self.wind_speed = 0.0
        self.wind_direction = Gf.Vec3f(1.0, 0.0, 0.0)

        # Initialize Mission Helper
        self._mission_helper = MissionHelper()

        # Get an instance of the pegasus simulator
        self._pegasus_sim: PegasusInterface = PegasusInterface()

        # Attribute that holds the currently selected scene from the dropdown menu
        self._scene_dropdown: ui.AbstractItemModel = None
        self._scene_names = list(SIMULATION_ENVIRONMENTS.keys())

        # Selected latitude, longitude and altitude
        self._latitude_field: ui.AbstractValueModel = None
        self._latitude = PegasusInterface().latitude
        self._longitude_field: ui.AbstractValueModel = None
        self._longitude = PegasusInterface().longitude
        self._altitude_field: ui.AbstractValueModel = None
        self._altitude = PegasusInterface().altitude

        # Attribute that hold the currently selected vehicle from the dropdown menu
        self._vehicle_dropdown: ui.AbstractItemModel = None
        self._vehicles_names = list(ROBOTS.keys())

        # Get an instance of the vehicle manager
        self._vehicle_manager = VehicleManager()

        # Selected option for broadcasting the simulated vehicle (PX4+ROS2 or just ROS2)
        self._streaming_backend: str = BACKENDS['betaflight']

        # Selected value for the the id of the vehicle
        self._vehicle_id_field: ui.AbstractValueModel = None
        self._vehicle_id: int = 0

        self._vehicles_count_field: ui.AbstractValueModel = None
        self._vehicles_count: int = 3

        self._ports_path_field: ui.AbstractValueModel = None
        self._ports_path: str = "../config/betaflight_ports.json"

        # Attribute that will save the model for the px4-autostart checkbox
        self._px4_autostart_checkbox: ui.AbstractValueModel = None
        self._autostart_px4: bool = True

        # Atributes to store the path for the Px4 directory
        self._px4_directory_field: ui.AbstractValueModel = None
        self._px4_dir: str = PegasusInterface().px4_path

        # Atributes to store the PX4 airframe
        self._px4_airframe_field: ui.AbstractValueModel = None
        self._px4_airframe: str = self._pegasus_sim.px4_default_airframe

        # Attribute that will save the model for the ardupilot-autostart checkbox
        self._ardupilot_autostart_checkbox: ui.AbstractValueModel = None
        self._autostart_ardupilot: bool = True

        # Atributes to store the path for the ArduPilot directory
        self._ardupilot_directory_field: ui.AbstractValueModel = None
        self._ardupilot_dir: str = PegasusInterface().ardupilot_path

        # Atributes to store the ArduPilot airframe
        self._ardupilot_airframe_field: ui.AbstractValueModel = None
        self._ardupilot_airframe: str = self._pegasus_sim.ardupilot_default_airframe

        self._betaflight_connection_type_field: ui.AbstractValueModel = None
        self._betaflight_connection_type: str = "tcp"

        # Connection IP or serial port string field
        self._betaflight_connection_ip_field: ui.AbstractValueModel = None
        self._betaflight_connection_ip: str = "127.0.0.1"

        self._betaflight_motor_port_field: ui.AbstractValueModel = None
        self._betaflight_motor_port: int = 9002

        # Connection base port (for TCP/UDP)
        self._betaflight_state_port_field: ui.AbstractValueModel = None
        self._betaflight_state_port: int = 9003

        # Betaflight autostart checkbox
        self._betaflight_autostart_checkbox: ui.AbstractValueModel = None
        self._betaflight_autostart: bool = False
        self._apriltag_reset_checkbox: ui.AbstractValueModel = None
        self._apriltag_reset = True

        self._tag_id_field: ui.AbstractValueModel = None
        self._tag_id: int = 0

        self._tag_dropdown: ui.AbstractItemModel = None
        self._tags_names = list(TAGS.keys())
        
        # Local Tag Storage (also synced to helper)
        self._tag = None
        self.tags = []

        self.multirotors = []
        self.betaflight_ports = {}
        self.betaflight_ports_count = 0
        self.euler_angles = []
        self.reset_enable = False
        self.pos = [0,0,0]

        self._obstacle_dropdown: ui.AbstractItemModel = None
        self._obstacle_mission_dropdown: ui.AbstractItemModel = None
        self._obstacle_names = list(OBSTACLES.keys())

        self.formation_distance = 2



    def set_window_bind(self, window):
        self._window = window
        self._mission_helper._window = window

    def set_scene_dropdown(self, scene_dropdown_model: ui.AbstractItemModel):
        self._scene_dropdown = scene_dropdown_model

    def set_latitude_field(self, latitude_model: ui.AbstractValueModel):
        self._latitude_field = latitude_model

    def set_longitude_field(self, longitude_model: ui.AbstractValueModel):
        self._longitude_field = longitude_model

    def set_altitude_field(self, altitude_model: ui.AbstractValueModel):
        self._altitude_field = altitude_model

    def set_vehicle_dropdown(self, vehicle_dropdown_model: ui.AbstractItemModel):
        self._vehicle_dropdown = vehicle_dropdown_model

    def set_vehicle_id_field(self, vehicle_id_field: ui.AbstractValueModel):
        self._vehicle_id_field = vehicle_id_field

    def set_vehicles_count_field(self, vehicles_count_field: ui.AbstractValueModel):
        self._vehicles_count_field = vehicles_count_field

    def set_ports_path_field(self, ports_path_field: ui.AbstractValueModel):
        self._ports_path_field = ports_path_field

    def set_streaming_backend(self, backend: str = BACKENDS['betaflight']):
        self._streaming_backend = backend

    def set_px4_autostart_checkbox(self, checkbox_model:ui.AbstractValueModel):
        self._px4_autostart_checkbox = checkbox_model

    def set_px4_directory_field(self, directory_field_model: ui.AbstractValueModel):
        self._px4_directory_field = directory_field_model

    def set_px4_airframe_field(self, airframe_field_model: ui.AbstractValueModel):
        self._px4_airframe_field = airframe_field_model

    def set_ardupilot_autostart_checkbox(self, checkbox_model: ui.AbstractValueModel):
        self._ardupilot_autostart_checkbox = checkbox_model

    def set_ardupilot_directory_field(self, directory_field_model: ui.AbstractValueModel):
        self._ardupilot_directory_field = directory_field_model

    def set_ardupilot_airframe_field(self, airframe_field_model: ui.AbstractValueModel):
        self._ardupilot_airframe_field = airframe_field_model

    def set_betaflight_connection_type_field(self, connection_type_model: ui.AbstractValueModel):
        self._betaflight_connection_type_field = connection_type_model

    def set_betaflight_connection_ip_field(self, connection_ip_model: ui.AbstractValueModel):
        self._betaflight_connection_ip_field = connection_ip_model

    def set_betaflight_motor_port_field(self, motor_port_model: ui.AbstractValueModel):
        self._betaflight_motor_port_field = motor_port_model

    def set_betaflight_state_port_field(self, state_port_model: ui.AbstractValueModel):
        self._betaflight_state_port_field = state_port_model

    def set_betaflight_autostart_checkbox(self, checkbox_model: ui.AbstractValueModel):
        self._betaflight_autostart_checkbox = checkbox_model

    def set_apriltag_reset_checkbox(self, apriltag_checkbox_model:ui.AbstractValueModel):
        self._apriltag_reset_checkbox = apriltag_checkbox_model

    def set_tag_id_field(self, tag_id_field: ui.AbstractValueModel):
        self._tag_id_field = tag_id_field

    def set_tag_dropdown(self, tag_dropdown_model: ui.AbstractItemModel):
        self._tag_dropdown = tag_dropdown_model
    
    def set_spawn_dropdown(self, spawn_dropdown_model: ui.AbstractItemModel):
        self._mission_helper._spawn_dropdown = spawn_dropdown_model

    # --- Delegate UI Elements to MissionHelper ---
    def set_dvd_enabled_checkbox(self, dvd_enabled_checkbox_model: ui.AbstractValueModel):
        self._mission_helper._dvd_enabled_checkbox = dvd_enabled_checkbox_model

    def set_obstacles_file_path_field(self, obstacles_file_path_field: ui.AbstractValueModel):
        self._mission_helper._obstacles_file_path_field = obstacles_file_path_field

    def set_obstacles_mesh_name_field(self, obstacles_mesh_name_field: ui.AbstractValueModel):
        self._mission_helper._obstacles_mesh_name_field = obstacles_mesh_name_field

    def set_mission_dropdown(self, mission_dropdown_model: ui.AbstractItemModel):
        self._mission_helper._mission_dropdown = mission_dropdown_model

    def set_mission_time_field(self, mission_time_field: ui.AbstractValueModel):
        self._mission_helper._mission_time_field = mission_time_field

    def set_seed_field(self, seed_field: ui.AbstractValueModel):
        self._mission_helper._seed_field = seed_field

    def set_obstacle_dropdown(self, obstacle_dropdown_model: ui.AbstractItemModel):
        self._mission_helper._obstacle_dropdown = obstacle_dropdown_model

    def set_mission_obstacle_dropdown(self, mission_obstacle_dropdown_model: ui.AbstractItemModel):
        self._mission_helper._mission_obstacle_dropdown = mission_obstacle_dropdown_model

    def set_obstacles_count_field(self, obstacles_count_field: ui.AbstractValueModel):
        self._mission_helper._obstacles_count_field = obstacles_count_field

    def set_obstacles_seed_field(self, obstacles_seed_field: ui.AbstractValueModel):
        self._obstacles_seed_field = obstacles_seed_field

    def set_obstacles_distance_field(self, obstacles_distance_field: ui.AbstractValueModel):
        self._mission_helper._obstacles_distance_field = obstacles_distance_field

    def set_bushes_count_field(self, bushes_count_field: ui.AbstractValueModel):
        self._mission_helper._bushes_count_field = bushes_count_field

    def set_bushes_distance_field(self, bushes_distance_field: ui.AbstractValueModel):
        self._mission_helper._bushes_distance_field = bushes_distance_field

    def set_mission_count_field(self, mission_count_field: ui.AbstractValueModel):
        self._mission_helper._missions_count_field = mission_count_field

    def set_quit_enabled_checkbox(self, quit_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._quit_enabled_checkbox = quit_enabled_checkbox

    def set_formations_dropdown(self, formations_dropdown_model: ui.AbstractItemModel):
        self._formations_dropdown = formations_dropdown_model

    def set_restart_success_enabled_checkbox(self, _restart_success_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._restart_success_enabled_checkbox = _restart_success_enabled_checkbox
    
    def set_restart_collision_enabled_checkbox(self, restart_collision_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._restart_collision_enabled_checkbox = restart_collision_enabled_checkbox
    
    def set_multiple_mission_enabled_checkbox(self, multiple_mission_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._multiple_mission_enabled_checkbox = multiple_mission_enabled_checkbox
    
    def set_random_scales_enabled_checkbox(self, random_scales_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._random_scales_enabled_checkbox = random_scales_enabled_checkbox

    def set_random_rotations_enabled_checkbox(self, random_rotations_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._random_rotations_enabled_checkbox = random_rotations_enabled_checkbox

    def set_ultimate_mission_enabled_checkbox(self, ultimate_mission_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._ultimate_mission_enabled_checkbox = ultimate_mission_enabled_checkbox

    def set_alt_spawn_enabled_checkbox(self, alt_spawn_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._alt_spawn_enabled_checkbox = alt_spawn_enabled_checkbox
    
    def set_mines_count_field(self, mines_count_field: ui.AbstractValueModel):
        self._mission_helper._mines_count_field = mines_count_field

    def set_mines_distance_field(self, mines_distance_field: ui.AbstractValueModel):
        self._mission_helper._mines_distance_field = mines_distance_field

    def set_garbage_count_field(self, garbage_count_field: ui.AbstractValueModel):
        self._mission_helper._garbage_count_field = garbage_count_field

    def set_garbage_distance_field(self, garbage_distance_field: ui.AbstractValueModel):
        self._mission_helper._garbage_distance_field = garbage_distance_field
    
    def set_save_spawn_enabled_checkbox(self, save_spawn_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._save_spawn_enabled_checkbox = save_spawn_enabled_checkbox
    
    def set_spawn_mines_enabled_checkbox(self, spawn_mines_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._spawn_mines_enabled_checkbox = spawn_mines_enabled_checkbox

    def set_spawn_garbage_enabled_checkbox(self, spawn_garbage_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._spawn_garbage_enabled_checkbox = spawn_garbage_enabled_checkbox
    
    def set_img_map_enabled_checkbox(self, img_map_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._img_map_enabled_checkbox = img_map_enabled_checkbox
    
    def set_mine_type_status(self, mine_type, model):
        """Callback for checkbox change."""
        is_checked = model.get_value_as_bool()
        self._mission_helper.set_mine_type_enabled(mine_type, is_checked)

    def set_garbage_type_status(self, garbage_type, model):
        """Callback for checkbox change."""
        is_checked = model.get_value_as_bool()
        self._mission_helper.set_garbage_type_enabled(garbage_type, is_checked)
    
    def set_tree_type_status(self, tree_type, model):
        """Callback for checkbox change."""
        is_checked = model.get_value_as_bool()
        self._mission_helper.set_tree_type_enabled(tree_type, is_checked)
    
    def set_thermal_textures_enabled_checkbox(self, thermal_textures_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._thermal_textures_enabled_checkbox = thermal_textures_enabled_checkbox
    
    def set_thermal_layout_enabled_checkbox(self, thermal_layout_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._thermal_layout_enabled_checkbox = thermal_layout_enabled_checkbox

    def set_trees_wind_enabled_checkbox(self, trees_wind_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._trees_wind_enabled_checkbox = trees_wind_enabled_checkbox

    def set_wind_speed_field(self, wind_speed_field: ui.AbstractValueModel):
        self._mission_helper._wind_speed_field = wind_speed_field

    def set_wind_strength_field(self, wind_strength_field: ui.AbstractValueModel):
        self._mission_helper._wind_strength_field = wind_strength_field

    def set_wind_distance_field(self, wind_distance_field: ui.AbstractValueModel):
        self._mission_helper._wind_distance_field = wind_distance_field
    
    def set_remove_enabled_checkbox(self, remove_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._remove_enabled_checkbox = remove_enabled_checkbox
        
    def set_deactivation_enabled_checkbox(self, deactivation_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._deactivation_enabled_checkbox = deactivation_enabled_checkbox

    def set_drag_calibration_checkbox(self, drag_calibration_checkbox: ui.AbstractValueModel):
        self._mission_helper._drag_calibration_checkbox = drag_calibration_checkbox

    def set_drag_custom_value_checkbox(self, drag_custom_value_checkbox: ui.AbstractValueModel):
        self._mission_helper._drag_custom_value_checkbox = drag_custom_value_checkbox
    
    def set_custom_imu_enabled_checkbox(self, custom_imu_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._custom_imu_enabled_checkbox = custom_imu_enabled_checkbox
    
    def set_custom_motors_enabled_checkbox(self, custom_motors_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._custom_motors_enabled_checkbox = custom_motors_enabled_checkbox
    
    def set_custom_motors_path_field(self, custom_motors_path_field: ui.AbstractValueModel):
        self._mission_helper._custom_motors_path_field = custom_motors_path_field
    
    def set_custom_imu_path_field(self, custom_imu_path_field: ui.AbstractValueModel):
        self._mission_helper._custom_imu_path_field = custom_imu_path_field

    def set_custom_state_enabled_checkbox(self, custom_state_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._custom_state_enabled_checkbox = custom_state_enabled_checkbox

    def set_custom_state_path_field(self, custom_state_path_field: ui.AbstractValueModel):
        self._mission_helper._custom_state_path_field = custom_state_path_field

    # --- Parameter Tuning setters ---
    def set_tuning_original_log_path_field(self, field: ui.AbstractValueModel):
        self._mission_helper.set_tuning_original_log_path_field(field)

    def set_tuning_motor_log_path_field(self, field: ui.AbstractValueModel):
        self._mission_helper.set_tuning_motor_log_path_field(field)

    def set_tuning_max_iterations_field(self, field: ui.AbstractValueModel):
        self._mission_helper.set_tuning_max_iterations_field(field)

    def set_tuning_threshold_field(self, field: ui.AbstractValueModel):
        self._mission_helper.set_tuning_threshold_field(field)

    def set_tuning_learning_rate_field(self, field: ui.AbstractValueModel):
        self._mission_helper.set_tuning_learning_rate_field(field)

    def set_tuning_cooldown_field(self, field: ui.AbstractValueModel):
        self._mission_helper.set_tuning_cooldown_field(field)

    def set_tuning_use_dtw_checkbox(self, checkbox: ui.AbstractValueModel):
        self._mission_helper.set_tuning_use_dtw_checkbox(checkbox)

    def set_tuning_current_params_label(self, label):
        self._mission_helper.set_tuning_current_params_label(label)

    def set_tuning_best_params_label(self, label):
        self._mission_helper.set_tuning_best_params_label(label)

    def set_tuning_best_error_label(self, label):
        self._mission_helper.set_tuning_best_error_label(label)

    def set_tuning_live_params_label(self, label):
        self._mission_helper.set_tuning_live_params_label(label)

    def set_tuning_current_error_label(self, label):
        self._mission_helper.set_tuning_current_error_label(label)

    def set_tuning_perturbation_indicator_label(self, label):
        self._mission_helper.set_tuning_perturbation_indicator_label(label)

    def set_tuning_allow_negative_checkbox(self, checkbox: ui.AbstractValueModel):
        self._mission_helper.set_tuning_allow_negative_checkbox(checkbox)

    def set_tuning_enabled_checkbox(self, tuning_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._tuning_enabled_checkbox = tuning_enabled_checkbox

    def set_grass_semantics_enabled_checkbox(self, grass_semantics_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._grass_semantics_enabled_checkbox = grass_semantics_enabled_checkbox

    def set_trees_semantics_enabled_checkbox(self, trees_semantics_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._trees_semantics_enabled_checkbox = trees_semantics_enabled_checkbox
    
    def set_mines_semantics_enabled_checkbox(self, mines_semantics_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._mines_semantics_enabled_checkbox = mines_semantics_enabled_checkbox

    def set_vegetation_enabled_checkbox(self, vegetation_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._vegetation_enabled_checkbox = vegetation_enabled_checkbox

    def set_vegetation_type_combo(self, vegetation_type_combo):
        self._mission_helper._vegetation_type_combo = vegetation_type_combo

    def set_vegetation_random_rotation_checkbox(self, vegetation_random_rotation_checkbox: ui.AbstractValueModel):
        self._mission_helper._vegetation_random_rotation_checkbox = vegetation_random_rotation_checkbox

    def set_obstacles_reload_enabled_checkbox(self, obstacles_reload_enabled_checkbox: ui.AbstractValueModel):
        self._mission_helper._obstacles_reload_enabled_checkbox = obstacles_reload_enabled_checkbox

    def set_perturbation_eps_field(self, perturbation_eps_field: ui.AbstractValueModel):
        self._mission_helper._perturbation_eps_field = perturbation_eps_field

    def set_collision_restart_time_field(self, collision_restart_time_field: ui.AbstractValueModel):
        self._mission_helper._collision_restart_time_field = collision_restart_time_field
    """
    ---------------------------------------------------------------------
    Callbacks to handle user interaction with the extension widget window
    ---------------------------------------------------------------------
    """

    def on_load_ready_scene(self):
        asyncio.ensure_future(self._async_load_ready_scene())

    async def _async_load_ready_scene(self):
        await self._async_load_scene()
        await self._async_load_vehicle()
        await self._async_load_formation_tags()

    def on_load_scene(self):
        asyncio.ensure_future(self._async_load_scene())

    async def _async_load_scene(self):
        """
        Coroutine that loads the selected world into the simulator.
        """
        if self._scene_dropdown is not None:
            environemnt_index = self._scene_dropdown.get_item_value_model().as_int
            selected_world = self._scene_names[environemnt_index]
            self._pegasus_sim.set_world_settings(**WORLD_SETTINGS[self._streaming_backend])
            await self._pegasus_sim.load_environment_async(SIMULATION_ENVIRONMENTS[selected_world], force_clear=True)

    def on_set_new_global_coordinates(self):
        self._pegasus_sim.set_global_coordinates(
            self._latitude_field.get_value_as_float(),
            self._longitude_field.get_value_as_float(),
            self._altitude_field.get_value_as_float())

    def on_reset_global_coordinates(self):
        self._pegasus_sim.set_default_global_coordinates()
        self._latitude_field.set_value(self._pegasus_sim.latitude)
        self._longitude_field.set_value(self._pegasus_sim.longitude)
        self._altitude_field.set_value(self._pegasus_sim.altitude)

    def on_set_new_default_global_coordinates(self):
        self._pegasus_sim.set_new_default_global_coordinates(
            self._latitude_field.get_value_as_float(),
            self._longitude_field.get_value_as_float(),
            self._altitude_field.get_value_as_float()
        )

    def on_clear_scene(self):
        self._pegasus_sim.clear_scene()

    def on_load_vehicle(self):
        asyncio.ensure_future(self._async_load_vehicle())

    async def _async_load_vehicle(self):
        if hasattr(self._pegasus_sim.world, "_physics_context") == False or self.reset_enable:
            await self._pegasus_sim.world.initialize_simulation_context_async()
            self.reset_enable = False

        if self._vehicle_dropdown is not None and self._window is not None:
            vehicle_index = self._vehicle_dropdown.get_item_value_model().as_int
            selected_robot = self._vehicles_names[vehicle_index]
            self._vehicle_id = self._vehicle_id_field.get_value_as_int()
            pos, euler_angles = self._window.get_selected_vehicle_attitude()

            backend_config: BackendConfig = None
            backend: Backend = None

            if self._streaming_backend == BACKENDS["px4"]:
                px4_autostart = self._px4_autostart_checkbox.get_value_as_bool()
                px4_path = os.path.expanduser(self._px4_directory_field.get_value_as_string())
                px4_airframe = self._px4_airframe_field.get_value_as_string()

                backend_config = PX4MavlinkBackendConfig({
                    "vehicle_id": self._vehicle_id,
                    "px4_autolaunch": px4_autostart,
                    "px4_dir": px4_path,
                    "px4_vehicle_model": px4_airframe
                })
                backend = PX4MavlinkBackend(config=backend_config)
                carb.log_warn("PX4 backend selected.")

            elif self._streaming_backend == BACKENDS["ardupilot"]:
                ardupilot_autostart = self._ardupilot_autostart_checkbox.get_value_as_bool()
                ardupilot_path = os.path.expanduser(self._ardupilot_directory_field.get_value_as_string())
                ardupilot_airframe = self._ardupilot_airframe_field.get_value_as_string()

                backend_config = ArduPilotMavlinkBackendConfig({
                    "vehicle_id": self._vehicle_id,
                    "ardupilot_autolaunch": ardupilot_autostart,
                    "ardupilot_dir": ardupilot_path,
                    "ardupilot_vehicle_model": ardupilot_airframe
                })
                backend = ArduPilotMavlinkBackend(config=backend_config)
                carb.log_warn("Ardupilot backend selected.")

            elif self._streaming_backend == BACKENDS["ros2"]:
                if ROS2_available:
                    backend = ROS2Backend(vehicle_id=self._vehicle_id, config={
                        "namespace": 'drone',
                        "pub_sensors": True,
                        "pub_graphical_sensors": True,
                        "pub_state": True,
                        "pub_tf": False,
                        "sub_control": True}
                        )
                    carb.log_warn("ROS2 backend selected.")
                else:
                    carb.log_warn("ROS2 not available. Please run Isaac Sim with ROS 2 extension correctly enabled.")
                    return
            elif self._streaming_backend == BACKENDS["betaflight"]:
                connection_ip = self._betaflight_connection_ip_field.get_value_as_string()
                motor_port = int(self._betaflight_motor_port_field.get_value_as_string())
                state_port = int(self._betaflight_state_port_field.get_value_as_string())

                backend_config = BetaflightBackendConfig({
                    "vehicle_id": self._vehicle_id,
                    "motor_port": motor_port,
                    "state_port": state_port,
                    "connection_ip": connection_ip,
                })

                backend = BetaflightBackend(config=backend_config)
                carb.log_warn("Betaflight backend selected.")

            config_multirotor = MultirotorConfig()
            config_multirotor.backends = [backend]

            multirotor = Multirotor(
                f"/World/quadrotor",
                ROBOTS[selected_robot],
                self._vehicle_id,
                pos,
                Rotation.from_euler("XYZ", euler_angles, degrees=True).as_quat(),
                config=config_multirotor,
            )

            self.multirotors.append(multirotor)
            # Sync with helper
            self._mission_helper.multirotors.append(multirotor)
            self._mission_helper._multirotor = multirotor

            carb.log_info("Spawned the robot: " + selected_robot + " using the Pegasus Simulator UI")
        else:
            carb.log_error("Could not spawn the robot using the Pegasus Simulator UI")

    def generate_positions(self, pos, spacing, n):
        start_x, start_y, start_z = pos
        dx, dy = spacing
        positions = []
        for i in range(n):
            step = (i + 1) // 2 
            sign = 1 if i % 2 == 1 else -1 

            if i == 0:
                positions.append((start_x, start_y, start_z))
            else:
                positions.append((start_x + sign * step * dx,
                                  start_y + sign * step * dy,
                                  start_z))
        return positions
    
    async def async_load_vehicles(self, positions):
        for i in range(self._vehicles_count):
            if hasattr(self._pegasus_sim.world, "_physics_context") == False:
                await self._pegasus_sim.world.initialize_simulation_context_async()

            if self._vehicle_dropdown is not None and self._window is not None:
                vehicle_index = self._vehicle_dropdown.get_item_value_model().as_int
                selected_robot = self._vehicles_names[vehicle_index]
                self._vehicle_id = i

                backend_config: BackendConfig = None
                backend: Backend = None

                if self._streaming_backend != BACKENDS["betaflight"]:
                    carb.log_error("Multiple vehicles load supported only for Betaflight backend. Change backend and try again.")
                    return

                connection_ip = self._betaflight_connection_ip_field.get_value_as_string()
                
                backend_config = BetaflightBackendConfig({
                    "vehicle_id": self._vehicle_id,
                    "motor_port": self.betaflight_ports[i]["motor"],
                    "state_port": self.betaflight_ports[i]["state"],
                    "connection_ip": connection_ip,
                })

                backend = BetaflightBackend(config=backend_config)
                carb.log_warn("Betaflight backend selected.")

                config_multirotor = MultirotorConfig()
                config_multirotor.backends = [backend]
                
                multirotor = Multirotor(
                    "/World/quadrotor",
                    ROBOTS[selected_robot],
                    self._vehicle_id,
                    positions[i],
                    Rotation.from_euler("XYZ", self.euler_angles, degrees=True).as_quat(),
                    config=config_multirotor,
                )

                self.multirotors.append(multirotor)
                # Sync with helper
                self._mission_helper.multirotors.append(multirotor)
                if i == 0:
                    self._mission_helper._multirotor = multirotor

                carb.log_info("Spawned the robot: " + selected_robot + " using the Pegasus Simulator UI")
            else:
                carb.log_error("Could not spawn the robot using the Pegasus Simulator UI")


    def on_load_vehicles(self):
        self._on_setup_vehicles()
        spacing = self._window.get_vehicles_spacing()
        positions = self.generate_positions(self.pos, spacing, self.betaflight_ports_count)
        asyncio.ensure_future(self.async_load_vehicles(positions))

    def get_formation_positions(self):
        formation = self._formations_dropdown.get_item_value_model().as_int
        if formation == 0: # Triangle
            return [[0, 0, 0.2], 
                    [-self.formation_distance, self.formation_distance, 0.2], 
                    [-self.formation_distance, -self.formation_distance, 0.2]]

    def on_load_formation(self):
        self._on_setup_vehicles()
        positions = self.get_formation_positions()
        asyncio.ensure_future(self.async_load_vehicles(positions))

    def _on_setup_vehicles(self):
        self._vehicles_count = self._vehicles_count_field.get_value_as_int()
        carb.log_warn("Vehicle count is set to {}".format(self._vehicles_count))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../config/betaflight_ports.json")

        with open(os.path.normpath(file_path), "r") as f:
            self.betaflight_ports = json.load(f)

        self.betaflight_ports_count = len(self.betaflight_ports)
        if self.betaflight_ports_count < self._vehicles_count:
            carb.log_warn(f"Vehicle count ({self._vehicles_count}) is greater than betaflight instances count ({self.betaflight_ports_count}).)")

        self.pos, self.euler_angles = self._window.get_selected_vehicle_attitude()

    def on_load_tag(self):
        async def async_load_tag():
            if self._tag_dropdown is not None and self._window is not None:
                tag_index = self._tag_dropdown.get_item_value_model().as_int
                selected_tag = self._tags_names[tag_index]
                self._tag_id = self._tag_id_field.get_value_as_int()
                pos, euler_angles = self._window.get_selected_tag_attitude()

                carb.log_warn(f"pos: {pos}")

                if self._tag is None:
                    self._tag = Tag(
                        "/World/tag",
                        TAGS[selected_tag],
                        pos,
                        Rotation.from_euler("XYZ", euler_angles, degrees=True).as_quat(),
                    )
                else:
                    self._tag.set_pos_and_orient(pos, euler_angles)
                
                # Sync with helper
                self._mission_helper._tag = self._tag

                carb.log_info("Spawned the tag: " + selected_tag + " using the Pegasus Simulator UI")
            else:
                carb.log_error("Could not spawn the tag using the Pegasus Simulator UI")

        asyncio.ensure_future(async_load_tag())

    def get_formation_tags_positions(self):
        pos, euler_angles = self._window.get_selected_tag_attitude()
        positions = [pos, [pos[0] - 2, pos[1] + 2, pos[2]], [pos[0] - 2, pos[1] - 2, pos[2]]]
        angles = [euler_angles, euler_angles, euler_angles]
        return (positions, angles)

    def on_load_formation_tags(self):
        asyncio.ensure_future(self._async_load_formation_tags())

    async def _async_load_formation_tags(self):
        if self._tag_dropdown is not None and self._window is not None:
            tag_index = self._tag_dropdown.get_item_value_model().as_int
            selected_tag = self._tags_names[tag_index]
            self._tag_id = self._tag_id_field.get_value_as_int()
            positions, euler_angles = self.get_formation_tags_positions()

            if self._tag is None:
                for i in range(len(positions)):
                    tag = Tag(
                        "/World/tag",
                        TAGS[selected_tag],
                        positions[i],
                        Rotation.from_euler("XYZ", euler_angles, degrees=True).as_quat(),
                    )
                    upgrade_prim_semantics_to_labels(tag._prim)
                    add_labels(tag._prim, ["tag"])
                    self.tags.append(tag)

                self._tag = self.tags[0]

            else:
                for i, tag in enumerate(self.tags):
                    tag.set_pos_and_orient(positions[i], euler_angles)

            # Sync with helper
            self._mission_helper._tag = self._tag
            self._mission_helper.tags = self.tags

            carb.log_info("Spawned the tag: " + selected_tag + " using the Pegasus Simulator UI")
            carb.log_warn("Formation tags were spawned successfully!")
        else:
            carb.log_error("Could not spawn the tag using the Pegasus Simulator UI")

    def on_load_obstacle(self):
        self._mission_helper.on_load_obstacle()

    def _apply_wind_to_bodies(self):
        """
        Iterates through the stage to find rigid bodies (drones) and applies 
        a constant force to simulate wind, bypassing the removed ForceField API.
        """
        # Local imports to prevent startup crashes
        
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
            
        # Calculate the final force vector based on speed and direction
        force_vec = (
            self.wind_direction.GetNormalized() * self.wind_speed 
            if self.wind_enabled else Gf.Vec3f(0.0, 0.0, 0.0)
        )

        # Traverse the stage to find dynamic rigid bodies (e.g., drone chassis)
        for prim in stage.Traverse():
            # Apply only to objects that have physics enabled (Rigid Bodies)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                
                # Apply the PhysxForceAPI if it doesn't already exist on this body
                if not prim.HasAPI(PhysxSchema.PhysxForceAPI):
                    PhysxSchema.PhysxForceAPI.Apply(prim)
                    
                force_api = PhysxSchema.PhysxForceAPI(prim)
                
                # Update the force vector and enable/disable it
                force_api.CreateForceAttr().Set(force_vec)
                force_api.CreateForceEnabledAttr().Set(self.wind_enabled)
                
                # Note: By default, this applies a spatial force (mode="force"). 
                # If you want the wind to push all drones equally regardless of their mass,
                # you can uncomment the line below to change it to an acceleration field.
                # force_api.CreateModeAttr().Set("acceleration") 
    
    def on_wind_enabled_changed(self, enabled: bool):
        self.wind_enabled = enabled
        self._apply_wind_to_bodies()

    def on_wind_speed_changed(self, speed: float):
        self.wind_speed = speed
        if self.wind_enabled:
            self._apply_wind_to_bodies()

    def on_wind_direction_changed(self, x: float, y: float, z: float):
        self.wind_direction = Gf.Vec3f(x, y, z)
        if self.wind_enabled:
            self._apply_wind_to_bodies()

    def on_variant_changed(self, model, item):
        """This fires every time the user clicks a new option in the ComboBox"""
        prim_path = "/World/layout/mines"
        variant_names = ["default", "thermal"]
        # Get the integer index of what the user selected (e.g., 1 for midday)
        index = model.get_item_value_model().as_int
        selected_variant = variant_names[index]
        
        # Grab the current USD stage
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
            
        # Find the landmine prim
        prim = stage.GetPrimAtPath(prim_path)
        
        if prim.IsValid():
            # Grab the Variant Set we made in Step 1
            variant_set = prim.GetVariantSets().GetVariantSet("texture")
            
            # Switch the texture!
            variant_set.SetVariantSelection(selected_variant)
            print(f"Swapped thermal texture to: {selected_variant}")
        else:
            print(f"Error: Could not find prim at {prim_path}")
   
    def on_set_viewport_camera(self):
        carb.log_warn("The viewport camera pose has been adjusted")
        if self._window:
            camera_position, camera_target = self._window.get_selected_camera_pos()
            if camera_position is not None and camera_target is not None:
                self._pegasus_sim.set_viewport_camera(eye=camera_position, target=camera_target)

    def on_set_new_default_px4_path(self):
        carb.log_warn("A new default PX4 Path will be set for the extension.")
        path = self._px4_directory_field.get_value_as_string()
        self._pegasus_sim.set_px4_path(path)

    def on_reset_px4_path(self):
        carb.log_warn("Reseting the path to the default one")
        self._px4_directory_field.set_value(self._pegasus_sim.px4_path)

    def on_set_new_default_ardupilot_path(self):
        carb.log_warn("A new default ArduPilot Path will be set for the extension.")
        path = self._ardupilot_directory_field.get_value_as_string()
        self._pegasus_sim.set_ardupilot_path(path)

    def on_reset_ardupilot_path(self):
        carb.log_warn("Reseting the path to the default one")
        self._ardupilot_directory_field.set_value(self._pegasus_sim.ardupilot_path)

    # --- Delegate methods to MissionHelper ---
    def on_parse_obstacles(self):
        self._mission_helper.on_parse_obstacles()

    def reinitialize_simulation(self):
        # This was commented out in the helper, check if needed
        pass

    def mission_cycle(self):
        self._mission_helper.mission_cycle()

    def on_start_mission(self):
        self._mission_helper.on_start_mission()

    def on_stop_mission(self):
        self._mission_helper.on_stop_mission()

    def on_tuning_start(self):
        self._mission_helper.on_tuning_start()

    def on_tuning_stop(self):
        self._mission_helper.on_tuning_stop()

    def on_tuning_apply_best(self):
        self._mission_helper.on_tuning_apply_best()

    def set_mission_status_label(self, label):
        self._mission_helper.set_mission_status_label(label)

    def set_mission_settings_label(self, label):
        self._mission_helper.set_mission_settings_label(label)

    def set_current_seed_label(self, label):
        self._mission_helper.set_current_seed_label(label)

    def set_current_mission_count_label(self, label):
        self._mission_helper.set_current_mission_count_label(label)

    def update_mission_status(self, running: bool):
        self._mission_helper.update_mission_status(running)

    def update_current_mission_count(self, running: bool):
        self._mission_helper.update_current_mission_count(running)

    def update_current_seed(self):
        self._mission_helper.update_current_seed()

    def check_mission_end(self):
        self._mission_helper.check_mission_end()

    def on_load_obstacles(self):
        self._mission_helper.on_load_obstacles()

    def on_destroy_obstacles(self):
        self._mission_helper.on_destroy_obstacles()

    def on_save_mission_settings(self, seed, mission_type, obstacles_count, obstacles_distance, mission_count, mines_count, mines_distance, garbage_count=0, garbage_distance=0):
        self._mission_helper.on_save_mission_settings(seed, mission_type, obstacles_count, obstacles_distance, mission_count, mines_count, mines_distance, garbage_count, garbage_distance)
    
    # --- Moved to Helper but potentially called from UI via delegate wrapper ---
    def on_reset_tag(self):
        self._mission_helper.on_reset_tag()
    
    def on_pause_tag(self):
        self._mission_helper.on_pause_tag()

    def on_start_tag(self):
        self._mission_helper.on_start_tag()

    def move_cycle(self):
        self._mission_helper.move_cycle()

    def on_reset_vehicles(self):
        self._mission_helper.on_reset_vehicles()

    def on_reset_vehicle(self, multirotor):
        self._mission_helper.on_reset_vehicle(multirotor)
    
    def set_current_drag_value_label(self, label):
        self._mission_helper.set_current_drag_value_label(label)