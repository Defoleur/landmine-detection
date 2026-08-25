import json
import omni.ui as ui
import omni.usd
from pxr import Usd, Gf, UsdGeom, UsdShade, Sdf, UsdLux, UsdPhysics
import os
import asyncio
import time
import random
import numpy as np
import carb
from datetime import datetime
import omni.kit.commands
import omni.timeline

from isaacsim.core.utils.semantics import upgrade_prim_semantics_to_labels, add_labels
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.tuning import ParameterTuner

from pegasus.simulator.logic.obstacles import Obstacle, obstacles_parameters
from pegasus.simulator.params import ROBOTS, TAGS, MISSIONS, OBSTACLES, GARBAGE, SIMULATION_ENVIRONMENTS, BACKENDS, WORLD_SETTINGS, SPAWN_MAPS
from pegasus.simulator.ui.image_map_helper import ImageMapHelper
from pegasus.simulator.helpers.environment_manager import EnvironmentManager

from omni.physx import get_physx_simulation_interface

from .mission.mission_physics import MissionPhysics
from .mission.mission_obstacles import MissionObstacles
from .mission.mission_mines import MissionMines
from .mission.mission_garbage import MissionGarbage
from .mission.mission_layout import MissionLayout
from .mission.mission_ui import MissionUI
from .mission.mission_tags_vehicles import MissionTagsVehicles
from .mission.mission_settings import MissionSettings


class MissionHelper:
    def __init__(self):
        # --- References to External Objects ---
        self._window = None 
        self.stage = None
        self._tag = None 
        self.multirotors = [] 
        self._multirotor = None 
        self.tags = [] 
        self.body_prims = [None, None, None] 

        # --- Mission State ---
        self.mission_time = None
        self.selected_mission = None
        self.current_mission_count = 1
        self.mission_finished = [False, False, False]
        self.move_cycle_state = "STOPPED"
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Mission Settings (UI Models & Defaults) ---
        self._mission_dropdown: ui.AbstractItemModel = None
        self._mission_names = MISSIONS

        self._mission_time_field: ui.AbstractValueModel = None
        self._mission_cycle_time: int = 400

        self._missions_count_field: ui.AbstractValueModel = None
        self._missions_count: int = 100

        self._seed_field: ui.AbstractValueModel = None
        self._seed: int = 1111

        self._mission_settings_path: str = "../config/mission_settings.json"

        # --- UI Status Labels ---
        self._mission_status_label = None
        self._mission_settings_label = None
        self._current_seed_label = None
        self._current_mission_count_label = None
        self._current_drag_value_label = None

        # --- Obstacles Configuration ---
        self._obstacles_file_path_field: ui.AbstractValueModel = None
        self._obstacles_path: str = "../config/obstacles_positions.json"
        
        self._mines_path: str = "../config/mines_positions.json"

        self._garbage_path: str = "../config/garbage_positions.json"

        self._obstacles_mesh_name_field: ui.AbstractValueModel = None
        self._mesh_name: str = "obstacle"

        self._obstacle_dropdown: ui.AbstractItemModel = None
        self._mission_obstacle_dropdown: ui.AbstractItemModel = None
        self._obstacle_names = list(OBSTACLES.keys())

        self._obstacles_count_field: ui.AbstractValueModel = None
        self._obstacles_count: int = 200

        self._obstacles_distance_field: ui.AbstractValueModel = None
        self._obstacles_distance: int = 5

        self._bushes_count_field: ui.AbstractValueModel = None
        self._bushes_count: int = 300

        self._bushes_distance_field: ui.AbstractValueModel = None
        self._bushes_distance: int = 2

        self._obstacles_parameters = obstacles_parameters
        
        self._random_scales_enabled_checkbox: ui.AbstractValueModel = None
        self._random_scales_enabled = True

        self._random_rotations_enabled_checkbox: ui.AbstractValueModel = None
        self._random_rotations_enabled = True

        # --- Obstacles State ---
        self._obstacles = []
        self._obstacles_positions = []

        self._mines = []
        self._mines_positions = []

        self._garbage = []
        self._garbage_positions = []

        self._vegetation = []

        self.bush_checkbox_models = {}

        # --- Reset & Collision Logic ---
        self.reset_time = None
        self.time_to_reset = 1
        self.second_reset_available = False
        self.target_reached_tolerance = 0.7

        self.last_collision_check_time = None
        self.collision_cycle_time = 0

        self._collision_restart_time_field: ui.AbstractValueModel = None
        self._collision_restart_time = 20

        # --- Checkboxes / Flags ---
        self._multiple_mission_enabled_checkbox: ui.AbstractValueModel = None
        self._multiple_mission_enabled = True

        self._ultimate_mission_enabled_checkbox: ui.AbstractValueModel = None
        self._ultimate_mission_enabled = False

        self._restart_success_enabled_checkbox: ui.AbstractValueModel = None
        self._restart_success_enabled = True

        self._restart_collision_enabled_checkbox: ui.AbstractValueModel = None
        self._restart_collision_enabled = True

        self._quit_enabled_checkbox: ui.AbstractValueModel = None
        self._quit_enabled = False
        
        self._dvd_enabled_checkbox: ui.AbstractValueModel = None

        self._alt_spawn_enabled_checkbox: ui.AbstractValueModel = None
        self._alt_spawn_enabled = True

        self._save_spawn_enabled_checkbox: ui.AbstractValueModel = None
        self._save_spawn_enabled = False

        self._spawn_mines_enabled_checkbox: ui.AbstractValueModel = None
        self._spawn_mines_enabled = True

        self._spawn_garbage_enabled_checkbox: ui.AbstractValueModel = None
        self._spawn_garbage_enabled = True

        self._road_enabled = False

        self._img_map_enabled_checkbox: ui.AbstractValueModel = None
        self._img_map_enabled = True

        self._thermal_textures_enabled_checkbox: ui.AbstractValueModel = None
        self._thermal_textures_enabled = False

        self._thermal_layout_enabled_checkbox: ui.AbstractValueModel = None
        self._thermal_layout_enabled = False 

        self._obstacles_reload_enabled_checkbox: ui.AbstractValueModel = None
        self._obstacles_reload_enabled = False

        self._remove_enabled_checkbox: ui.AbstractValueModel = None
        self._remove_enabled = False
                
        self._deactivation_enabled_checkbox: ui.AbstractValueModel = None
        self._deactivation_enabled = False

        self._drag_calibration_checkbox: ui.AbstractValueModel = None
        self._drag_calibration = False

        self._drag_custom_value_checkbox: ui.AbstractValueModel = None
        self._drag_custom_value = False

        self._custom_imu_enabled_checkbox: ui.AbstractValueModel = None
        self._custom_imu_enabled = False

        self._custom_motors_enabled_checkbox: ui.AbstractValueModel = None
        self._custom_motors_enabled = False

        self._last_wind_update = 0.0
        self.wind_update_rate = 1.0 / 20.0

        # --- Trees Wind Parameters ---
        self._trees_wind_enabled_checkbox: ui.AbstractValueModel = None
        self._trees_wind_enabled = False

        self._wind_speed_field: ui.AbstractValueModel = None
        self._wind_speed = 1.5

        self._wind_strength_field: ui.AbstractValueModel = None
        self._wind_strength = 2.0

        self._wind_distance_field: ui.AbstractValueModel = None
        self._wind_distance = 40.0

        self._custom_motors_path_field: ui.AbstractValueModel = None
        self._custom_motors_path = "/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/original_logs/motor_replay_log_20260529_114316.csv"

        self._custom_imu_path_field: ui.AbstractValueModel = None
        self._custom_imu_path = '/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/live_imu_logs/live_imu_record_20260506_114544.csv'

        self._custom_state_enabled_checkbox: ui.AbstractValueModel = None
        self._custom_state_enabled = False
        self._custom_state_path_field: ui.AbstractValueModel = None
        self._custom_state_path = "/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/state_logs/vehicle_state_20260529_114316.csv"

        # --- Parameter Tuning ---
        self._tuning_enabled_checkbox: ui.AbstractValueModel = None
        self._tuning_enabled = False

        self._tuning_allow_negative_checkbox: ui.AbstractValueModel = None
        self._tuning_allow_negative = False

        self._tuning_original_log_path_field: ui.AbstractValueModel = None
        self._tuning_original_log_path = "/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/state_logs/vehicle_state_20260529_114316.csv"

        self._tuning_motor_log_path_field: ui.AbstractValueModel = None
        self._tuning_motor_log_path = "/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/original_logs/motor_replay_log_20260529_114316.csv"

        self._tuning_max_iterations_field: ui.AbstractValueModel = None
        self._tuning_max_iterations = 100

        self._tuning_threshold_field: ui.AbstractValueModel = None
        self._tuning_threshold = 0.001

        self._tuning_learning_rate_field: ui.AbstractValueModel = None
        self._tuning_learning_rate = 0.01

        self._perturbation_eps_field: ui.AbstractValueModel = None
        self._perturbation_eps = 0.01

        self._tuning_cooldown_field: ui.AbstractValueModel = None
        self._tuning_cooldown = 2.0

        self._tuning_use_dtw_checkbox: ui.AbstractValueModel = None
        self._tuning_use_dtw = False

        self._tuning_current_params_label = None
        self._tuning_best_params_label = None
        self._tuning_best_error_label = None
        self._tuning_live_params_label = None
        self._tuning_current_error_label = None
        self._tuning_perturbation_indicator_label = None

        self.tuner = None

        self._grass_semantics_enabled_checkbox: ui.AbstractValueModel = None
        self._grass_semantics_enabled = True

        self._trees_semantics_enabled_checkbox: ui.AbstractValueModel = None
        self._trees_semantics_enabled = True

        self._mines_semantics_enabled_checkbox: ui.AbstractValueModel = None
        self._mines_semantics_enabled = True

        self._vegetation_enabled_checkbox: ui.AbstractValueModel = None
        self._vegetation_enabled = False

        self._vegetation_random_rotation_checkbox: ui.AbstractValueModel = None
        self._vegetation_random_rotation = False

        self._vegetation_type_combo = None
        self._current_season = None

        self.fade_margin = 10.0

        self.ultimate_mission_map_setup = [
            {
                # INFO: EMPTY SHORT MAP
                "tag_position" : [30, 0, 5],
                "obstacles_count" : 0,
                "obstacles_distances": 7,
                "obstacles_spawn_bounds" : [[15,-25,0], [90,25,0]],
            },
            {
                # INFO: LOW DENSE MEDIUM MAP
                "tag_position" : [105, 0, 5],
                "obstacles_count" : 30,
                "obstacles_distances": 7,
                "obstacles_spawn_bounds" : [[15,-25,0], [90,25,0]],
            },
            {
                # INFO: HIGH DENSE MEDIUM MAP
                "tag_position" : [105, 0, 5],
                "obstacles_count" : 60,
                "obstacles_distances": 5,
                "obstacles_spawn_bounds" : [[15,-25,0], [90,25,0]],
            },
            {
                # INFO: LOW DENSE LARGE MAP
                "tag_position" : [205, 0, 5],
                "obstacles_count" : 60,
                "obstacles_distances": 7,
                "obstacles_spawn_bounds" : [[15,-25,0], [190,25,0]],
            },
            {
                # INFO: HIGH DENSE LARGE MAP
                "tag_position" : [205, 0, 5],
                "obstacles_count" : 90,
                "obstacles_distances": 5,
                "obstacles_spawn_bounds" : [[15,-25,0], [190,25,0]],
            },
        ]

        self.ultimate_mission_formations = [
            {
                # INFO: TRIANGLE DEFAULT
                "formation_type": "triangle",
                "formation_distance" : 2,
            },
            {
                # INFO: TRIANGLE LARGE
                "formation_type": "triangle",
                "formation_distance" : 4,
            },
            {
                # INFO: KEY?
                "formation_type": "key",
                "formation_distance" : 3,
            },
            {
                # INFO: ROW
                "formation_type": "row",
                "formation_distance" : 3,
            }
        ]

        self.ultimate_mission_setup = []

        for map_setup in self.ultimate_mission_map_setup:
            for formation in self.ultimate_mission_formations:
                mission = {
                    "map_setup" : map_setup,
                    "formation": formation
                }

                self.ultimate_mission_setup.append(mission)


        self.ultimate_mission_count = 10
        self.ultimate_mission_maps = len(self.ultimate_mission_setup)

        self.current_ultimate_mission_count = 0
        self.current_ultimate_mission_map = 0

        self.current_ultimate_mission = None

        self.ultimate_mission_check = False

        self.default_formation_distance = 2

        self.default_formation_mapper = [
            [0.0, 0.0, 0.0],
            [self.default_formation_distance, -self.default_formation_distance, 0.0],
            [self.default_formation_distance, self.default_formation_distance, 0.0]
        ]

        self.generated_formation = [[0.0, 0.0, 0.0] for i in range(3)]
        self.current_drone_formation = [0.0, 0.0, 0.0]
        self.current_tag_formation = [15.0, 0.0, 5.0]

        self.maps_folder = os.path.join(self.script_dir, "../maps")
        self.maps_debug_folder = self.maps_folder + "/debug"

        self.trees_map_image = None
        self.mines_map_image = None
        
        self.trees_map = None
        self.mines_map = None

        self._spawn_dropdown: ui.AbstractItemModel = None
        self._spawn_names = list(SPAWN_MAPS.keys())

        self.trees_probability_map = f"{self.maps_folder}/trees_map_curved.png"
        self.mines_probability_map = f"{self.maps_folder}/mines_map_curved.png"
        self.empty_probability_map = f"{self.maps_folder}/empty_map.png"

        self.map_bounds = [[5, -100], [205, 100]]

        self._mines_count_field: ui.AbstractValueModel = None
        self._mines_count: int = 50

        self._mines_distance_field: ui.AbstractValueModel = None
        self._mines_distance: int = 3

        self._garbage_count_field: ui.AbstractValueModel = None
        self._garbage_count: int = 50

        self._garbage_distance_field: ui.AbstractValueModel = None
        self._garbage_distance: int = 3

        self._available_mine_types = [k for k in OBSTACLES.keys() if "Mine" in k]
        self._selected_mine_types = self._available_mine_types.copy()

        self._available_garbage_types = list(GARBAGE.keys())
        self._selected_garbage_types = self._available_garbage_types.copy()

        self._available_tree_types = [k for k in OBSTACLES.keys() if "Tree" in k]
        self._selected_tree_types = self._available_tree_types.copy()

        self.grass_names = ["cube_0_", "cube_013", "cube_1_"]

        self.drag_test_list = [[i / 10, j / 10, 0.0] for i in range(11) for j in range(11)]
        self.drag_test_index = 0

        self._custom_loop_running = False
        self._fixed_dt = 1.0 / 250.0
        self._accumulator = 0.0
        self._sim_time = 0.0
        self._physx_interface = omni.physx.get_physx_interface()

        # Motor CSV replay state
        self._motor_csv_data = []
        self._motor_csv_index = 0
        self._motor_csv_loaded = False
        self._motor_replay_mode = False
        self._motor_replay_start_time_ns = 0
        self._motor_csv_first_time_ns = 0
        self._motor_replay_requested = False
        self._motor_replay_log = []

        self._motor_started = False

        update_stream = omni.kit.app.get_app().get_update_event_stream()
        self._app_update_sub = update_stream.create_subscription_to_pop(
            self._on_app_update, name="Fixed_Step_Accumulator"
        )

        self.timeline = omni.timeline.get_timeline_interface()
        self.prev_time = None

        self.motor_record_index = 0
        self.motor_record_armed = False

        self.use_physx_sim = False

        self.is_first_start = True

        # --- Season Schedule ---
        # self._schedule_enabled = True
        self._schedule_enabled = False
        self._schedule_seasons = ["summer", "autumn", "winter", "spring"]
        self._schedule_daytimes = ["day", "evening", "night", "morning"]
        self._schedule_episodes = 5
        self._schedule_custom = None
        self._schedule_flat = []
        self._schedule_current_index = 0
        self._schedule_current_episode = 0
        self._env_manager = None

        # --- Sub-module Composition ---
        self.physics = MissionPhysics(self)
        self.obstacles_helper = MissionObstacles(self)
        self.mines_helper = MissionMines(self)
        self.garbage_helper = MissionGarbage(self)
        self.layout = MissionLayout(self)
        self.ui = MissionUI(self)
        self.tags_vehicles = MissionTagsVehicles(self)
        self.settings = MissionSettings(self)

    # ------------------------------------------------------------------
    # Physics delegation
    # ------------------------------------------------------------------
    def toggle_250hz_loop(self):
        return self.physics.toggle_250hz_loop()

    def start_motor_replay(self):
        return self.physics.start_motor_replay()

    def stop_motor_replay(self):
        return self.physics.stop_motor_replay()

    def _on_app_update(self, e: carb.events.IEvent):
        return self.physics._on_app_update(e)

    # ------------------------------------------------------------------
    # Obstacles delegation
    # ------------------------------------------------------------------
    def on_parse_obstacles(self):
        return self.obstacles_helper.on_parse_obstacles()

    def get_distance(self, point1, point2):
        return self.obstacles_helper.get_distance(point1, point2)

    def generate_obstacles_names(self, trees_count, trees_dist, bushes_count, bushes_dist, rng=None):
        return self.obstacles_helper.generate_obstacles_names(trees_count, trees_dist, bushes_count, bushes_dist, rng=rng)

    def on_load_obstacles(self):
        return self.obstacles_helper.on_load_obstacles()

    def on_load_obstacle(self):
        return self.obstacles_helper.on_load_obstacle()

    def on_destroy_obstacles(self):
        return self.obstacles_helper.on_destroy_obstacles()

    # ------------------------------------------------------------------
    # Mines delegation
    # ------------------------------------------------------------------
    def set_mine_type_enabled(self, mine_type, is_enabled):
        return self.mines_helper.set_mine_type_enabled(mine_type, is_enabled)

    def get_selected_mine_types(self):
        return self.mines_helper.get_selected_mine_types()

    def get_available_mine_types(self):
        return self.mines_helper.get_available_mine_types()

    def set_tree_type_enabled(self, tree_type, is_enabled):
        return self.mines_helper.set_tree_type_enabled(tree_type, is_enabled)

    def get_selected_tree_types(self):
        return self.mines_helper.get_selected_tree_types()

    def get_available_tree_types(self):
        return self.mines_helper.get_available_tree_types()

    def on_parse_mines(self):
        return self.mines_helper.on_parse_mines()

    def get_mines_names(self, mines_count):
        return self.mines_helper.get_mines_names(mines_count)

    def on_load_mines(self):
        return self.mines_helper.on_load_mines()

    # ------------------------------------------------------------------
    # Garbage delegation
    # ------------------------------------------------------------------
    def set_garbage_type_enabled(self, garbage_type, is_enabled):
        return self.garbage_helper.set_garbage_type_enabled(garbage_type, is_enabled)

    def get_selected_garbage_types(self):
        return self.garbage_helper.get_selected_garbage_types()

    def get_available_garbage_types(self):
        return self.garbage_helper.get_available_garbage_types()

    def on_parse_garbage(self):
        return self.garbage_helper.on_parse_garbage()

    def get_garbage_names(self, garbage_count):
        return self.garbage_helper.get_garbage_names(garbage_count)

    def on_load_garbage(self):
        return self.garbage_helper.on_load_garbage()

    # ------------------------------------------------------------------
    # Layout delegation
    # ------------------------------------------------------------------
    def spawn_lightweight_layout_copy(self, original_path="/World/layout", copy_path="/World/layout_mission", z_offset=-5000.0, custom_texture_path=""):
        return self.layout.spawn_lightweight_layout_copy(original_path, copy_path, z_offset, custom_texture_path)

    def delete_lightweight_layout_copy(self, copy_path="/World/layout_mission"):
        return self.layout.delete_lightweight_layout_copy(copy_path)

    # ------------------------------------------------------------------
    # UI delegation
    # ------------------------------------------------------------------
    def set_mission_status_label(self, label):
        return self.ui.set_mission_status_label(label)

    def set_mission_settings_label(self, label):
        return self.ui.set_mission_settings_label(label)

    def set_current_seed_label(self, label):
        return self.ui.set_current_seed_label(label)

    def set_current_mission_count_label(self, label):
        return self.ui.set_current_mission_count_label(label)

    def set_current_drag_value_label(self, label):
        return self.ui.set_current_drag_value_label(label)

    def set_tuning_original_log_path_field(self, field: ui.AbstractValueModel):
        return self.ui.set_tuning_original_log_path_field(field)

    def set_tuning_motor_log_path_field(self, field: ui.AbstractValueModel):
        return self.ui.set_tuning_motor_log_path_field(field)

    def set_tuning_max_iterations_field(self, field: ui.AbstractValueModel):
        return self.ui.set_tuning_max_iterations_field(field)

    def set_tuning_threshold_field(self, field: ui.AbstractValueModel):
        return self.ui.set_tuning_threshold_field(field)

    def set_tuning_learning_rate_field(self, field: ui.AbstractValueModel):
        return self.ui.set_tuning_learning_rate_field(field)

    def set_tuning_cooldown_field(self, field: ui.AbstractValueModel):
        return self.ui.set_tuning_cooldown_field(field)

    def set_tuning_use_dtw_checkbox(self, checkbox: ui.AbstractValueModel):
        return self.ui.set_tuning_use_dtw_checkbox(checkbox)

    def set_tuning_current_params_label(self, label):
        return self.ui.set_tuning_current_params_label(label)

    def set_tuning_best_params_label(self, label):
        return self.ui.set_tuning_best_params_label(label)

    def set_tuning_best_error_label(self, label):
        return self.ui.set_tuning_best_error_label(label)

    def set_tuning_live_params_label(self, label):
        return self.ui.set_tuning_live_params_label(label)

    def set_tuning_current_error_label(self, label):
        return self.ui.set_tuning_current_error_label(label)

    def set_tuning_perturbation_indicator_label(self, label):
        return self.ui.set_tuning_perturbation_indicator_label(label)

    def set_tuning_allow_negative_checkbox(self, checkbox: ui.AbstractValueModel):
        return self.ui.set_tuning_allow_negative_checkbox(checkbox)

    def update_drag_value(self):
        return self.ui.update_drag_value()

    def update_mission_status(self, running: bool):
        return self.ui.update_mission_status(running)

    def update_current_mission_count(self, running: bool):
        return self.ui.update_current_mission_count(running)

    def update_current_seed(self):
        return self.ui.update_current_seed()

    def update_mission_settings_label(self):
        return self.ui.update_mission_settings_label()

    # ------------------------------------------------------------------
    # Tags & Vehicles delegation
    # ------------------------------------------------------------------
    def on_reset_tag(self):
        return self.tags_vehicles.on_reset_tag()

    def on_start_tag(self):
        return self.tags_vehicles.on_start_tag()

    def on_pause_tag(self):
        return self.tags_vehicles.on_pause_tag()

    def move_cycle(self):
        return self.tags_vehicles.move_cycle()

    def generate_formation(self, current_formation, mapper_enabled=True):
        return self.tags_vehicles.generate_formation(current_formation, mapper_enabled)

    def on_reset_tags(self):
        return self.tags_vehicles.on_reset_tags()

    def on_reset_imu(self):
        return self.tags_vehicles.on_reset_imu()

    def on_reset_vehicles(self):
        return self.tags_vehicles.on_reset_vehicles()

    def on_reset_vehicle(self, multirotor, formation):
        return self.tags_vehicles.on_reset_vehicle(multirotor, formation)

    # ------------------------------------------------------------------
    # Settings delegation
    # ------------------------------------------------------------------
    def on_save_mission_settings(self, seed, mission_type, obstacles_count, obstacles_distance, mission_count, mines_count, mines_distance, garbage_count=0, garbage_distance=0):
        return self.settings.on_save_mission_settings(seed, mission_type, obstacles_count, obstacles_distance, mission_count, mines_count, mines_distance, garbage_count, garbage_distance)

    # ------------------------------------------------------------------
    # Core Mission Logic (kept in MissionHelper)
    # ------------------------------------------------------------------

    def check_custom_imu(self):
        if self._custom_imu_enabled_checkbox.get_value_as_bool():
            self.multirotors[0]._backends[0].custom_imu_enabled = True
            self.multirotors[0]._backends[0].custom_imu_path = self._custom_imu_path_field.get_value_as_string()
        else:
            self.multirotors[0]._backends[0].custom_imu_enabled = False

    def check_custom_motors(self):
        if self._custom_motors_enabled_checkbox.get_value_as_bool():
            self.multirotors[0]._backends[0].use_motor_csv = True
            if not self.multirotors[0]._backends[0].motor_csv_loaded:
                self.multirotors[0]._backends[0].load_motor_csv(self._custom_motors_path_field.get_value_as_string())
            self.start_motor_replay()
        else:
            self.multirotors[0]._backends[0].use_motor_csv = False

    def stop_custom_motors(self):
        self.multirotors[0]._backends[0].motor_csv_index = 0
        self.multirotors[0]._backends[0].use_motor_csv = False

    def check_custom_state(self):
        if self._custom_state_enabled_checkbox.get_value_as_bool():
            self.multirotors[0].use_state_csv = True
            if not self.multirotors[0].state_csv_loaded:
                self.multirotors[0].load_state_csv(self._custom_state_path_field.get_value_as_string())
            carb.log_warn("State CSV replay enabled")
        else:
            self.multirotors[0].use_state_csv = False

    def stop_custom_state(self):
        self.multirotors[0].state_csv_index = 0
        self.multirotors[0].state_csv_loaded = False
        self.multirotors[0].use_state_csv = False

    def get_collision_detection(self):
        if not self._multiple_mission_enabled_checkbox.get_value_as_bool():
            if self._multirotor:
                return self._multirotor.check_for_collisions()
            return False

        if self.ultimate_mission_check:
            multirotor_collisions = [multirotor.check_for_collisions() for multirotor in self.multirotors]
            if self.mission_time is not None and self.mission_time + self._collision_restart_time_field.get_value_as_float() < time.time():
                for i, collision in enumerate(multirotor_collisions):
                    if collision:
                        self.mission_finished[i] = True
            collision_detected = False
        else:
            collision_detected = any([multirotor.check_for_collisions() for multirotor in self.multirotors])
          
        return collision_detected

    def mission_cycle(self):
        if self.mission_time is None:
            return

        if self.last_collision_check_time is None:
            self.last_collision_check_time = time.time()

        self.check_custom_imu()
        
        collision_detected = False
        collision_restart_enabled = self._restart_collision_enabled_checkbox.get_value_as_bool()

        current_time = time.time()
        
        if self._trees_wind_enabled_checkbox.get_value_as_bool():
            
            if current_time - self._last_wind_update >= self.wind_update_rate:
                self._last_wind_update = current_time
                
                cull_distance = self._wind_distance_field.get_value_as_float()
                strength = self._wind_strength_field.get_value_as_float()
                speed = self._wind_speed_field.get_value_as_float()

                if self.stage is None:
                    self.stage = omni.usd.get_context().get_stage()
                
                if self.body_prims[0] is None:
                    self.body_prims[0] = self.stage.GetPrimAtPath(f"{self.multirotors[0].prim_path}/body")

                xform = UsdGeom.Xformable(self.body_prims[0]).ComputeLocalToWorldTransform(0.0)
                uav_pos = xform.ExtractTranslation()
                current_pos_vec = np.array(list(uav_pos))
                uav_point = np.array([current_pos_vec[0], current_pos_vec[1]])
                
                for obstacle in self._obstacles:
                    if "Tree" in obstacle._usd_file:
                        tree_point = np.array([obstacle.pos[0], obstacle.pos[1]])
                        distance = np.linalg.norm(uav_point - tree_point)
                        
                        if distance < cull_distance:
                            
                            blend_factor = max(0.0, min(1.0, (cull_distance - distance) / self.fade_margin))
                            
                            smooth_strength = strength * blend_factor
                            
                            obstacle.apply_wind(
                                current_time=current_time, 
                                wind_strength=smooth_strength, 
                                wind_speed=speed
                            )

        if (collision_restart_enabled and self.last_collision_check_time is not None 
            and self.last_collision_check_time + self.collision_cycle_time < time.time()):
            collision_detected = self.get_collision_detection()
            self.last_collision_check_time = time.time()

        if self.second_reset_available and self.mission_time + 5 < time.time():
            self.second_reset_available = False
            self.mission_time = time.time()
            self.check_custom_motors()
            self.check_custom_state()
            if self.tuner is not None:
                self.tuner._state = self.tuner.S_EPISODE_RUNNING


        if (self.mission_time + self._mission_cycle_time < time.time()
            or (self.reset_time is not None and self.reset_time + self.time_to_reset < time.time())
            or (collision_detected and self.mission_time + self._collision_restart_time_field.get_value_as_float() < time.time())):
            
            if self.current_mission_count == self._missions_count:
                self.on_stop_mission()
                return
            if self.ultimate_mission_check:
                self.current_ultimate_mission_count += 1
                if self.current_ultimate_mission_count >= self.ultimate_mission_count:
                    self.current_ultimate_mission_map += 1
                    self.current_ultimate_mission_count = 0
                    self.current_ultimate_mission = self.ultimate_mission_setup[self.current_ultimate_mission_map]
                    if self.current_ultimate_mission_map % 4 == 0:
                        self._seed += 1
                        random.seed(self._seed)
                        self.update_current_seed()
                    self.current_tag_formation = self.current_ultimate_mission["map_setup"]["tag_position"]
                    if self._remove_enabled_checkbox.get_value_as_bool():
                        self.on_destroy_obstacles()
                    self.on_reset_tags()
            
            if self._img_map_enabled_checkbox.get_value_as_bool():
                index = self._spawn_dropdown.get_item_value_model().as_int
                spawn_map_type = self._spawn_names[index]
                spawn_maps = SPAWN_MAPS[spawn_map_type]
                trees_map = f'{self.maps_folder}/{spawn_maps["trees"]}'
                mines_map = f'{self.maps_folder}/{spawn_maps["mines"]}'
            else:
                trees_map = self.empty_probability_map
                mines_map = self.empty_probability_map

            self.trees_map_image = ImageMapHelper.load_probability_map(trees_map)
            self.mines_map_image = ImageMapHelper.load_probability_map(mines_map)
           
            self.reset_velocities()

            self.on_reset_tag()
            self.on_reset_vehicles()
            self.on_reset_imu()
            self.mission_time = time.time()
            if self.selected_mission == "Movable":
                self.on_start_tag()
            elif self.selected_mission == "Static Random":
                self._seed += 1
                random.seed(self._seed)
                self.update_current_seed()
                box = [self._window.get_min_tag_box(), self._window.get_max_tag_box()]
                pos = self.generate_random_position(box)
                self._tag.set_pos(pos)

            if self.ultimate_mission_check:
                self.update_mission_settings_label()
            self.mission_finished = [False, False, False]
            self.second_reset_available = True
            self.reset_time = None
            self.current_mission_count += 1
            self.update_current_mission_count(True)

            if self._schedule_enabled and self._schedule_flat:
                self._schedule_flat = self._build_schedule_flat()
                if self._schedule_current_index < len(self._schedule_flat):
                    entry = self._schedule_flat[self._schedule_current_index]
                    self._schedule_current_episode += 1
                    if self._schedule_current_episode >= entry[0]:
                        self._schedule_current_episode = 0
                        self._schedule_current_index += 1
                        if self._schedule_current_index < len(self._schedule_flat):
                            next_e = self._schedule_flat[self._schedule_current_index]
                            self._get_env_manager().apply_season(next_e[1], replace_trees=True)
                            self._get_env_manager().apply_daytime(next_e[2])
                            self.mines_helper.update_vegetation_for_season(next_e[1])

            self._update_schedule_status()
            self.update_drag_value()

    def _update_schedule_status(self):
        if not hasattr(self, '_window') or self._window is None:
            return
        label = getattr(self._window, '_schedule_status_label', None)
        if label is None:
            return
        if self.mission_time is None or not self._schedule_enabled or not self._schedule_flat:
            label.text = "--"
        else:
            idx = self._schedule_current_index
            if idx >= len(self._schedule_flat):
                label.text = "Done"
            else:
                entry = self._schedule_flat[idx]
                ep = self._schedule_current_episode + 1
                label.text = f"#{idx + 1}: {ep}/{entry[0]} {entry[1]}/{entry[2]}"

        preview_frame = getattr(self._window, '_preview_frame', None)
        if preview_frame is not None:
            preview_frame.rebuild()

    def generate_random_position(self, random_box, rng=None):
        if rng is None:
            rng = random
        x = rng.uniform(random_box[0][0], random_box[1][0])
        y = rng.uniform(random_box[0][1], random_box[1][1])
        z = rng.uniform(random_box[0][2], random_box[1][2])
        return [x, y, z]

    def _get_env_manager(self):
        if self._env_manager is None:
            self._env_manager = EnvironmentManager()
            self._env_manager.set_stage()
        return self._env_manager

    def _build_schedule_flat(self):
        if self._schedule_custom is not None:
            return list(self._schedule_custom)
        flat = []
        for s in self._schedule_seasons:
            for d in self._schedule_daytimes:
                flat.append((self._schedule_episodes, s, d))
        return flat

    def on_start_mission(self):
        mission_index = self._mission_dropdown.get_item_value_model().as_int
        self.selected_mission = self._mission_names[mission_index]
        self._seed = int(self._seed_field.get_value_as_string())
        self._mission_cycle_time = int(self._mission_time_field.get_value_as_string())

        obstacles_count = int(self._obstacles_count_field.get_value_as_string())
        obstacles_distance = float(self._obstacles_distance_field.get_value_as_string())

        mines_count = int(self._mines_count_field.get_value_as_string())
        mines_distance = float(self._mines_distance_field.get_value_as_string())

        garbage_count = int(self._garbage_count_field.get_value_as_string())
        garbage_distance = float(self._garbage_distance_field.get_value_as_string())

        self.ultimate_mission_check = self._ultimate_mission_enabled_checkbox.get_value_as_bool()

        if self.ultimate_mission_check:
            self.current_ultimate_mission = self.ultimate_mission_setup[self.current_ultimate_mission_map]

        if self._img_map_enabled_checkbox.get_value_as_bool():
            index = self._spawn_dropdown.get_item_value_model().as_int
            spawn_map_type = self._spawn_names[index]
            spawn_maps = SPAWN_MAPS[spawn_map_type]
            trees_map = f'{self.maps_folder}/{spawn_maps["trees"]}'
            mines_map = f'{self.maps_folder}/{spawn_maps["mines"]}'
        else:
            trees_map = self.empty_probability_map
            mines_map = self.empty_probability_map

        self.trees_map_image = ImageMapHelper.load_probability_map(trees_map)
        self.mines_map_image = ImageMapHelper.load_probability_map(mines_map)
        
        self.trees_map = np.array(self.trees_map_image)
        self.mines_map = np.array(self.mines_map_image)

        random.seed(self._seed)
        if self.selected_mission == "Movable":
            self.on_start_tag()
        elif self.selected_mission == "Static Random":
            box = [self._window.get_min_tag_box(), self._window.get_max_tag_box()]
            pos = self.generate_random_position(box)
            self._tag.set_pos(pos)
        elif self.selected_mission == "Static Obstacles":
            self.update_current_seed()
            if self.is_first_start or self._obstacles_reload_enabled_checkbox.get_value_as_bool():
                self.on_load_obstacles()
                self.is_first_start = False
            if self._spawn_mines_enabled_checkbox.get_value_as_bool():    
                self.on_load_mines()
            if self._spawn_garbage_enabled_checkbox.get_value_as_bool():
                self.on_load_garbage()
        
        if self.selected_mission != "Static Obstacles":
            self.on_parse_obstacles()

        if self.ultimate_mission_check:
            self._missions_count = self.ultimate_mission_maps * self.ultimate_mission_count
        else:
            self._missions_count = int(self._missions_count_field.get_value_as_string())
        
        if self.ultimate_mission_check:
            mission_count = self.ultimate_mission_count * self.ultimate_mission_maps
            self.on_save_mission_settings(self._seed, self.selected_mission, -1, obstacles_distance, mission_count, 0, 0, 0, 0)
        else:    
            self.on_save_mission_settings(self._seed, self.selected_mission, obstacles_count, obstacles_distance, self._missions_count, mines_count, mines_distance, garbage_count, garbage_distance)
        
        self.mission_time = time.time()
        self.update_mission_status(True)

        if self._drag_calibration_checkbox.get_value_as_bool():
            self.drag_test_index = 0
        
        self.update_drag_value()

        self.update_current_mission_count(True)
        if self.ultimate_mission_check:
            self.update_mission_settings_label()
        # TEST: OFF RESET VEHICLES
        # self.on_reset_vehicles()
        self.on_reset_imu()
        if self.ultimate_mission_check:
            self.current_tag_formation = self.current_ultimate_mission["map_setup"]["tag_position"]
            self.on_reset_tags()
        
        thermal_layout_enabled = self._thermal_layout_enabled_checkbox.get_value_as_bool()
        
        if thermal_layout_enabled:
            self.spawn_lightweight_layout_copy()

        if self._tuning_enabled_checkbox.get_value_as_bool():
            self.on_tuning_start()

        self.check_custom_motors()
        self.check_custom_state()
        if self.tuner is not None:
            self.tuner._state = self.tuner.S_EPISODE_RUNNING

        if self._schedule_enabled:
            self._schedule_flat = self._build_schedule_flat()
            self._schedule_current_index = 0
            self._schedule_current_episode = 0
            if self._schedule_flat:
                e = self._schedule_flat[0]
                self._get_env_manager().apply_season(e[1], replace_trees=True)
                self._get_env_manager().apply_daytime(e[2])
                self.mines_helper.update_vegetation_for_season(e[1])

        self._update_schedule_status()
        carb.log_warn("Mission started!")

    def on_stop_mission(self):
        self.mission_time = None
        self.reset_velocities()
        self.on_reset_tag()
        self.on_reset_vehicles()
        self.update_mission_status(False)
        self.second_reset_available = False
        self.stop_custom_motors()
        self.stop_custom_state()

        if self._remove_enabled_checkbox.get_value_as_bool():
            self.on_destroy_obstacles()
            
        self.current_mission_count = 1
        self.update_current_mission_count(False)
        self.last_collision_check_time = None
        
        self.on_save_mission_settings(0, "Static", 0, 0, 1, 0, 0)

        if self._quit_enabled_checkbox.get_value_as_bool():
            app = omni.kit.app.get_app()
            app.post_quit()
        
        thermal_layout_enabled = self._thermal_layout_enabled_checkbox.get_value_as_bool()
        
        if thermal_layout_enabled:
            self.delete_lightweight_layout_copy()

        self.stop_motor_replay()

        if self.tuner is not None:
            self.on_tuning_stop()

        self._update_schedule_status()

    def reset_velocities(self):
        self.multirotors[0].reset_velocities()

    def check_mission_end(self):
        if not self._restart_success_enabled_checkbox.get_value_as_bool():
            return
        
        vehicles_to_check = []

        if not self._multiple_mission_enabled_checkbox.get_value_as_bool():
            if self.multirotors:
                vehicles_to_check.append(self.multirotors[0])
        else:
            vehicles_to_check = self.multirotors

        if self.mission_time is not None:
            if self.stage is None:
                self.stage = omni.usd.get_context().get_stage()
            for i, vehicle in enumerate(vehicles_to_check):
                if i >= len(self.body_prims): 
                    break
                    
                if self.body_prims[i] is None:
                    self.body_prims[i] = self.stage.GetPrimAtPath(f"{vehicle.prim_path}/body")
                
                if not self.body_prims[i] or not self.body_prims[i].IsValid():
                    continue

                xform = UsdGeom.Xformable(self.body_prims[i]).ComputeLocalToWorldTransform(0.0)
                if xform is not None and self.reset_time is None and not self.mission_finished[i]:
                    position = xform.ExtractTranslation()
                    current_pos_vec = np.array(list(position))

                    if i < len(self.tags):
                        tag_pos = np.array(list(self.tags[i].pos))
                        distance_to_target = np.linalg.norm(tag_pos - current_pos_vec)
                        if distance_to_target < self.target_reached_tolerance:
                            self.mission_finished[i] = True
                        else:
                            self.mission_finished[i] = False
            
            if all([self.mission_finished[i] for i in range(len(vehicles_to_check))]) and self.reset_time is None:
                self.reset_time = time.time()

    # ------------------------------------------------------------------
    # Parameter Tuning
    # ------------------------------------------------------------------
    def _get_tuner(self) -> ParameterTuner:
        if self.tuner is None:
            self.tuner = ParameterTuner(self)
        return self.tuner

    def on_tuning_start(self):
        original_log_path = self._custom_state_path_field.get_value_as_string().strip()
        motor_log_path = self._custom_motors_path_field.get_value_as_string().strip()

        if not original_log_path or not motor_log_path:
            carb.log_warn("[mission_helper] Both original log and motor log paths are required.")
            return
        try:
            max_iter = int(self._tuning_max_iterations_field.get_value_as_string())
            threshold = float(self._tuning_threshold_field.get_value_as_string())
            learning_rate = float(self._tuning_learning_rate_field.get_value_as_string())
            perturbation_eps = self._perturbation_eps_field.get_value_as_float()
            cooldown = float(self._tuning_cooldown_field.get_value_as_string()) if self._tuning_cooldown_field is not None else 2.0
        except ValueError as e:
            carb.log_error(f"[mission_helper] Invalid tuning numeric input: {e}")
            return
        
        use_dtw = self._tuning_use_dtw_checkbox.get_value_as_bool() if self._tuning_use_dtw_checkbox is not None else False
        allow_negative = (
            self._tuning_allow_negative_checkbox.get_value_as_bool()
            if self._tuning_allow_negative_checkbox is not None
            else False
        )

        self._get_tuner().start(
            original_log_path=original_log_path,
            motor_log_path=motor_log_path,
            max_iterations=max_iter,
            threshold=threshold,
            learning_rate=learning_rate,
            error_method="dtw" if use_dtw else "euclidean",
            cooldown_seconds=cooldown,
            allow_negative=allow_negative,
            perturbation_eps=perturbation_eps
        )

    def on_tuning_stop(self):
        if self.tuner is not None:
            self.tuner.stop()

    def on_tuning_apply_best(self):
        if self.tuner is None:
            carb.log_warn("[mission_helper] No tuner state; cannot apply best parameters.")
            return
        self.tuner.apply_best_parameters()
