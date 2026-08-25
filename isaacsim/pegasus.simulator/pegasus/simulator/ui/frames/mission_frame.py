"""
File: mission_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Mission Planner frame for the UI
"""

__all__ = [
    "build_mission_frame",
    "get_min_tag_box",
    "get_max_tag_box",
    "get_min_obstacle_box",
    "get_max_obstacle_box",
    "get_min_mines_box",
    "get_max_mines_box",
    "get_min_garbage_box",
    "get_max_garbage_box",
    "get_drag_value",
]

import numpy as np
import omni.ui as ui

from pegasus.simulator.params import MISSIONS, OBSTACLES, MINES_VEGETATION_TYPES
from pegasus.simulator.ui.utils.coords_builder import coords_builder


def build_mission_frame(window, delegate):
    """
    Builds the Mission Planner frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    min_box_default_values = [15, -5, 3]
    max_box_default_values = [25, 5, 7]

    min_box_obstacles_default_values = [15, -90, 0]
    max_box_obstacles_default_values = [190, 90, 0]

    min_box_mines_default_values = [15, -40, 0]
    max_box_mines_default_values = [190, 40, 0]

    min_box_garbage_default_values = [15, -40, 0]
    max_box_garbage_default_values = [190, 40, 0]

    drag_default_values = [0.1, 0.1, 0.0]

    helper = delegate._mission_helper
    widgets_to_disable = []

    with ui.CollapsableFrame("Mission Planner"):
        with ui.VStack(spacing=10, height=0):
            with ui.HStack():
                ui.Label("Mission Status:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                window.mission_status_label = ui.Label("Stopped", name="mission_status")
                delegate.set_mission_status_label(window.mission_status_label)
            with ui.HStack():
                ui.Label("Current Count:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                window.current_mission_count_label = ui.Label("0", name="mission_status")
                delegate.set_current_mission_count_label(window.current_mission_count_label)
            with ui.HStack():
                ui.Label("Current Seed:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                window.current_seed_label = ui.Label(str(helper._seed), name="mission_status")
                delegate.set_current_seed_label(window.current_seed_label)
            with ui.HStack():
                ui.Label("Mission Settings:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                window.mission_settings_label = ui.Label("Undefined", name="mission_settings")
                delegate.set_mission_settings_label(window.mission_settings_label)

            with ui.HStack():
                ui.Label("Mission Type", name="label", width=window.LABEL_PADDING,
                         alignment=ui.Alignment.TOP)
                dropdown_menu = ui.ComboBox(0, name="missions")
                for mission in MISSIONS:
                    dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(mission))
                delegate.set_mission_dropdown(dropdown_menu.model)
                widgets_to_disable.append(dropdown_menu)

            with ui.HStack():
                ui.Label("Mission Time (s)", name="label", width=window.LABEL_PADDING,
                         alignment=ui.Alignment.TOP)
                mission_time_field = ui.StringField(name="mission_time")
                mission_time_field.model.set_value(str(helper._mission_cycle_time))
                delegate.set_mission_time_field(mission_time_field.model)
                widgets_to_disable.append(mission_time_field)

            with ui.HStack():
                ui.Label("Mission Count", name="label", width=window.LABEL_PADDING)
                mission_count_field = ui.StringField(name="mission_count")
                mission_count_field.model.set_value(str(helper._missions_count))
                delegate.set_mission_count_field(mission_count_field.model)
                widgets_to_disable.append(mission_count_field)

            with ui.HStack():
                ui.Label("Initial Seed", name="label", width=window.LABEL_PADDING)
                seed_field = ui.StringField(name="initial_seed")
                seed_field.model.set_value(str(helper._seed))
                delegate.set_seed_field(seed_field.model)
                widgets_to_disable.append(seed_field)

            with ui.HStack(spacing=10):
                ui.Label("Quit When Finished", name="label", width=window.LABEL_PADDING)
                quit_enabled_checkbox = ui.CheckBox()
                quit_enabled_checkbox.model.set_value(helper._quit_enabled)
                delegate.set_quit_enabled_checkbox(quit_enabled_checkbox.model)
                widgets_to_disable.append(quit_enabled_checkbox)

            with ui.HStack(spacing=10):
                ui.Label("Restart When Success", name="label", width=window.LABEL_PADDING)
                restart_success_enabled_checkbox = ui.CheckBox()
                restart_success_enabled_checkbox.model.set_value(helper._restart_success_enabled)
                delegate.set_restart_success_enabled_checkbox(restart_success_enabled_checkbox.model)
                widgets_to_disable.append(restart_success_enabled_checkbox)

            with ui.HStack(spacing=10):
                ui.Label("Restart When Collided", name="label", width=window.LABEL_PADDING)
                restart_collision_enabled_checkbox = ui.CheckBox()
                restart_collision_enabled_checkbox.model.set_value(helper._restart_collision_enabled)
                delegate.set_restart_collision_enabled_checkbox(restart_collision_enabled_checkbox.model)
                widgets_to_disable.append(restart_collision_enabled_checkbox)
            
            with ui.HStack(spacing=10): 
                ui.Label("Collision Restart Time", name="label", width=window.LABEL_PADDING)
                collision_restart_time_field = ui.FloatField()
                collision_restart_time_field.model.set_value(helper._collision_restart_time)
                delegate.set_collision_restart_time_field(collision_restart_time_field.model)

            with ui.HStack(spacing=10):
                ui.Label("Multiple Vehicles Mission", name="label", width=window.LABEL_PADDING)
                multiple_mission_enabled_checkbox = ui.CheckBox()
                multiple_mission_enabled_checkbox.model.set_value(helper._multiple_mission_enabled)
                delegate.set_multiple_mission_enabled_checkbox(multiple_mission_enabled_checkbox.model)
                widgets_to_disable.append(multiple_mission_enabled_checkbox)

            with ui.HStack(spacing=10):
                ui.Label("Alternative Spawn Logic", name="label", width=window.LABEL_PADDING)
                alt_spawn_enabled_checkbox = ui.CheckBox()
                alt_spawn_enabled_checkbox.model.set_value(helper._alt_spawn_enabled)
                delegate.set_alt_spawn_enabled_checkbox(alt_spawn_enabled_checkbox.model)

            with ui.HStack(spacing=10):
                ui.Label("Save Spawn Images", name="label", width=window.LABEL_PADDING)
                save_spawn_enabled_checkbox = ui.CheckBox()
                save_spawn_enabled_checkbox.model.set_value(helper._save_spawn_enabled)
                delegate.set_save_spawn_enabled_checkbox(save_spawn_enabled_checkbox.model)

            with ui.HStack(spacing=10):
                ui.Label("Spawn Landmines", name="label", width=window.LABEL_PADDING)
                spawn_mines_enabled_checkbox = ui.CheckBox()
                spawn_mines_enabled_checkbox.model.set_value(helper._spawn_mines_enabled)
                delegate.set_spawn_mines_enabled_checkbox(spawn_mines_enabled_checkbox.model)

            with ui.HStack(spacing=10):
                ui.Label("Spawn Garbage", name="label", width=window.LABEL_PADDING)
                spawn_garbage_enabled_checkbox = ui.CheckBox()
                spawn_garbage_enabled_checkbox.model.set_value(helper._spawn_garbage_enabled)
                delegate.set_spawn_garbage_enabled_checkbox(spawn_garbage_enabled_checkbox.model)

            with ui.CollapsableFrame("Deprecated Options", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Ultimate Test Missions", name="label", width=window.LABEL_PADDING)
                        ultimate_mission_enabled_checkbox = ui.CheckBox()
                        ultimate_mission_enabled_checkbox.model.set_value(helper._ultimate_mission_enabled)
                        delegate.set_ultimate_mission_enabled_checkbox(ultimate_mission_enabled_checkbox.model)               
                    with ui.HStack(spacing=10):
                        ui.Label("Thermal Textures", name="label", width=window.LABEL_PADDING)
                        thermal_textures_enabled_checkbox = ui.CheckBox()
                        thermal_textures_enabled_checkbox.model.set_value(helper._thermal_textures_enabled)
                        delegate.set_thermal_textures_enabled_checkbox(thermal_textures_enabled_checkbox.model)
                    with ui.HStack(spacing=10):
                        ui.Label("Spawn Thermal Layout", name="label", width=window.LABEL_PADDING)
                        thermal_layout_enabled_checkbox = ui.CheckBox()
                        thermal_layout_enabled_checkbox.model.set_value(helper._thermal_layout_enabled)
                        delegate.set_thermal_layout_enabled_checkbox(thermal_layout_enabled_checkbox.model)
                
                    with ui.CollapsableFrame("Drag Calibration Setup", collapsed=True):
                        with ui.VStack(spacing=10, height=0):
                            with ui.HStack(spacing=10):
                                ui.Label("Drag Calibration", name="label", width=window.LABEL_PADDING)
                                drag_calibration_checkbox = ui.CheckBox()
                                drag_calibration_checkbox.model.set_value(helper._drag_calibration)
                                delegate.set_drag_calibration_checkbox(drag_calibration_checkbox.model)
                            with ui.HStack(spacing=10):
                                ui.Label("Drag Custom Value", name="label", width=window.LABEL_PADDING)
                                drag_custom_value_checkbox = ui.CheckBox()
                                drag_custom_value_checkbox.model.set_value(helper._drag_custom_value)
                                delegate.set_drag_custom_value_checkbox(drag_custom_value_checkbox.model)
                            with ui.HStack(spacing=10):
                                ui.Label("Current Drag Value:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.current_drag_value_label = ui.Label("Undefined", name="drag_value")
                                delegate.set_current_drag_value_label(window.current_drag_value_label)
                            coords_builder(window, "Drag Value", window.drag_value_models, all_axis, colors, drag_default_values, 0, 1)
                    ui.Spacer(height=1)
                    ui.Separator()
            reproducing_frame = ui.CollapsableFrame("Mission Reproducing", collapsed=True)
            widgets_to_disable.append(reproducing_frame)
            with reproducing_frame:
                with ui.VStack(spacing=10, height=0):
                    with ui.CollapsableFrame("IMU Setup", collapsed=True):
                        with ui.VStack(spacing=10, height=0):
                            with ui.HStack(spacing=10):
                                ui.Label("Custom IMU", name="label", width=window.LABEL_PADDING)
                                custom_imu_enabled_checkbox = ui.CheckBox()
                                custom_imu_enabled_checkbox.model.set_value(helper._custom_imu_enabled)
                                delegate.set_custom_imu_enabled_checkbox(custom_imu_enabled_checkbox.model)
                            with ui.HStack(spacing=10):
                                ui.Label("Custom IMU Path", name="label", width=window.LABEL_PADDING)
                                custom_imu_path_field = ui.StringField()
                                custom_imu_path_field.model.set_value(helper._custom_imu_path)
                                delegate.set_custom_imu_path_field(custom_imu_path_field.model)
                            ui.Spacer(height=1)
                            ui.Separator()
                    with ui.CollapsableFrame("Motors Setup", collapsed=False):
                        with ui.VStack(spacing=10, height=0):
                            with ui.HStack(spacing=10):
                                ui.Label("Custom Motors", name="label", width=window.LABEL_PADDING)
                                custom_motors_enabled_checkbox = ui.CheckBox()
                                custom_motors_enabled_checkbox.model.set_value(helper._custom_motors_enabled)
                                delegate.set_custom_motors_enabled_checkbox(custom_motors_enabled_checkbox.model)
                            with ui.HStack(spacing=10):
                                ui.Label("Custom Motors Path", name="label", width=window.LABEL_PADDING)
                                custom_motors_path_field = ui.StringField()
                                custom_motors_path_field.model.set_value(helper._custom_motors_path)
                                delegate.set_custom_motors_path_field(custom_motors_path_field.model)
                            ui.Spacer(height=1)
                            ui.Separator()                                
                    with ui.CollapsableFrame("State Setup", collapsed=False):
                        with ui.VStack(spacing=10, height=0):
                            with ui.HStack(spacing=10):
                                ui.Label("Custom State", name="label", width=window.LABEL_PADDING)
                                custom_state_enabled_checkbox = ui.CheckBox()
                                custom_state_enabled_checkbox.model.set_value(helper._custom_state_enabled)
                                delegate.set_custom_state_enabled_checkbox(custom_state_enabled_checkbox.model)
                            with ui.HStack(spacing=10):
                                ui.Label("Custom State Path", name="label", width=window.LABEL_PADDING)
                                custom_state_path_field = ui.StringField()
                                custom_state_path_field.model.set_value(helper._custom_state_path)
                                delegate.set_custom_state_path_field(custom_state_path_field.model)
                            ui.Spacer(height=1)
                            ui.Separator()
                    with ui.CollapsableFrame("Parameter Tuning", collapsed=True):
                        with ui.VStack(spacing=10, height=0):
                            with ui.HStack(spacing=10): 
                                ui.Label("Enable Tuning", name="label", width=window.LABEL_PADDING)
                                tuning_enabled_checkbox = ui.CheckBox()
                                tuning_enabled_checkbox.model.set_value(helper._tuning_enabled)
                                delegate.set_tuning_enabled_checkbox(tuning_enabled_checkbox.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Max Iterations", name="label", width=window.LABEL_PADDING)
                                tuning_max_iterations_field = ui.StringField()
                                tuning_max_iterations_field.model.set_value(str(helper._tuning_max_iterations))
                                delegate.set_tuning_max_iterations_field(tuning_max_iterations_field.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Threshold", name="label", width=window.LABEL_PADDING)
                                tuning_threshold_field = ui.StringField()
                                tuning_threshold_field.model.set_value(str(helper._tuning_threshold))
                                delegate.set_tuning_threshold_field(tuning_threshold_field.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Adam Step Size", name="label", width=window.LABEL_PADDING, tooltip="Adam optimizer step size. Typical range: 0.0001 .. 0.05 (default 0.01).")
                                tuning_learning_rate_field = ui.StringField()
                                tuning_learning_rate_field.model.set_value(str(helper._tuning_learning_rate))
                                delegate.set_tuning_learning_rate_field(tuning_learning_rate_field.model)

                            with ui.HStack(spacing=10): 
                                ui.Label("Perturbation Eps", name="label", width=window.LABEL_PADDING)
                                perturbation_eps_field = ui.FloatField()
                                perturbation_eps_field.model.set_value(helper._perturbation_eps)
                                delegate.set_perturbation_eps_field(perturbation_eps_field.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Cooldown (s)", name="label", width=window.LABEL_PADDING)
                                tuning_cooldown_field = ui.StringField()
                                tuning_cooldown_field.model.set_value(str(helper._tuning_cooldown))
                                delegate.set_tuning_cooldown_field(tuning_cooldown_field.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Use DTW", name="label", width=window.LABEL_PADDING)
                                tuning_use_dtw_checkbox = ui.CheckBox()
                                tuning_use_dtw_checkbox.model.set_value(helper._tuning_use_dtw)
                                delegate.set_tuning_use_dtw_checkbox(tuning_use_dtw_checkbox.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Allow Negative", name="label", width=window.LABEL_PADDING)
                                tuning_allow_negative_checkbox = ui.CheckBox()
                                tuning_allow_negative_checkbox.model.set_value(helper._tuning_allow_negative)
                                delegate.set_tuning_allow_negative_checkbox(tuning_allow_negative_checkbox.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Current Parameters:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.tuning_current_params_label = ui.Label("N/A", name="tuning_label")
                                delegate.set_tuning_current_params_label(window.tuning_current_params_label)

                            with ui.HStack(spacing=10):
                                ui.Label("Best Parameters:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.tuning_best_params_label = ui.Label("N/A", name="tuning_label")
                                delegate.set_tuning_best_params_label(window.tuning_best_params_label)

                            with ui.HStack(spacing=10):
                                ui.Label("Status:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.tuning_perturbation_indicator_label = ui.Label("N/A", name="tuning_perturbation_indicator")
                                delegate.set_tuning_perturbation_indicator_label(window.tuning_perturbation_indicator_label)

                            with ui.HStack(spacing=10):
                                ui.Label("Live Parameters:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.tuning_live_params_label = ui.Label("N/A", name="tuning_label")
                                delegate.set_tuning_live_params_label(window.tuning_live_params_label)

                            with ui.HStack(spacing=10):
                                ui.Label("Current Error:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.tuning_current_error_label = ui.Label("N/A", name="tuning_label")
                                delegate.set_tuning_current_error_label(window.tuning_current_error_label)

                            with ui.HStack(spacing=10):
                                ui.Label("Best Error:", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                                window.tuning_best_error_label = ui.Label("N/A", name="tuning_label")
                                delegate.set_tuning_best_error_label(window.tuning_best_error_label)

                            with ui.HStack():
                                ui.Button(
                                    "Apply Best",
                                    height=window.BUTTON_HEIGHT,
                                    clicked_fn=delegate.on_tuning_apply_best,
                                )
                    ui.Spacer(height=1)
                    ui.Separator()
            target_frame = ui.CollapsableFrame("Target Setup", collapsed=True)
            widgets_to_disable.append(target_frame)
            with target_frame:
                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    coords_builder(window, "Min Spawn", window.min_tag_box_models, all_axis, colors, min_box_default_values)
                    coords_builder(window, "Max Spawn", window.max_tag_box_models, all_axis, colors, max_box_default_values)

            # ui.Spacer(height=1)

            obstacles_frame = ui.CollapsableFrame("Obstacles Setup", collapsed=True)
            widgets_to_disable.append(obstacles_frame)
            with obstacles_frame:
                available_trees = helper.get_available_tree_types()
                available_bushes = [k for k in OBSTACLES.keys() if k.startswith("Bush_")]

                delegate.bush_checkbox_models = {}

                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    SPAWN_PRESETS_OBS = {
                        "Default": {
                            "trees_count": 200,
                            "trees_distance": 5.0,
                            "bushes_count": 300,
                            "bushes_distance": 2.0,
                            "min_spawn": [15, -90, 0],
                            "max_spawn": [190, 90, 0]
                        },
                        "Lightweight": {
                            "trees_count": 100,
                            "trees_distance": 5.0,
                            "bushes_count": 0,
                            "bushes_distance": 2.0,
                            "min_spawn": [15, -90, 0],
                            "max_spawn": [190, 90, 0]
                        },
                        "Without Bushes": {
                            "trees_count": 200,
                            "trees_distance": 5.0,
                            "bushes_count": 0,
                            "bushes_distance": 2.0,
                            "min_spawn": [15, -90, 0],
                            "max_spawn": [190, 90, 0]
                        },
                        "Empty": {
                            "trees_count": 0,
                            "trees_distance": 5.0,
                            "bushes_count": 0,
                            "bushes_distance": 2.0,
                            "min_spawn": [15, -90, 0],
                            "max_spawn": [190, 90, 0]
                        },
                        "Single": {
                            "trees_count": 1,
                            "trees_distance": 5.0,
                            "bushes_count": 0,
                            "bushes_distance": 2.0,
                            "min_spawn": [15, 0, 0],
                            "max_spawn": [15, 0, 0]
                        },
                    }
                    spawn_preset_obs_names = list(SPAWN_PRESETS_OBS.keys()) + ["Custom"]
                    spawn_obs_custom_index = len(spawn_preset_obs_names) - 1

                    with ui.HStack():
                        ui.Label("Spawn Presets", name="header", width=window.LABEL_PADDING, style={"font_weight": "bold"})
                        spawn_preset_obs_combo = ui.ComboBox(spawn_preset_obs_names.index("Lightweight"), *spawn_preset_obs_names)

                    ui.Spacer(height=1)
                    ui.Separator()
                    ui.Spacer(height=2)

                    window._updating_spawn_obs_ui = False

                    def on_spawn_preset_obs_changed(model, item):
                        if window._updating_spawn_obs_ui:
                            return
                        idx = model.get_item_value_model().as_int
                        if idx == spawn_obs_custom_index:
                            return
                        window._updating_spawn_obs_ui = True
                        selected = spawn_preset_obs_names[idx]
                        config = SPAWN_PRESETS_OBS[selected]
                        obstacles_count_field.model.set_value(config["trees_count"])
                        obstacles_distance_field.model.set_value(config["trees_distance"])
                        bushes_count_field.model.set_value(config["bushes_count"])
                        bushes_distance_field.model.set_value(config["bushes_distance"])
                        for i, v in enumerate(config["min_spawn"]):
                            window.min_obstacle_box_models[i].set_value(v)
                        for i, v in enumerate(config["max_spawn"]):
                            window.max_obstacle_box_models[i].set_value(v)
                        window._updating_spawn_obs_ui = False

                    def on_spawn_obs_field_changed(*_):
                        if window._updating_spawn_obs_ui:
                            return
                        spawn_preset_obs_combo.model.get_item_value_model().set_value(spawn_obs_custom_index)

                    with ui.HStack():
                        ui.Label("Trees Count", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        obstacles_count_field = ui.IntField()
                        obstacles_count_field.model.set_value(helper._obstacles_count)
                        delegate.set_obstacles_count_field(obstacles_count_field.model)

                    with ui.HStack():
                        ui.Label("Trees Distance (m)", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        obstacles_distance_field = ui.FloatField()
                        obstacles_distance_field.model.set_value(helper._obstacles_distance)
                        delegate.set_obstacles_distance_field(obstacles_distance_field.model)

                    with ui.HStack():
                        ui.Label("Bushes Count", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        bushes_count_field = ui.IntField()
                        bushes_count_field.model.set_value(getattr(helper, '_bushes_count', 0))
                        delegate.set_bushes_count_field(bushes_count_field.model)

                    with ui.HStack():
                        ui.Label("Bushes Distance (m)", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        bushes_distance_field = ui.FloatField()
                        bushes_distance_field.model.set_value(getattr(helper, '_bushes_distance', 0))
                        delegate.set_bushes_distance_field(bushes_distance_field.model)

                    with ui.HStack(spacing=10):
                        ui.Label("Random scales", name="label", width=window.LABEL_PADDING)
                        random_scales_enabled_checkbox = ui.CheckBox()
                        random_scales_enabled_checkbox.model.set_value(helper._random_scales_enabled)
                        delegate.set_random_scales_enabled_checkbox(random_scales_enabled_checkbox.model)

                    with ui.HStack(spacing=10):
                        ui.Label("Random rotations", name="label", width=window.LABEL_PADDING)
                        random_rotations_enabled_checkbox = ui.CheckBox()
                        random_rotations_enabled_checkbox.model.set_value(helper._random_rotations_enabled)
                        delegate.set_random_rotations_enabled_checkbox(random_rotations_enabled_checkbox.model)
                    with ui.HStack(spacing=10): 
                        ui.Label("Reload After Start", name="label", width=window.LABEL_PADDING)
                        obstacles_reload_enabled_checkbox = ui.CheckBox()
                        obstacles_reload_enabled_checkbox.model.set_value(helper._obstacles_reload_enabled)
                        delegate.set_obstacles_reload_enabled_checkbox(obstacles_reload_enabled_checkbox.model)
                    with ui.HStack(spacing=10):
                        ui.Label("Remove After Stop", name="label", width=window.LABEL_PADDING)
                        remove_enabled_checkbox = ui.CheckBox()
                        remove_enabled_checkbox.model.set_value(helper._remove_enabled)
                        delegate.set_remove_enabled_checkbox(remove_enabled_checkbox.model)
                    with ui.HStack(spacing=10):
                        ui.Label("Deactivate After Stop", name="label", width=window.LABEL_PADDING)
                        deactivation_enabled_checkbox = ui.CheckBox()
                        deactivation_enabled_checkbox.model.set_value(helper._deactivation_enabled)
                        delegate.set_deactivation_enabled_checkbox(deactivation_enabled_checkbox.model)

                    coords_builder(window, "Min Spawn", window.min_obstacle_box_models, all_axis, colors,
                                  min_box_obstacles_default_values)
                    coords_builder(window, "Max Spawn", window.max_obstacle_box_models, all_axis, colors,
                                  max_box_obstacles_default_values)

                    default_spawn_obs_idx = spawn_preset_obs_combo.model.get_item_value_model().as_int
                    if default_spawn_obs_idx != spawn_obs_custom_index:
                        config = SPAWN_PRESETS_OBS[spawn_preset_obs_names[default_spawn_obs_idx]]
                        obstacles_count_field.model.set_value(config["trees_count"])
                        obstacles_distance_field.model.set_value(config["trees_distance"])
                        bushes_count_field.model.set_value(config["bushes_count"])
                        bushes_distance_field.model.set_value(config["bushes_distance"])
                        for i, v in enumerate(config["min_spawn"]):
                            window.min_obstacle_box_models[i].set_value(v)
                        for i, v in enumerate(config["max_spawn"]):
                            window.max_obstacle_box_models[i].set_value(v)

                    obstacles_count_field.model.add_value_changed_fn(on_spawn_obs_field_changed)
                    obstacles_distance_field.model.add_value_changed_fn(on_spawn_obs_field_changed)
                    bushes_count_field.model.add_value_changed_fn(on_spawn_obs_field_changed)
                    bushes_distance_field.model.add_value_changed_fn(on_spawn_obs_field_changed)
                    for m in window.min_obstacle_box_models:
                        m.add_value_changed_fn(on_spawn_obs_field_changed)
                    for m in window.max_obstacle_box_models:
                        m.add_value_changed_fn(on_spawn_obs_field_changed)

                    spawn_preset_obs_combo.model.add_item_changed_fn(on_spawn_preset_obs_changed)

                    obstacles_wind_frame = ui.CollapsableFrame("Obstacles Wind Setup", collapsed=True)
                    with obstacles_wind_frame:
                        with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                            with ui.HStack(spacing=10):
                                ui.Label("Enable Trees Wind", name="label", width=window.LABEL_PADDING)
                                trees_wind_enabled_checkbox = ui.CheckBox()
                                trees_wind_enabled_checkbox.model.set_value(helper._trees_wind_enabled)
                                delegate.set_trees_wind_enabled_checkbox(trees_wind_enabled_checkbox.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Wind Speed", name="label", width=window.LABEL_PADDING)
                                wind_speed_field = ui.FloatField()
                                wind_speed_field.model.set_value(helper._wind_speed)
                                delegate.set_wind_speed_field(wind_speed_field.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Wind Strength", name="label", width=window.LABEL_PADDING)
                                wind_strength_field = ui.FloatField()
                                wind_strength_field.model.set_value(helper._wind_strength)
                                delegate.set_wind_strength_field(wind_strength_field.model)

                            with ui.HStack(spacing=10):
                                ui.Label("Wind Cull Distance (m)", name="label", width=window.LABEL_PADDING)
                                wind_distance_field = ui.FloatField()
                                wind_distance_field.model.set_value(helper._wind_distance)
                                delegate.set_wind_distance_field(wind_distance_field.model)

                    with ui.CollapsableFrame("Semantics Setup", collapsed=True):
                        with ui.VStack(height=0, spacing=10):
                            with ui.HStack(spacing=10): 
                                ui.Label("Grass Semantics", name="label", width=window.LABEL_PADDING)
                                grass_semantics_enabled_checkbox = ui.CheckBox()
                                grass_semantics_enabled_checkbox.model.set_value(helper._grass_semantics_enabled)
                                delegate.set_grass_semantics_enabled_checkbox(grass_semantics_enabled_checkbox.model)
                            with ui.HStack(spacing=10): 
                                ui.Label("Trees and Bushes Semantics", name="label", width=window.LABEL_PADDING)
                                trees_semantics_enabled_checkbox = ui.CheckBox()
                                trees_semantics_enabled_checkbox.model.set_value(helper._trees_semantics_enabled)
                                delegate.set_trees_semantics_enabled_checkbox(trees_semantics_enabled_checkbox.model)
                            with ui.HStack(spacing=10): 
                                ui.Label("Mines Semantics", name="label", width=window.LABEL_PADDING)
                                mines_semantics_enabled_checkbox = ui.CheckBox()
                                mines_semantics_enabled_checkbox.model.set_value(helper._mines_semantics_enabled)
                                delegate.set_mines_semantics_enabled_checkbox(mines_semantics_enabled_checkbox.model)
                    ui.Spacer(height=1)
                    ui.Separator()

                    video_list = [tree for tree in available_trees if tree != "Tree_4K_New_2" and tree != "Tree_Oak_Scarlet"]

                    video_list.append("Bush_Fountain_Grass")
                    video_list.append("Bush_Pampas_Grass")

                    # TEST: MULTISEASONAL SETUP
                    PRESETS_OBS = {
                        "All Flora": available_trees + available_bushes,
                        "Trees Only": available_trees,
                        "Bushes Only": available_bushes,
                        "Thermal Ready": ["Tree_4K_New", "Tree_Fraxinus", "Tree_Yellow_Pine", "Bush_Fountain_Grass", "Bush_Pampas_Grass"],
                        "Video": video_list,
                        "Summer": ["Tree_Oak_Scarlet", "Tree_Oak_Shumard", "Tree_Common_Apple", "Tree_Fraxinus", "Tree_Largetooth_Aspen", "Tree_Lombardy_Poplar", "Tree_Red_Ash"],
                        "Autumn": ["Tree_Black_Oak_Fall", "Tree_Gray_Birch_fall", "Tree_Lombardy_Poplar_Fall", "Tree_Scarlet_Oak_fall", "Tree_Shumard_Oak_Fall", "Tree_Sycamore", "Bush_Fountain_Grass", "Bush_Pampas_Grass"],
                        "Winter": ["Tree_Fraxinus_winter", "Tree_Largetooth_Aspen_winter", "Tree_Lombardy_Poplar_winter", "Tree_Red_Ash_winter", "Tree_Scarlet_Oak_bbc77_winter", "Tree_Shumard_Oak_a354d_winter", "Tree_Sycamore_winter"],
                        "Multiseasonal": ["Tree_Big_Pine", "Tree_Yellow_Pine", "Tree_Oak_Shumard", "Tree_Lombardy_Poplar", "Tree_Red_Ash", "Tree_Fraxinus", "Tree_Largetooth_Aspen", "Tree_Common_Apple", "Bush_Fountain_Grass", "Bush_Pampas_Grass"],
                        "Clear All": []
                    }

                    obs_preset_names = list(PRESETS_OBS.keys()) + ["Custom"]
                    obs_custom_index = len(obs_preset_names) - 1
                    obs_default_index = obs_preset_names.index("Multiseasonal")

                    window._obs_cb_models = {}
                    window._updating_obs_ui = False

                    helper.bush_checkbox_models = {}

                    def on_obs_checkbox_changed(model, o_type, is_tree):
                        if is_tree:
                            delegate.set_tree_type_status(o_type, model)

                        if window._updating_obs_ui:
                            return
                        window._updating_obs_ui = True

                        current_active = set(o for o, mod in window._obs_cb_models.items() if mod.as_bool)
                        matched_idx = obs_custom_index

                        for i, p_name in enumerate(list(PRESETS_OBS.keys())):
                            if set(PRESETS_OBS[p_name]) == current_active:
                                matched_idx = i
                                break

                        obs_preset_combo.model.get_item_value_model().set_value(matched_idx)
                        window._updating_obs_ui = False

                    def on_obs_preset_changed(model, item):
                        if window._updating_obs_ui:
                            return

                        idx = model.get_item_value_model().as_int
                        if idx == obs_custom_index:
                            return

                        window._updating_obs_ui = True
                        selected_preset = obs_preset_names[idx]
                        obs_in_preset = PRESETS_OBS[selected_preset]

                        for o_type, cb_model in window._obs_cb_models.items():
                            is_active = o_type in obs_in_preset
                            if cb_model.as_bool != is_active:
                                cb_model.set_value(is_active)
                                if o_type in available_trees:
                                    delegate.set_tree_type_status(o_type, cb_model)

                        window._updating_obs_ui = False

                    with ui.HStack():
                        ui.Label("Flora Presets", name="header", width=window.LABEL_PADDING, style={"font_weight": "bold"})
                        obs_preset_combo = ui.ComboBox(obs_default_index, *obs_preset_names)
                        obs_preset_combo.model.add_item_changed_fn(on_obs_preset_changed)

                    ui.Spacer(height=2)
                    # ui.Label("Active Obstacle Types", name="header", alignment=ui.Alignment.LEFT, height=2, style={"font_weight": "bold"})
                    with ui.CollapsableFrame("Active Obstacle Types", collapsed=True):
                        with ui.HStack(spacing=20):
                            with ui.VStack(spacing=5):
                                ui.Label("Trees", alignment=ui.Alignment.LEFT, style={"color": 0xFFAAAAAA, "font_weight": "bold"})
                                for m_type in available_trees:
                                    with ui.HStack():
                                        cb = ui.CheckBox(width=20)
                                        window._obs_cb_models[m_type] = cb.model

                                        is_active = m_type in PRESETS_OBS[obs_preset_names[obs_default_index]]
                                        cb.model.set_value(is_active)

                                        delegate.set_tree_type_status(m_type, cb.model)

                                        cb.model.add_value_changed_fn(
                                            lambda m, mt=m_type: on_obs_checkbox_changed(m, mt, is_tree=True)
                                        )

                                        ui.Label(m_type, name="label", alignment=ui.Alignment.LEFT)

                            with ui.VStack(spacing=5):
                                ui.Label("Bushes", alignment=ui.Alignment.LEFT, style={"color": 0xFFAAAAAA, "font_weight": "bold"})

                                for b_type in available_bushes:
                                    with ui.HStack():
                                        cb = ui.CheckBox(width=20)
                                        window._obs_cb_models[b_type] = cb.model

                                        is_active = b_type in PRESETS_OBS[obs_preset_names[obs_default_index]]
                                        cb.model.set_value(is_active)

                                        helper.bush_checkbox_models[b_type] = cb.model

                                        cb.model.add_value_changed_fn(
                                            lambda m, mt=b_type: on_obs_checkbox_changed(m, mt, is_tree=False)
                                        )

                                        ui.Label(b_type.replace("Bush_", ""), name="label", alignment=ui.Alignment.LEFT)

            
            # ui.Spacer(height=1)

            mines_frame = ui.CollapsableFrame("Mines Setup", collapsed=True)
            widgets_to_disable.append(mines_frame)
            with mines_frame:
                available_mines = helper.get_available_mine_types()

                MINE_TYPE_PRESETS = {
                    "All Mines": available_mines,
                    "TM-62 Only": [m for m in available_mines if "TM62" in m and "TM62_Thermal" not in m],
                    "Dataset": [m for m in available_mines if not any(x in m for x in ["TM83", "Dzika", "Cloth", "MOB"])],
                    "Magnetometer Mines" : ["Mine_TM62", "Mine_TM62D", "Mine_TM62P", "Mine_PMN", "Mine_PFM1"],
                    "Anti-Tank Only": [m for m in available_mines if ("TM" in m and "TM62_Thermal" not in m) or "PT" in m],
                    "Anti-Personnel Only": [m for m in available_mines
                                            if "PMN" in m
                                            or "PFM" in m
                                            or "PMD" in m
                                            or "POM" in m
                                            or "OZM" in m
                                            or "MON" in m],
                    "Clear All": []
                }
                mine_type_preset_names = list(MINE_TYPE_PRESETS.keys()) + ["Custom"]
                mine_type_custom_index = len(mine_type_preset_names) - 1

                mine_type_default_index = mine_type_preset_names.index("Dataset")

                window._mine_cb_models = {}
                window._updating_ui = False

                with ui.VStack(spacing=5):
                    SPAWN_PRESETS = {
                        "Default": {
                            "count": 50,
                            "distance": 3,
                            "min_spawn": [15, -40, 0],
                            "max_spawn": [190, 40, 0]
                        },
                        "Dataset": {
                            "count": 300,
                            "distance": 1,
                            "min_spawn": [15, -90, 0],
                            "max_spawn": [190, 90, 0]
                        },
                        "Survey": {
                            "count": 150,
                            "distance": 3,
                            "min_spawn": [15, -80, 0],
                            "max_spawn": [190, 80, 0]
                        },
                        "Magnetometer": {
                            "count": 10,
                            "distance": 3,
                            "min_spawn": [15, -3, 0],
                            "max_spawn": [190, 3, 0]
                        }
                    }
                    spawn_preset_names = list(SPAWN_PRESETS.keys()) + ["Custom"]
                    spawn_custom_index = len(spawn_preset_names) - 1

                    with ui.HStack():
                        ui.Label("Spawn Presets", name="header", width=window.LABEL_PADDING, style={"font_weight": "bold"})
                        spawn_preset_combo = ui.ComboBox(spawn_preset_names.index("Dataset"), *spawn_preset_names)

                    window._updating_spawn_ui = False

                    def on_spawn_preset_changed(model, item):
                        if window._updating_spawn_ui:
                            return
                        idx = model.get_item_value_model().as_int
                        if idx == spawn_custom_index:
                            return
                        window._updating_spawn_ui = True
                        selected = spawn_preset_names[idx]
                        config = SPAWN_PRESETS[selected]
                        mines_count_field.model.set_value(config["count"])
                        mines_distance_field.model.set_value(config["distance"])
                        for i, v in enumerate(config["min_spawn"]):
                            window.min_mines_box_models[i].set_value(v)
                        for i, v in enumerate(config["max_spawn"]):
                            window.max_mines_box_models[i].set_value(v)
                        window._updating_spawn_ui = False

                    def on_spawn_field_changed(*_):
                        if window._updating_spawn_ui:
                            return
                        spawn_preset_combo.model.get_item_value_model().set_value(spawn_custom_index)

                    ui.Spacer(height=2)
                    ui.Separator()
                    ui.Spacer(height=2)
                    
                    with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                        with ui.HStack():
                            ui.Label("Count", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                            mines_count_field = ui.IntField()
                            mines_count_field.model.set_value(helper._mines_count)
                            delegate.set_mines_count_field(mines_count_field.model)

                        with ui.HStack():
                            ui.Label("Distance (m)", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                            mines_distance_field = ui.IntField()
                            mines_distance_field.model.set_value(helper._mines_distance)
                            delegate.set_mines_distance_field(mines_distance_field.model)

                        coords_builder(window, "Min Spawn", window.min_mines_box_models, all_axis, colors, min_box_mines_default_values)
                        coords_builder(window, "Max Spawn", window.max_mines_box_models, all_axis, colors, max_box_mines_default_values)

                        default_spawn_idx = spawn_preset_combo.model.get_item_value_model().as_int
                        if default_spawn_idx != spawn_custom_index:
                            config = SPAWN_PRESETS[spawn_preset_names[default_spawn_idx]]
                            mines_count_field.model.set_value(config["count"])
                            mines_distance_field.model.set_value(config["distance"])
                            for i, v in enumerate(config["min_spawn"]):
                                window.min_mines_box_models[i].set_value(v)
                            for i, v in enumerate(config["max_spawn"]):
                                window.max_mines_box_models[i].set_value(v)

                        ui.Spacer(height=1)
                        ui.Separator()

                        mines_count_field.model.add_value_changed_fn(on_spawn_field_changed)
                        mines_distance_field.model.add_value_changed_fn(on_spawn_field_changed)
                        for m in window.min_mines_box_models:
                            m.add_value_changed_fn(on_spawn_field_changed)
                        for m in window.max_mines_box_models:
                            m.add_value_changed_fn(on_spawn_field_changed)

                        spawn_preset_combo.model.add_item_changed_fn(on_spawn_preset_changed)

                        with ui.HStack():
                            ui.Label("Mine Type Presets", name="header", width=window.LABEL_PADDING, style={"font_weight": "bold"})
                            mine_type_preset_combo = ui.ComboBox(mine_type_default_index, *mine_type_preset_names)

                        ui.Spacer(height=2)
                        ui.Label("Active Mine Types", name="header", alignment=ui.Alignment.LEFT, height=2, style={"font_weight": "bold"})

                        columns = [[], [], []]
                        for idx, m_type in enumerate(available_mines):
                            columns[idx % 3].append(m_type)

                        def on_checkbox_changed(model, m_type):
                            delegate.set_mine_type_status(m_type, model)

                            if window._updating_ui:
                                return
                            window._updating_ui = True

                            current_active = set(m for m, mod in window._mine_cb_models.items() if mod.as_bool)
                            matched_idx = mine_type_custom_index

                            for i, p_name in enumerate(list(MINE_TYPE_PRESETS.keys())):
                                if set(MINE_TYPE_PRESETS[p_name]) == current_active:
                                    matched_idx = i
                                    break

                            mine_type_preset_combo.model.get_item_value_model().set_value(matched_idx)
                            window._updating_ui = False

                        def on_preset_changed(model, item):
                            if window._updating_ui:
                                return

                            idx = model.get_item_value_model().as_int

                            if idx == mine_type_custom_index:
                                return

                            window._updating_ui = True
                            selected_preset_name = mine_type_preset_names[idx]
                            mines_in_preset = MINE_TYPE_PRESETS[selected_preset_name]

                            for m_type, cb_model in window._mine_cb_models.items():
                                is_active = m_type in mines_in_preset
                                if cb_model.as_bool != is_active:
                                    cb_model.set_value(is_active)
                                    delegate.set_mine_type_status(m_type, cb_model)

                            window._updating_ui = False

                        mine_type_preset_combo.model.add_item_changed_fn(on_preset_changed)

                        with ui.HStack(spacing=10):
                            for col_items in columns:
                                with ui.VStack(spacing=5):
                                    for m_type in col_items:
                                        with ui.HStack():
                                            cb = ui.CheckBox(width=20)
                                            window._mine_cb_models[m_type] = cb.model
                                            is_active = m_type in MINE_TYPE_PRESETS["Dataset"]
                                            cb.model.set_value(is_active)

                                            delegate.set_mine_type_status(m_type, cb.model)
                                            cb.model.add_value_changed_fn(
                                                lambda m, mt=m_type: on_checkbox_changed(m, mt)
                                            )

                                            ui.Label(m_type, name="label", alignment=ui.Alignment.LEFT)

                        ui.Spacer(height=2)
                        ui.Separator()

                        with ui.HStack(spacing=10):
                            ui.Label("Vegetation on Mines", name="label", width=window.LABEL_PADDING)
                            vegetation_enabled_checkbox = ui.CheckBox()
                            vegetation_enabled_checkbox.model.set_value(helper._vegetation_enabled)
                            delegate.set_vegetation_enabled_checkbox(vegetation_enabled_checkbox.model)

                        with ui.HStack(spacing=10):
                            ui.Label("Random Rotation", name="label", width=window.LABEL_PADDING)
                            vegetation_random_rotation_checkbox = ui.CheckBox()
                            vegetation_random_rotation_checkbox.model.set_value(helper._vegetation_random_rotation)
                            delegate.set_vegetation_random_rotation_checkbox(vegetation_random_rotation_checkbox.model)

                        vegetation_type_names = list(MINES_VEGETATION_TYPES.keys())

                        with ui.HStack(spacing=10):
                            ui.Label("Vegetation Type", name="label", width=window.LABEL_PADDING)
                            vegetation_type_combo = ui.ComboBox(0, *vegetation_type_names)
                            delegate.set_vegetation_type_combo(vegetation_type_combo)

                        def on_vegetation_type_changed(model, item):
                            idx = model.get_item_value_model().as_int
                            if 0 <= idx < len(vegetation_type_names):
                                delegate._mission_helper.mines_helper.swap_vegetation_type(vegetation_type_names[idx])

                        vegetation_type_combo.model.add_item_changed_fn(on_vegetation_type_changed)

            garbage_frame = ui.CollapsableFrame("Garbage Setup", collapsed=True)
            widgets_to_disable.append(garbage_frame)
            with garbage_frame:
                available_garbage = helper.get_available_garbage_types()

                GARBAGE_TYPE_PRESETS = {
                    "All Garbage": available_garbage,
                    "Default": [m for m in available_garbage if "Hedgehog" not in m and "Key" not in m],
                    "Footwear": [m for m in available_garbage if any(x in m for x in ["Boot", "Sandal"])],
                    "Containers": [m for m in available_garbage if any(x in m for x in ["Can", "Bucket"])],
                    "Wheels & Tyres": [m for m in available_garbage if any(x in m for x in ["Wheel", "Tyre"])],
                    "Other": [m for m in available_garbage
                              if not any(x in m for x in ["Boot", "Sandal", "Can", "Bucket", "Wheel", "Tyre"])],
                    "Clear All": []
                }
                garbage_type_preset_names = list(GARBAGE_TYPE_PRESETS.keys()) + ["Custom"]
                garbage_type_custom_index = len(garbage_type_preset_names) - 1

                garbage_type_default_index = garbage_type_preset_names.index("Default")

                window._garbage_cb_models = {}
                window._updating_garbage_ui = False

                with ui.VStack(spacing=5):
                    GARBAGE_SPAWN_PRESETS = {
                        "Default": {
                            "count": 50,
                            "distance": 3,
                            "min_spawn": [15, -40, 0],
                            "max_spawn": [190, 40, 0]
                        },
                        "Dataset": {
                            "count": 200,
                            "distance": 1,
                            "min_spawn": [15, -90, 0],
                            "max_spawn": [190, 90, 0]
                        },
                        "Survey": {
                            "count": 150,
                            "distance": 3,
                            "min_spawn": [15, -80, 0],
                            "max_spawn": [190, 80, 0]
                        },
                        "Roadside": {
                            "count": 30,
                            "distance": 5,
                            "min_spawn": [15, -5, 0],
                            "max_spawn": [190, 5, 0]
                        }
                    }
                    garbage_spawn_preset_names = list(GARBAGE_SPAWN_PRESETS.keys()) + ["Custom"]
                    garbage_spawn_custom_index = len(garbage_spawn_preset_names) - 1

                    with ui.HStack():
                        ui.Label("Spawn Presets", name="header", width=window.LABEL_PADDING, style={"font_weight": "bold"})
                        garbage_spawn_preset_combo = ui.ComboBox(garbage_spawn_preset_names.index("Dataset"), *garbage_spawn_preset_names)

                    window._updating_garbage_spawn_ui = False

                    def on_garbage_spawn_preset_changed(model, item):
                        if window._updating_garbage_spawn_ui:
                            return
                        idx = model.get_item_value_model().as_int
                        if idx == garbage_spawn_custom_index:
                            return
                        window._updating_garbage_spawn_ui = True
                        selected = garbage_spawn_preset_names[idx]
                        config = GARBAGE_SPAWN_PRESETS[selected]
                        garbage_count_field.model.set_value(config["count"])
                        garbage_distance_field.model.set_value(config["distance"])
                        for i, v in enumerate(config["min_spawn"]):
                            window.min_garbage_box_models[i].set_value(v)
                        for i, v in enumerate(config["max_spawn"]):
                            window.max_garbage_box_models[i].set_value(v)
                        window._updating_garbage_spawn_ui = False

                    def on_garbage_spawn_field_changed(*_):
                        if window._updating_garbage_spawn_ui:
                            return
                        garbage_spawn_preset_combo.model.get_item_value_model().set_value(garbage_spawn_custom_index)

                    ui.Spacer(height=2)
                    ui.Separator()
                    ui.Spacer(height=2)

                    with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                        with ui.HStack():
                            ui.Label("Count", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                            garbage_count_field = ui.IntField()
                            garbage_count_field.model.set_value(helper._garbage_count)
                            delegate.set_garbage_count_field(garbage_count_field.model)

                        with ui.HStack():
                            ui.Label("Distance (m)", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                            garbage_distance_field = ui.IntField()
                            garbage_distance_field.model.set_value(helper._garbage_distance)
                            delegate.set_garbage_distance_field(garbage_distance_field.model)

                        coords_builder(window, "Min Spawn", window.min_garbage_box_models, all_axis, colors, min_box_garbage_default_values)
                        coords_builder(window, "Max Spawn", window.max_garbage_box_models, all_axis, colors, max_box_garbage_default_values)

                        default_spawn_idx = garbage_spawn_preset_combo.model.get_item_value_model().as_int
                        if default_spawn_idx != garbage_spawn_custom_index:
                            config = GARBAGE_SPAWN_PRESETS[garbage_spawn_preset_names[default_spawn_idx]]
                            garbage_count_field.model.set_value(config["count"])
                            garbage_distance_field.model.set_value(config["distance"])
                            for i, v in enumerate(config["min_spawn"]):
                                window.min_garbage_box_models[i].set_value(v)
                            for i, v in enumerate(config["max_spawn"]):
                                window.max_garbage_box_models[i].set_value(v)

                        ui.Spacer(height=1)
                        ui.Separator()

                        garbage_count_field.model.add_value_changed_fn(on_garbage_spawn_field_changed)
                        garbage_distance_field.model.add_value_changed_fn(on_garbage_spawn_field_changed)
                        for m in window.min_garbage_box_models:
                            m.add_value_changed_fn(on_garbage_spawn_field_changed)
                        for m in window.max_garbage_box_models:
                            m.add_value_changed_fn(on_garbage_spawn_field_changed)

                        garbage_spawn_preset_combo.model.add_item_changed_fn(on_garbage_spawn_preset_changed)

                        with ui.HStack():
                            ui.Label("Garbage Type Presets", name="header", width=window.LABEL_PADDING, style={"font_weight": "bold"})
                            garbage_type_preset_combo = ui.ComboBox(garbage_type_default_index, *garbage_type_preset_names)

                        ui.Spacer(height=2)
                        ui.Label("Active Garbage Types", name="header", alignment=ui.Alignment.LEFT, height=2, style={"font_weight": "bold"})

                        columns = [[], [], []]
                        for idx, g_type in enumerate(available_garbage):
                            columns[idx % 3].append(g_type)

                        def on_garbage_checkbox_changed(model, g_type):
                            delegate.set_garbage_type_status(g_type, model)

                            if window._updating_garbage_ui:
                                return
                            window._updating_garbage_ui = True

                            current_active = set(m for m, mod in window._garbage_cb_models.items() if mod.as_bool)
                            matched_idx = garbage_type_custom_index

                            for i, p_name in enumerate(list(GARBAGE_TYPE_PRESETS.keys())):
                                if set(GARBAGE_TYPE_PRESETS[p_name]) == current_active:
                                    matched_idx = i
                                    break

                            garbage_type_preset_combo.model.get_item_value_model().set_value(matched_idx)
                            window._updating_garbage_ui = False

                        def on_garbage_preset_changed(model, item):
                            if window._updating_garbage_ui:
                                return

                            idx = model.get_item_value_model().as_int

                            if idx == garbage_type_custom_index:
                                return

                            window._updating_garbage_ui = True
                            selected_preset_name = garbage_type_preset_names[idx]
                            garbage_in_preset = GARBAGE_TYPE_PRESETS[selected_preset_name]

                            for g_type, cb_model in window._garbage_cb_models.items():
                                is_active = g_type in garbage_in_preset
                                if cb_model.as_bool != is_active:
                                    cb_model.set_value(is_active)
                                    delegate.set_garbage_type_status(g_type, cb_model)

                            window._updating_garbage_ui = False

                        garbage_type_preset_combo.model.add_item_changed_fn(on_garbage_preset_changed)

                        with ui.HStack(spacing=10):
                            for col_items in columns:
                                with ui.VStack(spacing=5):
                                    for g_type in col_items:
                                        with ui.HStack():
                                            cb = ui.CheckBox(width=20)
                                            window._garbage_cb_models[g_type] = cb.model
                                            is_active = g_type in GARBAGE_TYPE_PRESETS["Default"]
                                            cb.model.set_value(is_active)

                                            delegate.set_garbage_type_status(g_type, cb.model)
                                            cb.model.add_value_changed_fn(
                                                lambda m, gt=g_type: on_garbage_checkbox_changed(m, gt)
                                            )

                                            ui.Label(g_type, name="label", alignment=ui.Alignment.LEFT)

            with ui.HStack():
                ui.Button(
                    "Stop",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_stop_mission,
                )
                ui.Button(
                    "Start",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_start_mission,
                )
                ui.Button(
                    "Toggle 250Hz Manual Step",
                    height=window.BUTTON_HEIGHT,
                    style=window.BUTTON_BASE_STYLE,
                    clicked_fn=delegate._mission_helper.toggle_250hz_loop
                )

            def update_ui_state(model):
                is_ultimate_enabled = model.get_value_as_bool()
                for widget in widgets_to_disable:
                    widget.enabled = not is_ultimate_enabled

            ultimate_mission_enabled_checkbox.model.add_value_changed_fn(update_ui_state)
            update_ui_state(ultimate_mission_enabled_checkbox.model)


def get_min_tag_box(window):
    if len(window.min_tag_box_models) != 3:
        return None

    min_tag_box = np.array([model.get_value_as_float() for model in window.min_tag_box_models])

    return min_tag_box


def get_max_tag_box(window):
    if len(window.max_tag_box_models) != 3:
        return None

    max_tag_box = np.array([model.get_value_as_float() for model in window.max_tag_box_models])

    return max_tag_box


def get_min_obstacle_box(window):
    if len(window.min_obstacle_box_models) != 3:
        return None

    min_obstacle_box = np.array([model.get_value_as_float() for model in window.min_obstacle_box_models])

    return min_obstacle_box


def get_max_obstacle_box(window):
    if len(window.max_obstacle_box_models) != 3:
        return None

    max_tag_box = np.array([model.get_value_as_float() for model in window.max_obstacle_box_models])

    return max_tag_box


def get_min_mines_box(window):
    if len(window.min_mines_box_models) != 3:
        return None

    min_obstacle_box = np.array([model.get_value_as_float() for model in window.min_mines_box_models])

    return min_obstacle_box


def get_max_mines_box(window):
    if len(window.max_mines_box_models) != 3:
        return None

    max_tag_box = np.array([model.get_value_as_float() for model in window.max_mines_box_models])

    return max_tag_box


def get_min_garbage_box(window):
    if len(window.min_garbage_box_models) != 3:
        return None

    min_garbage_box = np.array([model.get_value_as_float() for model in window.min_garbage_box_models])

    return min_garbage_box


def get_max_garbage_box(window):
    if len(window.max_garbage_box_models) != 3:
        return None

    max_garbage_box = np.array([model.get_value_as_float() for model in window.max_garbage_box_models])

    return max_garbage_box


def get_drag_value(window):
    if len(window.drag_value_models) != 3:
        return None

    drag_value = np.array([model.get_value_as_float() for model in window.drag_value_models])

    return drag_value