from __future__ import annotations
from typing import TYPE_CHECKING
import json
import os
import numpy as np
import carb
import omni.ui as ui

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper

from pegasus.simulator.logic.dynamics import LinearDrag


class MissionUI:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def set_mission_status_label(self, label):
        self.helper._mission_status_label = label

    def set_mission_settings_label(self, label):
        self.helper._mission_settings_label = label

    def set_current_seed_label(self, label):
        self.helper._current_seed_label = label

    def set_current_mission_count_label(self, label):
        self.helper._current_mission_count_label = label

    def set_current_drag_value_label(self, label):
        self.helper._current_drag_value_label = label

    def set_tuning_original_log_path_field(self, field: ui.AbstractValueModel):
        self.helper._tuning_original_log_path_field = field

    def set_tuning_motor_log_path_field(self, field: ui.AbstractValueModel):
        self.helper._tuning_motor_log_path_field = field

    def set_tuning_max_iterations_field(self, field: ui.AbstractValueModel):
        self.helper._tuning_max_iterations_field = field

    def set_tuning_threshold_field(self, field: ui.AbstractValueModel):
        self.helper._tuning_threshold_field = field

    def set_tuning_learning_rate_field(self, field: ui.AbstractValueModel):
        self.helper._tuning_learning_rate_field = field

    def set_tuning_cooldown_field(self, field: ui.AbstractValueModel):
        self.helper._tuning_cooldown_field = field

    def set_tuning_use_dtw_checkbox(self, checkbox: ui.AbstractValueModel):
        self.helper._tuning_use_dtw_checkbox = checkbox

    def set_tuning_current_params_label(self, label):
        self.helper._tuning_current_params_label = label

    def set_tuning_best_params_label(self, label):
        self.helper._tuning_best_params_label = label

    def set_tuning_best_error_label(self, label):
        self.helper._tuning_best_error_label = label

    def set_tuning_live_params_label(self, label):
        self.helper._tuning_live_params_label = label

    def set_tuning_current_error_label(self, label):
        self.helper._tuning_current_error_label = label

    def set_tuning_perturbation_indicator_label(self, label):
        self.helper._tuning_perturbation_indicator_label = label

    def set_tuning_allow_negative_checkbox(self, checkbox: ui.AbstractValueModel):
        self.helper._tuning_allow_negative_checkbox = checkbox

    def update_drag_value(self):
        helper = self.helper
        if helper._drag_custom_value_checkbox.get_value_as_bool():
            drag = helper._window.get_drag_value()
        elif helper._drag_calibration_checkbox.get_value_as_bool():
            drag = helper.drag_test_list[helper.drag_test_index]
            helper.drag_test_index += 1

            if helper.drag_test_index >= len(helper.drag_test_list):
                helper.drag_test_index = 0
        else:
            drag = [0.1, 0.1, 0.0]

        helper.multirotors[0]._drag = LinearDrag(drag)

        if helper._current_drag_value_label:
            helper._current_drag_value_label.text = f"{drag}" if len(drag) != 0 else "No value provided!"

    def update_mission_status(self, running: bool):
        if self.helper._mission_status_label:
            self.helper._mission_status_label.text = "Running" if running else "Stopped"

    def update_current_mission_count(self, running: bool):
        if self.helper._current_mission_count_label:
            self.helper._current_mission_count_label.text = (
                f"{self.helper.current_mission_count} of {self.helper._missions_count}" if running else "0"
            )

    def update_current_seed(self):
        if self.helper._current_seed_label:
            self.helper._current_seed_label.text = str(self.helper._seed)

    def update_mission_settings_label(self):
        helper = self.helper

        obstacles_spawn_bounds = helper.current_ultimate_mission["map_setup"]["obstacles_spawn_bounds"]
        obstacles_count = helper.current_ultimate_mission["map_setup"]["obstacles_count"]
        tag_pos = helper.current_ultimate_mission["map_setup"]["tag_position"]
        formation_type = helper.current_ultimate_mission["formation"]["formation_type"]
        formation_distance = helper.current_ultimate_mission["formation"]["formation_distance"]
        obstacles_distance = helper.current_ultimate_mission["map_setup"]["obstacles_distances"]

        status_data = {
            "obstacles_count": obstacles_count,
            "obstacles_spawn_bounds": obstacles_spawn_bounds,
            "tag_position": tag_pos,
            "formation_type": formation_type,
            "formation_distance": formation_distance
        }

        if helper._mission_settings_label:
            helper._mission_settings_label.text = (
                f"Maps Count: {helper.current_ultimate_mission_map + 1} of {helper.ultimate_mission_maps}\n"
                f"Missions Count: {helper.current_ultimate_mission_count + 1} of {helper.ultimate_mission_count}\n"
                f"Obstacles Count: {status_data['obstacles_count']}\n"
                f"Obstacles Spawn Bounds: {status_data['obstacles_spawn_bounds']}\n"
                f"Obstacles Distance: {obstacles_distance}\n"
                f"Tag Position: {status_data['tag_position']}\n"
                f"Formation Type: {status_data['formation_type']}\n"
                f"Formation Distance: {status_data['formation_distance']}"
            )

        status_file_path = "../config/mission_status.json"
        file_path = os.path.join(helper.script_dir, status_file_path)

        try:
            with open(file_path, "w") as f:
                json.dump(status_data, f, indent=4)
        except Exception as e:
            carb.log_warn(f"Could not save mission status: {e}")
