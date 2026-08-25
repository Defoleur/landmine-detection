from __future__ import annotations
from typing import TYPE_CHECKING
import json
import os

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper


class MissionSettings:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def on_save_mission_settings(
        self,
        seed, mission_type,
        obstacles_count, obstacles_distance,
        mission_count,
        mines_count, mines_distance,
        garbage_count=0, garbage_distance=0
    ):
        if "Obstacle" not in mission_type:
            obstacles_count = 0
            obstacles_distance = 0

        settings = {
            "seed": seed,
            "mission_type": mission_type,
            "obstacles_count": obstacles_count,
            "obstacles_distance": obstacles_distance,
            "missions_count": mission_count,
            "mines_count": mines_count,
            "mines_distance": mines_distance,
            "garbage_count": garbage_count,
            "garbage_distance": garbage_distance
        }

        file_path = os.path.join(self.helper.script_dir, self.helper._mission_settings_path)
        with open(file_path, "w") as f:
            json.dump(settings, f, indent=4)
