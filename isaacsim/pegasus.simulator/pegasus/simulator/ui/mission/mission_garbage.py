from __future__ import annotations
from typing import TYPE_CHECKING
import json
import os
import random
import asyncio
import carb
import omni.kit.commands
import omni.usd
import omni.physx
from pxr import Usd, UsdGeom

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper

from pegasus.simulator.params import GARBAGE
from pegasus.simulator.logic.obstacles import Obstacle
from pegasus.simulator.ui.image_map_helper import ImageMapHelper


class MissionGarbage:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def set_garbage_type_enabled(self, garbage_type, is_enabled):
        if is_enabled:
            if garbage_type not in self.helper._selected_garbage_types:
                self.helper._selected_garbage_types.append(garbage_type)
        else:
            if garbage_type in self.helper._selected_garbage_types:
                self.helper._selected_garbage_types.remove(garbage_type)

    def get_selected_garbage_types(self):
        return self.helper._selected_garbage_types

    def get_available_garbage_types(self):
        return self.helper._available_garbage_types

    def on_parse_garbage(self):
        stage = omni.usd.get_context().get_stage()
        garbage_positions = {}
        file_path = os.path.join(self.helper.script_dir, self.helper._garbage_path)

        for prim in stage.Traverse():
            if "obstacle_garbage" in prim.GetName().lower():
                xform = UsdGeom.Xformable(prim)
                local_pos = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        local_pos = op.Get()
                        break

                if local_pos is not None:
                    world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = world_matrix.ExtractTranslation()
                    garbage_positions[prim.GetPath().pathString] = [translation[0], translation[1], translation[2]]

        with open(file_path, "w") as f:
            json.dump(garbage_positions, f, indent=4)

    def get_garbage_names(self, garbage_count):
        active_types = self.helper._selected_garbage_types
        num_types = len(active_types)

        if num_types == 0:
            print("Warning: No garbage types selected in UI! Skipping generation.")
            return []

        base_count = garbage_count // num_types
        remainder = garbage_count % num_types

        garbage_names = []

        for i, garbage_type in enumerate(active_types):
            count_for_this_type = base_count + (1 if i < remainder else 0)

            for j in range(count_for_this_type):
                garbage_names.append({
                    "garbage_type": garbage_type,
                    "stage_prefix": f"obstacle_{garbage_type.lower()}_{j}",
                })

        return garbage_names

    def on_load_garbage(self):
        async def async_load_garbage():
            helper = self.helper
            rng = random.Random(helper._seed)

            garbage_count = int(helper._garbage_count_field.get_value_as_string())
            garbage_distance = float(helper._garbage_distance_field.get_value_as_string())

            if helper.ultimate_mission_check:
                box = helper.current_ultimate_mission["map_setup"]["obstacles_spawn_bounds"]
            else:
                box = [helper._window.get_min_garbage_box(), helper._window.get_max_garbage_box()]

            if helper._alt_spawn_enabled_checkbox.get_value_as_bool():
                cdf_data = ImageMapHelper.build_probability_cdf(helper.mines_map, helper.map_bounds, box)
            else:
                cdf_data = None

            garbage_names = self.get_garbage_names(garbage_count)
            total_count = len(garbage_names)

            if total_count == 0:
                return

            mine_positions = []
            for mine in helper._mines:
                mine_positions.append([mine.pos[0], mine.pos[1]])

            if len(helper._garbage) == 0:
                for i in range(total_count):
                    current_name = garbage_names[i]["stage_prefix"]
                    current_type = garbage_names[i]["garbage_type"]

                    obstacle = Obstacle(
                        f"/World/layout/garbage/{current_name}",
                        GARBAGE[current_type],
                        [1000, 1000, 0]
                    )

                    helper._garbage.append(obstacle)
            else:
                garbage_paths = []
                for garbage in helper._garbage:
                    garbage_paths.append(garbage._prim.GetPath())

                if helper._deactivation_enabled_checkbox.get_value_as_bool():
                    omni.kit.commands.execute(
                        'ToggleActivePrims',
                        prim_paths=garbage_paths,
                        active=True
                    )

            physx_qi = omni.physx.get_physx_scene_query_interface()

            garbage_positions = []
            for i in range(len(helper._garbage)):
                tries = 1000
                current_try = 0

                while current_try < tries:
                    collision = False

                    if cdf_data:
                        pos = ImageMapHelper.generate_position_from_cdf(cdf_data, [0, 0], rng=rng)
                    else:
                        pos = self.helper.generate_random_position(box, rng=rng)

                    current_garbage_point = [pos[0], pos[1]]

                    for garbage in garbage_positions:
                        if self.helper.get_distance(garbage, current_garbage_point) < garbage_distance:
                            collision = True
                            break

                    if not collision:
                        for mine in mine_positions:
                            if self.helper.get_distance(mine, current_garbage_point) < garbage_distance:
                                collision = True
                                break

                    if not collision:
                        origin = carb.Float3(pos[0], pos[1], 10000.0)
                        direction = carb.Float3(0.0, 0.0, -1.0)

                        hit_info = physx_qi.raycast_closest(origin, direction, 20000.0)

                        if hit_info["hit"]:
                            pos[2] = hit_info["position"][2]
                        else:
                            pos[2] = 0.0
                        break

                    current_try += 1

                if current_try >= tries:
                    carb.log_warn("Cannot spawn all garbage! Breaking...")
                    break

                garbage_positions.append(current_garbage_point)

                helper._garbage[i].set_pos(pos)

            helper._garbage_positions = garbage_positions

            if (
                helper._alt_spawn_enabled_checkbox.get_value_as_bool()
                and helper._save_spawn_enabled_checkbox.get_value_as_bool()
            ):
                from datetime import datetime
                current_time = datetime.now().isoformat()
                ImageMapHelper.save_spawn_visualization(
                    garbage_positions,
                    helper.mines_map_image,
                    helper.map_bounds,
                    f"{helper.maps_debug_folder}/debug_garbage_spawn_{current_time}.png"
                )

            self.on_parse_garbage()

        asyncio.ensure_future(async_load_garbage())
