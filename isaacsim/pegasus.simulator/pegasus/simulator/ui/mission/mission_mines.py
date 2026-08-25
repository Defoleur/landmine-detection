from __future__ import annotations
from typing import TYPE_CHECKING
import json
import os
import random
import asyncio
import carb
import numpy as np
from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf
import omni.usd
import omni.physx
from isaacsim.core.utils.semantics import upgrade_prim_semantics_to_labels, add_labels
import omni.kit.commands

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper

from pegasus.simulator.params import (
    OBSTACLES,
    MINES_VEGETATION_TYPES,
    MINES_VEGETATION_BY_SEASON,
    MINES_VEGETATION_ROAD_ELEVATION_DELTA,
    MINES_VEGETATION_ROAD_PROBE_DELTA,
)
from pegasus.simulator.logic.obstacles import Obstacle
from pegasus.simulator.helpers.environment_manager import EnvironmentManager
from pegasus.simulator.ui.image_map_helper import ImageMapHelper
from scipy.spatial.transform import Rotation


class MissionMines:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def set_mine_type_enabled(self, mine_type, is_enabled):
        if is_enabled:
            if mine_type not in self.helper._selected_mine_types:
                self.helper._selected_mine_types.append(mine_type)
        else:
            if mine_type in self.helper._selected_mine_types:
                self.helper._selected_mine_types.remove(mine_type)

    def get_selected_mine_types(self):
        return self.helper._selected_mine_types

    def get_available_mine_types(self):
        return self.helper._available_mine_types

    def _get_mine_label(self, prim):
        prim_name = prim.GetName().lower()
        matches = [m for m in self.helper._available_mine_types if m.lower() in prim_name]
        if not matches:
            return "mine"
        best = max(matches, key=len)
        return best[len("Mine_"):] if best.startswith("Mine_") else best

    def set_tree_type_enabled(self, tree_type, is_enabled):
        if is_enabled:
            if tree_type not in self.helper._selected_tree_types:
                self.helper._selected_tree_types.append(tree_type)
        else:
            if tree_type in self.helper._selected_tree_types:
                self.helper._selected_tree_types.remove(tree_type)

    def get_selected_tree_types(self):
        return self.helper._selected_tree_types

    def get_available_tree_types(self):
        return self.helper._available_tree_types

    def get_vegetation_type(self):
        helper = self.helper
        combo = helper._vegetation_type_combo
        type_names = list(MINES_VEGETATION_TYPES.keys())
        if combo is None:
            return type_names[0]
        idx = combo.model.get_item_value_model().as_int
        if 0 <= idx < len(type_names):
            return type_names[idx]
        return type_names[0]

    def is_mine_on_road(self, mine_pos):
        """True if the mine sits on an elevated surface (e.g. a road deck).

        The probe ray starts just below the mine's surface, so it passes under an
        elevated road deck and hits the true terrain underneath. On solid ground
        the ray starts inside the terrain (backface-culled) and reports no hit.
        This handles uneven terrain since it queries the exact same (x, y) column.
        """
        qi = omni.physx.get_physx_scene_query_interface()
        hit = qi.raycast_closest(
            carb.Float3(mine_pos[0], mine_pos[1], mine_pos[2] - MINES_VEGETATION_ROAD_PROBE_DELTA),
            carb.Float3(0.0, 0.0, -1.0),
            20000.0,
        )
        if not hit["hit"]:
            return False
        return mine_pos[2] - hit["position"][2] > MINES_VEGETATION_ROAD_ELEVATION_DELTA

    def _create_vegetation(self, index, grass_pos, rng, veg_config):
        orientation = [1.0, 0.0, 0.0, 0.0]
        if self.helper._vegetation_random_rotation_checkbox.get_value_as_bool():
            yaw = rng.uniform(0, 360)
            rot_quat = Rotation.from_euler("z", yaw, degrees=True).as_quat()
            orientation = [rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]]
        return Obstacle(
            f"/World/layout/vegetation/grass_{index}",
            veg_config["path"],
            grass_pos,
            init_orientation=orientation,
        )

    def swap_vegetation_type(self, type_name):
        helper = self.helper
        config = MINES_VEGETATION_TYPES.get(type_name)
        if config is None:
            carb.log_warn(f"Unknown vegetation type: {type_name}")
            return

        for i, grass in enumerate(helper._vegetation):
            if grass is None:
                continue
            if i < len(helper._mines):
                mine_pos = helper._mines[i].pos
                grass_pos = [mine_pos[0] + config["x"], mine_pos[1] + config["y"], mine_pos[2] + config["z"]]
                grass.set_pos(grass_pos)

            prim = grass._prim
            if prim and prim.IsValid():
                prim.GetReferences().ClearReferences()
                prim.GetReferences().AddReference(config["path"])

        carb.log_info(f"Swapped mines vegetation to: {type_name}")

    def update_vegetation_for_season(self, season):
        helper = self.helper
        helper._current_season = season

        target_type = MINES_VEGETATION_BY_SEASON.get(season)
        if target_type not in MINES_VEGETATION_TYPES:
            target_type = list(MINES_VEGETATION_TYPES.keys())[0]

        combo = helper._vegetation_type_combo
        if combo is not None:
            type_names = list(MINES_VEGETATION_TYPES.keys())
            idx = type_names.index(target_type)
            current_idx = combo.model.get_item_value_model().as_int
            if current_idx != idx:
                combo.model.get_item_value_model().set_value(idx)
                return

        if len(helper._vegetation) > 0:
            self.swap_vegetation_type(target_type)

    def on_parse_mines(self):
        stage = omni.usd.get_context().get_stage()
        mine_positions = {}
        file_path = os.path.join(self.helper.script_dir, self.helper._mines_path)

        for prim in stage.Traverse():
            if prim.GetPath().pathString.startswith("/World/layout/vegetation/"):
                upgrade_prim_semantics_to_labels(prim)
                add_labels(prim, ["vegetation"])

            if "obstacle_mine" in prim.GetName().lower():
                if self.helper._mines_semantics_enabled_checkbox.get_value_as_bool():
                    upgrade_prim_semantics_to_labels(prim)
                    add_labels(prim, [self._get_mine_label(prim)])
                xform = UsdGeom.Xformable(prim)
                local_pos = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        local_pos = op.Get()
                        break

                if local_pos is not None:
                    world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = world_matrix.ExtractTranslation()
                    mine_positions[prim.GetPath().pathString] = [translation[0], translation[1], translation[2]]

        with open(file_path, "w") as f:
            json.dump(mine_positions, f, indent=4)

    def get_mines_names(self, mines_count):
        active_types = self.helper._selected_mine_types
        num_types = len(active_types)

        if num_types == 0:
            print("Warning: No mine types selected in UI! Skipping generation.")
            return []

        base_count = mines_count // num_types
        remainder = mines_count % num_types

        mines_names = []

        for i, mine_type in enumerate(active_types):
            count_for_this_type = base_count + (1 if i < remainder else 0)

            for j in range(count_for_this_type):
                mines_names.append({
                    "mine_type": mine_type,
                    "stage_prefix": f"obstacle_{mine_type.lower()}_{j}"
                })

        return mines_names

    def on_load_mines(self):
        async def async_load_mines():
            helper = self.helper
            rng = random.Random(helper._seed)
            thermal_enabled = helper._thermal_textures_enabled_checkbox.get_value_as_bool()

            mines_count = int(helper._mines_count_field.get_value_as_string())
            mines_distance = int(helper._mines_distance_field.get_value_as_string())

            if helper.ultimate_mission_check:
                box = helper.current_ultimate_mission["map_setup"]["obstacles_spawn_bounds"]
            else:
                box = [helper._window.get_min_mines_box(), helper._window.get_max_mines_box()]

            if helper._alt_spawn_enabled_checkbox.get_value_as_bool():
                cdf_data = ImageMapHelper.build_probability_cdf(helper.mines_map, helper.map_bounds, box)
            else:
                cdf_data = None

            mines_names = self.get_mines_names(mines_count)

            if len(helper._mines) == 0:
                if thermal_enabled:
                    stage = omni.usd.get_context().get_stage()

                    material_path = "/World/layout/mines/tm_62_thermal/_materials/Material_002"
                    material_prim = stage.GetPrimAtPath(material_path)
                    master_material = UsdShade.Material(material_prim)
                    carb.log_warn(f"Material was found: {master_material}")

                for i in range(mines_count):
                    if not thermal_enabled:
                        current_name = mines_names[i]["stage_prefix"]
                        current_type = mines_names[i]["mine_type"]
                    else:
                        current_type = "Mine_TM62_Thermal"
                        current_name = "obstacle_mine_tm_62_thermal"

                    obstacle = Obstacle(
                        f"/World/layout/mines/{current_name}",
                        OBSTACLES[current_type],
                        [1000, 1000, 0]
                    )

                    helper._mines.append(obstacle)

                    if thermal_enabled:
                        mine_prim = stage.GetPrimAtPath(f"{obstacle._stage_prefix}/xform_0/mesh_0")
                        if mine_prim.IsValid() and master_material:
                            carb.log_warn(f"/World/{obstacle._stage_prefix} was found. Binding material...")
                            UsdShade.MaterialBindingAPI(mine_prim).Bind(master_material)
            else:
                mines_paths = []
                for mine in helper._mines:
                    mines_paths.append(mine._prim.GetPath())

                if helper._deactivation_enabled_checkbox.get_value_as_bool():
                    omni.kit.commands.execute(
                        'ToggleActivePrims',
                        prim_paths=mines_paths,
                        active=True
                    )
            
            physx_qi = omni.physx.get_physx_scene_query_interface()

            obstacles_positions = []
            for i in range(len(helper._mines)):
                tries = 1000
                current_try = 0
                while current_try < tries:
                    collision = False

                    if helper.ultimate_mission_check:
                        box = helper.current_ultimate_mission["map_setup"]["obstacles_spawn_bounds"]

                    if cdf_data:
                        pos = ImageMapHelper.generate_position_from_cdf(cdf_data, [0, 0], rng=rng)
                    else:
                        pos = self.helper.generate_random_position(box, rng=rng)

                    current_obstacle_point = [pos[0], pos[1]]

                    for obstacle in obstacles_positions:
                        if self.helper.get_distance(obstacle, current_obstacle_point) < mines_distance:
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
                    carb.log_warn("Cannot spawn all mines! Breaking...")
                    break

                obstacles_positions.append(current_obstacle_point)

                helper._mines[i].set_pos(pos)

            if (
                helper._alt_spawn_enabled_checkbox.get_value_as_bool()
                and helper._save_spawn_enabled_checkbox.get_value_as_bool()
            ):
                from datetime import datetime
                current_time = datetime.now().isoformat()
                ImageMapHelper.save_spawn_visualization(
                    obstacles_positions,
                    helper.mines_map_image,
                    helper.map_bounds,
                    f"{helper.maps_debug_folder}/debug_mines_spawn_{current_time}.png"
                )

            helper._obstacles_positions = obstacles_positions

            if helper._vegetation_enabled_checkbox.get_value_as_bool():
                vegetation_paths = []
                veg_type = self.get_vegetation_type()
                veg_config = MINES_VEGETATION_TYPES[veg_type]

                for i in range(len(helper._mines)):
                    on_road = self.is_mine_on_road(helper._mines[i].pos)
                    grass_pos = [
                        helper._mines[i].pos[0] + veg_config["x"],
                        helper._mines[i].pos[1] + veg_config["y"],
                        helper._mines[i].pos[2] + veg_config["z"],
                    ]

                    if i >= len(helper._vegetation):
                        if on_road:
                            helper._vegetation.append(None)
                            continue
                        helper._vegetation.append(self._create_vegetation(i, grass_pos, rng, veg_config))
                    else:
                        if helper._vegetation[i] is None:
                            if not on_road:
                                # Mine moved off the road since the last load -> spawn vegetation now.
                                helper._vegetation[i] = self._create_vegetation(i, grass_pos, rng, veg_config)
                            continue

                        if on_road:
                            # Mine moved onto the road since the last load -> remove stale vegetation.
                            omni.kit.commands.execute(
                                'DeletePrims', paths=[helper._vegetation[i]._prim.GetPath()]
                            )
                            helper._vegetation[i] = None
                            continue

                        if helper._deactivation_enabled_checkbox.get_value_as_bool():
                            vegetation_paths.append(helper._vegetation[i]._prim.GetPath())
                        helper._vegetation[i].set_pos(grass_pos)

                if (
                    helper._deactivation_enabled_checkbox.get_value_as_bool()
                    and len(vegetation_paths) > 0
                ):
                    omni.kit.commands.execute(
                        'ToggleActivePrims',
                        prim_paths=vegetation_paths,
                        active=True
                    )
            elif (
                helper._deactivation_enabled_checkbox.get_value_as_bool()
                and len(helper._vegetation) > 0
            ):
                vegetation_paths = [
                    grass._prim.GetPath() for grass in helper._vegetation if grass is not None
                ]
                omni.kit.commands.execute(
                    'ToggleActivePrims',
                    prim_paths=vegetation_paths,
                    active=False
                )

            self.on_parse_mines()

            try:
                # Mines are fully positioned here; record their fresh spawn
                # positions as the lift origin, then apply the season offset
                # on top of that.
                for mine in helper._mines:
                    prim = mine._prim
                    if prim is not None and prim.IsValid():
                        prim.CreateAttribute(
                            EnvironmentManager.MINE_ORIGIN_ATTR,
                            Sdf.ValueTypeNames.Vector3d,
                        ).Set(Gf.Vec3d(mine.pos[0], mine.pos[1], mine.pos[2]))

                current_season = getattr(self.helper, "_current_season", None) or "summer"
                self.helper._get_env_manager().lift_mines(current_season)
            except Exception as e:
                carb.log_warn(f"Lifting mines for season failed: {e}")

        asyncio.ensure_future(async_load_mines())
