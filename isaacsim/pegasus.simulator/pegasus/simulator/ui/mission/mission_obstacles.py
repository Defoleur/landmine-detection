from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import json
import os
import asyncio
import math
import random
import carb
import numpy as np
from scipy.spatial.transform import Rotation
from datetime import datetime
import omni.kit.commands
import omni.usd
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.semantics import upgrade_prim_semantics_to_labels, add_labels

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper

from pegasus.simulator.params import OBSTACLES, SPAWN_MAPS
from pegasus.simulator.logic.obstacles import Obstacle, obstacles_parameters
from pegasus.simulator.ui.image_map_helper import ImageMapHelper


class MissionObstacles:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def on_parse_obstacles(self):
        stage = omni.usd.get_context().get_stage()
        tree_positions = {}

        file_path_field = self.helper._obstacles_file_path_field
        rel_path = file_path_field.get_value_as_string() if file_path_field is not None else self.helper._obstacles_path
        file_path = os.path.join(self.helper.script_dir, rel_path)

        mesh_name_field = self.helper._obstacles_mesh_name_field
        mesh_name = mesh_name_field.get_value_as_string() if mesh_name_field is not None else self.helper._mesh_name

        for prim in stage.Traverse():
            if (
                mesh_name in prim.GetName().lower()
                and "mine" not in prim.GetName().lower()
                and "garbage" not in prim.GetName().lower()
            ):
                if self.helper._trees_semantics_enabled_checkbox.get_value_as_bool():
                    upgrade_prim_semantics_to_labels(prim)
                    if "tree" in prim.GetName().lower():
                        add_labels(prim, ["tree"])
                    else:
                        add_labels(prim, ["bush"])
                xform = UsdGeom.Xformable(prim)
                local_pos = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        local_pos = op.Get()
                        break

                if local_pos is not None:
                    world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = world_matrix.ExtractTranslation()
                    tree_positions[prim.GetPath().pathString] = [translation[0], translation[1], translation[2]]

            # if self.helper._grass_semantics_enabled_checkbox.get_value_as_bool():
            #     for grass_name in self.helper.grass_names:
            #         if grass_name in prim.GetName().lower() and (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
            #             upgrade_prim_semantics_to_labels(prim)
            #             add_labels(prim, ["grass"])
        with open(file_path, "w") as f:
            json.dump(tree_positions, f, indent=4)

    def get_distance(self, point1, point2):
        return math.dist(point1, point2)

    def generate_obstacles_names(
        self,
        trees_count, trees_dist,
        bushes_count, bushes_dist,
        rng=None
    ):
        if rng is None:
            rng = random
        generated_obstacles = []
        global_index = 0

        active_trees = sorted(self.helper._selected_tree_types)
        if active_trees and trees_count > 0:
            base_t_count = trees_count // len(active_trees)
            rem_t = trees_count % len(active_trees)

            if self.helper._random_scales_enabled_checkbox.get_value_as_bool():
                tree_scales = [k for k in obstacles_parameters.keys() if k != "obstacle_bush"]
            else:
                tree_scales = ["obstacle_default"]

            for i, obs_type in enumerate(active_trees):
                count = base_t_count + (1 if i < rem_t else 0)
                for _ in range(count):
                    random_scale = rng.choice(tree_scales)
                    generated_obstacles.append({
                        "obstacle_type": obs_type,
                        "parameter_name": random_scale,
                        "stage_prefix": f"{random_scale}_{obs_type}_{global_index}",
                        "required_distance": trees_dist,
                        "is_bush": False
                    })
                    global_index += 1

        enabled_bushes = [name for name, model in self.helper.bush_checkbox_models.items() if model.as_bool]
        if not enabled_bushes:
            enabled_bushes = [k for k in OBSTACLES.keys() if k.startswith("Bush_")]

        if enabled_bushes and bushes_count > 0:
            base_b_count = bushes_count // len(enabled_bushes)
            rem_b = bushes_count % len(enabled_bushes)

            for i, bush_type in enumerate(enabled_bushes):
                count = base_b_count + (1 if i < rem_b else 0)
                for _ in range(count):
                    generated_obstacles.append({
                        "obstacle_type": bush_type,
                        "parameter_name": "obstacle_bush",
                        "stage_prefix": f"obstacle_bush_{bush_type}_{global_index}",
                        "required_distance": bushes_dist,
                        "is_bush": True
                    })
                    global_index += 1

        rng.shuffle(generated_obstacles)

        return generated_obstacles

    def on_load_obstacles(self):
        helper = self.helper
        rng = random.Random(helper._seed)

        if helper.ultimate_mission_check:
            trees_count = helper.current_ultimate_mission["map_setup"]["obstacles_count"]
            trees_distance = helper.current_ultimate_mission["map_setup"]["obstacles_distances"]
            bushes_count = helper.current_ultimate_mission["map_setup"].get("bushes_count", 0)
            bushes_distance = helper.current_ultimate_mission["map_setup"].get("bushes_distances", 0)
            box = helper.current_ultimate_mission["map_setup"]["obstacles_spawn_bounds"]
        else:
            trees_count = int(helper._obstacles_count_field.get_value_as_string())
            trees_distance = float(helper._obstacles_distance_field.get_value_as_string())
            bushes_count = int(helper._bushes_count_field.get_value_as_string())
            bushes_distance = float(helper._bushes_distance_field.get_value_as_string())
            box = [helper._window.get_min_obstacle_box(), helper._window.get_max_obstacle_box()]

        if helper._alt_spawn_enabled_checkbox.get_value_as_bool():
            cdf_data = ImageMapHelper.build_probability_cdf(helper.trees_map, helper.map_bounds, box)
        else:
            cdf_data = None

        obstacles_names = self.generate_obstacles_names(
            trees_count, trees_distance,
            bushes_count, bushes_distance,
            rng=rng
        )
        total_count = len(obstacles_names)

        if len(helper._obstacles) == 0:
            stage = omni.usd.get_context().get_stage()

            for i in range(total_count):
                current_name = obstacles_names[i]["parameter_name"]
                obstacle_type = obstacles_names[i]["obstacle_type"]
                is_bush = obstacles_names[i]["is_bush"]
                stage_prefix = obstacles_names[i]["stage_prefix"]
                orientation = [0.0, 0.0, 0.0, 1.0]

                if helper._random_rotations_enabled_checkbox.get_value_as_bool():
                    yaw = rng.uniform(0, 360)
                    rot_quat = Rotation.from_euler("z", yaw, degrees=True).as_quat()
                    orientation = [rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]]

                if is_bush:
                    final_usd_path = OBSTACLES[obstacle_type]
                else:
                    base_usd_path = OBSTACLES[obstacle_type].replace("Trees_main", "Trees_main_scaled")
                    path_without_ext, ext = os.path.splitext(base_usd_path)
                    final_usd_path = f"{path_without_ext}_{current_name}{ext}"

                prim_path = f"/World/layout/trees/{stage_prefix}"

                obstacle = Obstacle(
                    prim_path,
                    final_usd_path,
                    [1000, 1000, 0],
                    init_scale=[1.0, 1.0, 1.0],
                    init_orientation=orientation
                )
                helper._obstacles.append(obstacle)
        else:
            obstacles_paths = []
            for obstacle in helper._obstacles:
                obstacles_paths.append(obstacle._prim.GetPath())

            if helper._deactivation_enabled_checkbox.get_value_as_bool():
                omni.kit.commands.execute(
                    'ToggleActivePrims',
                    prim_paths=obstacles_paths,
                    active=True
                )

        obstacles_positions = []
        physx_qi = omni.physx.get_physx_scene_query_interface()

        for i in range(len(helper._obstacles)):
            tries = 1000
            current_try = 0

            required_dist = obstacles_names[i]["required_distance"]

            while current_try < tries:
                collision = False

                if cdf_data:
                    pos = ImageMapHelper.generate_position_from_cdf(cdf_data, [0, 0], rng=rng)
                else:
                    pos = self.helper.generate_random_position(box, rng=rng)

                current_obstacle_point = [pos[0], pos[1]]

                for obstacle in obstacles_positions:
                    if self.get_distance(obstacle, current_obstacle_point) < required_dist:
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
                carb.log_warn("Cannot spawn all obstacles! Breaking...")
                break

            obstacles_positions.append(current_obstacle_point)

            if obstacles_names[i]["is_bush"]:
                pos[2] += obstacles_parameters["obstacle_bush"]["pos"][2]

            helper._obstacles[i].set_pos(pos)

        helper._obstacles_positions = obstacles_positions

        if (
            helper._alt_spawn_enabled_checkbox.get_value_as_bool()
            and helper._save_spawn_enabled_checkbox.get_value_as_bool()
        ):
            current_time = datetime.now().isoformat()
            ImageMapHelper.save_spawn_visualization(
                obstacles_positions,
                helper.trees_map_image,
                helper.map_bounds,
                f"{helper.maps_debug_folder}/debug_obstacles_spawn_{current_time}.png"
            )
        self.on_parse_obstacles()

    def on_load_obstacle(self):
        async def async_load_obstacle():
            helper = self.helper
            if helper._obstacle_dropdown is not None and helper._window is not None:
                obstacle_index = helper._obstacle_dropdown.get_item_value_model().as_int
                selected_obstacle = helper._obstacle_names[obstacle_index]

                pos, euler_angles = helper._window.get_selected_obstacle_attitude()

                Obstacle(
                    "/World/obstacle",
                    OBSTACLES[selected_obstacle],
                    pos,
                    Rotation.from_euler("XYZ", euler_angles, degrees=True).as_quat(),
                )

                carb.log_info("Spawned the obstacle: " + selected_obstacle + " using the Pegasus Simulator UI")
            else:
                carb.log_error("Could not spawn the obstacle using the Pegasus Simulator UI")

        asyncio.ensure_future(async_load_obstacle())

    def on_destroy_obstacles(self):
        async def async_destroy_obstacles():
            helper = self.helper
            paths_to_delete = []

            for obstacle in helper._obstacles:
                paths_to_delete.append(obstacle._prim.GetPath())

            for mine in helper._mines:
                paths_to_delete.append(mine._prim.GetPath())

            for garbage in helper._garbage:
                paths_to_delete.append(garbage._prim.GetPath())

            for grass in helper._vegetation:
                if grass is not None:
                    paths_to_delete.append(grass._prim.GetPath())

            if helper._deactivation_enabled_checkbox.get_value_as_bool():
                omni.kit.commands.execute(
                    'ToggleActivePrims',
                    prim_paths=paths_to_delete,
                    active=False
                )
            else:
                omni.kit.commands.execute('DeletePrims', paths=paths_to_delete)

                helper._obstacles_positions = []
                helper._obstacles = []

                helper._mines_positions = []
                helper._mines = []

                helper._garbage_positions = []
                helper._garbage = []

                helper._vegetation = []

        asyncio.ensure_future(async_destroy_obstacles())
