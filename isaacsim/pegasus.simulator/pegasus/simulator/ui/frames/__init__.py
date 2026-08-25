# Copyright (c) 2023, Marcelo Jacinto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .backend_frame import build_backend_frame
from .scene_frame import build_scene_frame
from .robot_frame import build_robot_frame, get_selected_vehicle_attitude, get_vehicles_spacing
from .apriltag_frame import build_apriltag_frame, get_selected_tag_attitude, get_tag_velocities
from .mission_frame import (
    build_mission_frame,
    get_min_tag_box,
    get_max_tag_box,
    get_min_obstacle_box,
    get_max_obstacle_box,
    get_min_mines_box,
    get_max_mines_box,
    get_min_garbage_box,
    get_max_garbage_box,
    get_drag_value,
)
from .wind_frame import build_wind_frame
from .texture_frame import build_texture_frame
from .obstacle_frame import build_obstacle_frame, get_selected_obstacle_attitude
from .parser_frame import build_parser_frame
from .camera_frame import build_camera_frame, get_selected_camera_pos
from .transform_frame import build_transform_frame
from .extension_frame import build_extension_frame
from .environment_frame import build_environment_frame
from .daytime_frame import build_daytime_frame
from .schedule_frame import build_schedule_frame
from .road_frame import build_road_frame

__all__ = [
    "build_backend_frame",
    "build_scene_frame",
    "build_robot_frame",
    "get_selected_vehicle_attitude",
    "get_vehicles_spacing",
    "build_apriltag_frame",
    "get_selected_tag_attitude",
    "get_tag_velocities",
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
    "build_wind_frame",
    "build_texture_frame",
    "build_obstacle_frame",
    "get_selected_obstacle_attitude",
    "build_parser_frame",
    "build_camera_frame",
    "get_selected_camera_pos",
    "build_transform_frame",
    "build_extension_frame",
    "build_environment_frame",
    "build_daytime_frame",
    "build_schedule_frame",
    "build_road_frame",
]