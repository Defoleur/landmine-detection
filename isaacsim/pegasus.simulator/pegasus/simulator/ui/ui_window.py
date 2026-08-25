"""
File: ui_window.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Definition of WidgetWindow which contains all the UI code that defines the extension GUI
"""

__all__ = ["WidgetWindow"]

import carb
import omni.ui as ui
from omni.ui import color as cl
import omni.kit.app

from pegasus.simulator.ui.ui_delegate import UIDelegate
from pegasus.simulator.params import WINDOW_TITLE

from pegasus.simulator.ui.frames import (
    build_backend_frame,
    build_scene_frame,
    build_robot_frame,
    build_apriltag_frame,
    build_mission_frame,
    build_wind_frame,
    build_texture_frame,
    build_obstacle_frame,
    build_parser_frame,
    build_camera_frame,
    build_transform_frame,
    build_extension_frame,
    build_environment_frame,
    build_daytime_frame,
    build_schedule_frame,
    build_road_frame,
    get_selected_vehicle_attitude,
    get_vehicles_spacing,
    get_selected_tag_attitude,
    get_tag_velocities,
    get_selected_obstacle_attitude,
    get_selected_camera_pos,
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


class WidgetWindow(ui.Window):

    LABEL_PADDING = 120
    BUTTON_HEIGHT = 50
    GENERAL_SPACING = 5

    WINDOW_WIDTH = 325
    WINDOW_HEIGHT = 850

    BUTTON_SELECTED_STYLE = {
        "Button": {
            "background_color": cl("#3780ae"),
            "border_color": cl("#29587c"),
            "border_width": 2,
            "border_radius": 5,
            "padding": 2,
        }
    }

    BUTTON_BASE_STYLE = {
        "Button": {
            "background_color": cl("#292929"),
            "border_color": cl("#292929"),
            "border_width": 2,
            "border_radius": 5,
            "padding": 5,
        }
    }

    def __init__(self, delegate: UIDelegate, **kwargs):
        super().__init__(
            WINDOW_TITLE, width=WidgetWindow.WINDOW_WIDTH, height=WidgetWindow.WINDOW_HEIGHT, visible=True, **kwargs
        )

        self.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

        self._delegate = delegate
        self._delegate.set_window_bind(self)

        self._camera_transform_models = []
        self._vehicle_transform_models = []
        self._vehicles_spacing_models = []
        self._tag_transform_models = []
        self._tag_velocities_models = []

        self.min_tag_box_models = []
        self.max_tag_box_models = []

        self.min_obstacle_box_models = []
        self.max_obstacle_box_models = []

        self.min_mines_box_models = []
        self.max_mines_box_models = []

        self.min_garbage_box_models = []
        self.max_garbage_box_models = []

        self.drag_value_models = []

        self._obstacle_transform_models = []

        self.timeline = omni.timeline.get_timeline_interface()
        update_stream = omni.kit.app.get_app().get_update_event_stream()
        self.sub = update_stream.create_subscription_to_pop(self.on_update, name="My Subscription Name")

        self._physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_physics_step)

        self._vehicle_created = False

        self._build_window()

    def on_update(self, e: carb.events.IEvent):
        if self.timeline.is_playing() and self._vehicle_created is False:
            try:
                self._delegate.on_save_mission_settings(0, "Static", 0, 0, 1, 0, 0)
                self._delegate.on_parse_obstacles()
                self._vehicle_created = True
            except Exception as e:
                carb.log_warn(e)

    # INFO: IF "PLAY"
    def on_physics_step(self, dt: float):
        if self.timeline.is_playing() and self._vehicle_created:
            self._delegate.move_cycle()
            self._delegate.mission_cycle()
            self._delegate.check_mission_end()

    def destroy(self):
        self._delegate.on_clear_scene()
        self._vehicle_created = False
        super().destroy()

    def _build_window(self):
        with self.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON, vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                with ui.VStack():
                    build_backend_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_scene_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_robot_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_apriltag_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_mission_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_environment_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_road_frame(self, self._delegate)
                    ui.Spacer(height=5)                    

                    build_schedule_frame(self, self._delegate)
                    ui.Spacer(height=5)                

                    build_daytime_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_extension_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_wind_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_texture_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_obstacle_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_parser_frame(self, self._delegate)
                    ui.Spacer(height=5)

                    build_camera_frame(self, self._delegate)
                    ui.Spacer()

    def get_selected_vehicle_attitude(self):
        return get_selected_vehicle_attitude(self)

    def get_vehicles_spacing(self):
        return get_vehicles_spacing(self)

    def get_selected_tag_attitude(self):
        return get_selected_tag_attitude(self)

    def get_tag_velocities(self):
        return get_tag_velocities(self)

    def get_selected_obstacle_attitude(self):
        return get_selected_obstacle_attitude(self)

    def get_selected_camera_pos(self):
        return get_selected_camera_pos(self)

    def get_min_tag_box(self):
        return get_min_tag_box(self)

    def get_max_tag_box(self):
        return get_max_tag_box(self)

    def get_min_obstacle_box(self):
        return get_min_obstacle_box(self)

    def get_max_obstacle_box(self):
        return get_max_obstacle_box(self)

    def get_min_mines_box(self):
        return get_min_mines_box(self)

    def get_max_mines_box(self):
        return get_max_mines_box(self)

    def get_min_garbage_box(self):
        return get_min_garbage_box(self)

    def get_max_garbage_box(self):
        return get_max_garbage_box(self)

    def get_drag_value(self):
        return get_drag_value(self)