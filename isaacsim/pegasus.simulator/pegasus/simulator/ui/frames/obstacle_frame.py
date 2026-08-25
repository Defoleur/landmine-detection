"""
File: obstacle_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Obstacles Settings frame for the UI
"""

__all__ = ["build_obstacle_frame", "get_selected_obstacle_attitude"]

import numpy as np
import omni.ui as ui

from pegasus.simulator.params import OBSTACLES, OBSTACLE_THUMBNAIL


def build_obstacle_frame(window, delegate):
    """
    Builds the Obstacles Settings frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    components = ["Position", "Rotation"]
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    default_values = [10.0, 0.0, 0.0]

    with ui.CollapsableFrame("Obstacles Settings"):
        with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
            ui.Spacer(height=window.GENERAL_SPACING)
            with ui.HStack():
                with ui.ZStack(width=window.LABEL_PADDING, height=window.BUTTON_HEIGHT * 2):
                    ui.Rectangle()
                    ui.Image(
                        OBSTACLE_THUMBNAIL,
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.CENTER,
                    )
                ui.Spacer(width=10)
                with ui.VStack(spacing=10):
                    with ui.VStack():
                        with ui.HStack():
                            ui.Label("Obstacle Type", name="label", width=window.LABEL_PADDING,
                                     alignment=ui.Alignment.TOP)

                            dropdown_menu = ui.ComboBox(0, name="obstacles")
                            for obstacle in OBSTACLES:
                                dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(obstacle))
                            delegate.set_obstacle_dropdown(dropdown_menu.model)

            for component in components:
                with ui.HStack():
                    with ui.HStack():
                        ui.Label(component, name="transform", width=50)
                        ui.Spacer()
                    for axis, default_value in zip(all_axis, default_values):
                        with ui.HStack():
                            with ui.ZStack(width=15):
                                ui.Rectangle(
                                    width=15,
                                    height=20,
                                    style={
                                        "background_color": colors[axis],
                                        "border_radius": 3,
                                        "corner_flag": ui.CornerFlag.LEFT,
                                    },
                                )
                                ui.Label(axis, name="transform_label", alignment=ui.Alignment.CENTER)
                            if component == "Position":
                                float_drag = ui.FloatDrag(name="transform", min=-1000000, max=1000000, step=0.01)
                                float_drag.model.set_value(default_value)
                            else:
                                float_drag = ui.FloatDrag(name="transform", min=-180.0, max=180.0, step=0.01)

                            window._obstacle_transform_models.append(float_drag.model)
                            ui.Circle(name="transform", width=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)
            ui.Spacer(height=0)

            with ui.HStack():
                ui.Button(
                    "Load Obstacle",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_load_obstacle,
                )


def get_selected_obstacle_attitude(window):
    """
    Returns the selected obstacle position and orientation.

    Args:
        window: The main WidgetWindow instance

    Returns:
        tuple: (position, orientation) as numpy arrays, or (None, None) if not set
    """
    if len(window._obstacle_transform_models) == 6:
        obstacle_pos = np.array([window._obstacle_transform_models[i].get_value_as_float() for i in range(3)])
        obstacle_orientation = np.array(
            [window._obstacle_transform_models[i].get_value_as_float() for i in range(3, 6)]
        )
        return obstacle_pos, obstacle_orientation

    return None, None