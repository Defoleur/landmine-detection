"""
File: camera_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Viewport Camera frame for the UI
"""

__all__ = ["build_camera_frame", "get_selected_camera_pos"]

import numpy as np
import omni.ui as ui


def build_camera_frame(window, delegate):
    """
    Builds the Viewport Camera frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    default_values = [5.0, 5.0, 5.0]
    target_default_values = [0.0, 0.0, 0.0]

    with ui.CollapsableFrame("Viewport Camera"):
        with ui.VStack(spacing=8):
            ui.Spacer(height=0)

            with ui.HStack():
                with ui.HStack():
                    ui.Label("Position", name="transform", width=50, height=20)
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
                            ui.Label(axis, height=20, name="transform_label", alignment=ui.Alignment.CENTER)
                        float_drag = ui.FloatDrag(name="transform", min=-1000000, max=1000000, step=0.01)
                        float_drag.model.set_value(default_value)
                        window._camera_transform_models.append(float_drag.model)
                        ui.Circle(
                            name="transform", width=20, height=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED
                        )

            with ui.HStack():
                with ui.HStack():
                    ui.Label("Target", name="transform", width=50, height=20)
                    ui.Spacer()
                for axis, default_value in zip(all_axis, target_default_values):
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
                            ui.Label(axis, height=20, name="transform_label", alignment=ui.Alignment.CENTER)
                        float_drag = ui.FloatDrag(name="transform", min=-1000000, max=1000000, step=0.01)
                        float_drag.model.set_value(default_value)
                        window._camera_transform_models.append(float_drag.model)
                        ui.Circle(
                            name="transform", width=20, height=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED
                        )

            ui.Button(
                "Set Camera Pose",
                height=window.BUTTON_HEIGHT,
                clicked_fn=delegate.on_set_viewport_camera,
                style=window.BUTTON_BASE_STYLE,
            )
            ui.Spacer()


def get_selected_camera_pos(window):
    """
    Returns the selected camera position and target.

    Args:
        window: The main WidgetWindow instance

    Returns:
        tuple: (position, target) as numpy arrays, or (None, None) if not set
    """
    if len(window._camera_transform_models) == 6:
        camera_pos = np.array([window._camera_transform_models[i].get_value_as_float() for i in range(3)])
        camera_target = np.array([window._camera_transform_models[i].get_value_as_float() for i in range(3, 6)])
        return camera_pos, camera_target

    return None, None