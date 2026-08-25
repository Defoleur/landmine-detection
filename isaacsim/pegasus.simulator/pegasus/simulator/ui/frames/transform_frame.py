"""
File: transform_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Position and Orientation frame for the UI
"""

__all__ = ["build_transform_frame"]

import omni.ui as ui


def build_transform_frame(window, delegate):
    """
    Builds the Position and Orientation frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    components = ["Position", "Rotation"]
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    default_values = [0.0, 0.0, 0.26]

    with ui.CollapsableFrame("Position and Orientation"):
        with ui.VStack(spacing=8):

            ui.Spacer(height=0)

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
                            window._vehicle_transform_models.append(float_drag.model)
                            ui.Circle(name="transform", width=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)
            ui.Spacer(height=0)