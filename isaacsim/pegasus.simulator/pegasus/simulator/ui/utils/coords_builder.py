"""
File: coords_builder.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Shared utility for building coordinate input widgets in the UI
"""

__all__ = ["coords_builder"]

import omni.ui as ui


def coords_builder(window, name, position, all_axis, colors, default_values, min=-1000000, max=1000000):
    """
    Builds a row of coordinate inputs (X, Y, Z) with colored labels and FloatDrag widgets.

    Args:
        window: The main WidgetWindow instance (for accessing BUTTON_HEIGHT, LABEL_PADDING constants)
        name: Label name for the row
        position: List to append the FloatDrag models to (for later value retrieval)
        all_axis: List of axis labels (e.g., ["X", "Y", "Z"])
        colors: Dictionary mapping axis to color value
        default_values: List of default float values for each axis
        min: Minimum value for FloatDrag
        max: Maximum value for FloatDrag
    """
    with ui.HStack():
        with ui.HStack():
            ui.Label(name, name="transform", width=50)
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

                float_drag = ui.FloatDrag(name="transform", min=min, max=max, step=0.01)
                float_drag.model.set_value(default_value)

                position.append(float_drag.model)
                ui.Circle(name="transform", width=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)