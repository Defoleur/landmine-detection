"""
File: apriltag_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Apriltag Settings frame for the UI
"""

__all__ = ["build_apriltag_frame", "get_selected_tag_attitude", "get_tag_velocities"]

import numpy as np
import omni.ui as ui

from pegasus.simulator.params import TAGS, TAG_THUMBNAIL


def build_apriltag_frame(window, delegate):
    """
    Builds the Apriltag Settings frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    components = ["Position", "Rotation"]
    all_axis = ["X", "Y", "Z"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371, "Z": 0xFFA07D4F}
    default_values = [212.0, 0, 5.38507]
    # default_values = [206.3, -33.0, 5.38507]

    all_movement_axis = ["VX", "VY", "VZ"]
    movement_colors = {"VX": 0xFF5555AA, "VY": 0xFF76A371, "VZ": 0xFFA07D4F}
    movement_default_values = [0.1, 0.1, 0.0]

    with ui.CollapsableFrame("Apriltag Settings", collapsed=True):
        with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
            ui.Spacer(height=window.GENERAL_SPACING)
            with ui.HStack():
                with ui.ZStack(width=window.LABEL_PADDING, height=window.BUTTON_HEIGHT * 2):
                    ui.Rectangle()
                    ui.Image(
                        TAG_THUMBNAIL,
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.CENTER,
                    )
                ui.Spacer(width=10)
                with ui.VStack(spacing=10):
                    with ui.VStack():
                        with ui.HStack():
                            ui.Label("Tag Number", name="label", width=window.LABEL_PADDING,
                                     alignment=ui.Alignment.TOP)

                            dropdown_menu = ui.ComboBox(0, name="tags")
                            for tag in TAGS:
                                dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(tag))
                            delegate.set_tag_dropdown(dropdown_menu.model)
                    with ui.HStack():
                        ui.Label("Tag ID", name="label", width=window.LABEL_PADDING,
                                 alignment=ui.Alignment.TOP)
                        tag_id_field = ui.IntField()
                        delegate.set_tag_id_field(tag_id_field.model)

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

                            window._tag_transform_models.append(float_drag.model)
                            ui.Circle(name="transform", width=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)
            ui.Spacer(height=0)

            with ui.HStack():
                ui.Button(
                    "Load Tag",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_load_tag,
                )
                ui.Button(
                    "Load Formation Tags",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_load_formation_tags,
                )
            ui.Spacer(height=0)
            with ui.CollapsableFrame("Movement", collapsed=False):
                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    with ui.HStack():
                        with ui.HStack():
                            ui.Label("Velocity", name="transform", width=50)
                            ui.Spacer()
                        for axis, default_value in zip(all_movement_axis, movement_default_values):
                            with ui.HStack():
                                with ui.ZStack(width=15):
                                    ui.Rectangle(
                                        width=15,
                                        height=20,
                                        style={
                                            "background_color": movement_colors[axis],
                                            "border_radius": 3,
                                            "corner_flag": ui.CornerFlag.LEFT,
                                        },
                                    )
                                    ui.Label(axis, name="transform_label", alignment=ui.Alignment.CENTER)

                                float_drag = ui.FloatDrag(name="transform", min=-1000000, max=1000000,
                                                          step=0.01)
                                float_drag.model.set_value(default_value)

                                window._tag_velocities_models.append(float_drag.model)
                                ui.Circle(name="transform", width=20, radius=3.5,
                                          size_policy=ui.CircleSizePolicy.FIXED)
                    with ui.HStack():
                        ui.Label("DVD Move", name="label", width=window.LABEL_PADDING - 20)
                        dvd_enabled_checkbox = ui.CheckBox()
                        dvd_enabled_checkbox.model.set_value(True)
                        delegate.set_dvd_enabled_checkbox(dvd_enabled_checkbox.model)
                    with ui.HStack():
                        ui.Button(
                            "Reset",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=delegate.on_reset_tag,
                        )
                        ui.Button(
                            "Pause",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=delegate.on_pause_tag,
                        )
                        ui.Button(
                            "Start",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=delegate.on_start_tag,
                        )


def get_selected_tag_attitude(window):
    """
    Returns the selected tag position and orientation.

    Args:
        window: The main WidgetWindow instance

    Returns:
        tuple: (position, orientation) as numpy arrays, or (None, None) if not set
    """
    if len(window._tag_transform_models) == 6:
        tag_pos = np.array([window._tag_transform_models[i].get_value_as_float() for i in range(3)])
        tag_orientation = np.array(
            [window._tag_transform_models[i].get_value_as_float() for i in range(3, 6)]
        )
        return tag_pos, tag_orientation

    return None, None


def get_tag_velocities(window):
    """
    Returns the tag velocity values.

    Args:
        window: The main WidgetWindow instance

    Returns:
        numpy array of velocity values, or None if not set
    """
    if len(window._tag_velocities_models) != 3:
        return None

    tag_velocities = np.array([model.get_value_as_float() for model in window._tag_velocities_models])

    return tag_velocities