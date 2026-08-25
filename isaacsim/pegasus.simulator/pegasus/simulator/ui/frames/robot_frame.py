"""
File: robot_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Vehicle Selection frame for the UI
"""

__all__ = ["build_robot_frame", "get_selected_vehicle_attitude", "get_vehicles_spacing"]

import numpy as np
import omni.ui as ui

from pegasus.simulator.params import ROBOTS, THUMBNAIL


def build_robot_frame(window, delegate):
    """
    Builds the Vehicle Selection frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    all_axis = ["X", "Y"]
    colors = {"X": 0xFF5555AA, "Y": 0xFF76A371}
    default_values = [0, 3]
    formations = ["Triangle"]

    with ui.CollapsableFrame(title="Vehicle Selection", collapsed=True):
        with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
            with ui.HStack():
                with ui.ZStack(width=window.LABEL_PADDING, height=window.BUTTON_HEIGHT * 2):
                    ui.Rectangle()
                    ui.Image(
                        THUMBNAIL,
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.CENTER,
                    )
                ui.Spacer(width=10)

                with ui.VStack():
                    with ui.HStack():
                        ui.Label("Vehicle Model", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)

                        dropdown_menu = ui.ComboBox(0, name="robots")
                        for robot in ROBOTS:
                            dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(robot))
                        delegate.set_vehicle_dropdown(dropdown_menu.model)

                    with ui.HStack():
                        ui.Label("Vehicle ID", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        vehicle_id_field = ui.IntField()
                        delegate.set_vehicle_id_field(vehicle_id_field.model)

            with ui.CollapsableFrame("Multiple Vehicles", collapsed=True):
                with ui.VStack(height=0, spacing=10):
                    with ui.HStack():
                        ui.Label("Config Path", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        ports_path_field = ui.StringField(name="ports_path")
                        ports_path_field.model.set_value(delegate._ports_path)
                        delegate.set_ports_path_field(ports_path_field.model)
                    with ui.HStack():
                        ui.Label("Vehicles Count", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                        vehicles_count_field = ui.IntField()
                        vehicles_count_field.model.set_value(delegate._vehicles_count)
                        delegate.set_vehicles_count_field(vehicles_count_field.model)
                    with ui.CollapsableFrame("Spacing", collapsed=True):
                        with ui.VStack(height=0):
                            with ui.HStack():
                                ui.Label("Spacing", name="label", width=window.LABEL_PADDING,
                                        alignment=ui.Alignment.TOP)
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

                                        float_drag = ui.FloatDrag(name="transform", min=-1000000, max=1000000,
                                                                step=0.01)
                                        float_drag.model.set_value(default_value)

                                        window._vehicles_spacing_models.append(float_drag.model)
                                        ui.Circle(name="transform", width=20, radius=3.5,
                                                size_policy=ui.CircleSizePolicy.FIXED)
                            ui.Button(
                                "Load Vehicles",
                                height=window.BUTTON_HEIGHT,
                                clicked_fn=delegate.on_load_vehicles,
                            )
            with ui.CollapsableFrame("Formation", collapsed=True):
                with ui.VStack(height=0):
                    with ui.HStack():
                        ui.Label("Formation Type", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)

                        dropdown_menu = ui.ComboBox(0, name="fornations")
                        for formation in formations:
                            dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(formation))
                        delegate.set_formations_dropdown(dropdown_menu.model)

                    ui.Button(
                        "Load Formation",
                        height=window.BUTTON_HEIGHT,
                        clicked_fn=delegate.on_load_formation,
                    )
            with ui.HStack():
                from pegasus.simulator.ui.frames.transform_frame import build_transform_frame
                build_transform_frame(window, delegate)

            with ui.HStack():
                ui.Button(
                    "Reset Vehicle",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_reset_vehicles,
                )
                ui.Button(
                    "Load Vehicle",
                    height=window.BUTTON_HEIGHT,
                    clicked_fn=delegate.on_load_vehicle,
                )


def get_selected_vehicle_attitude(window):
    """
    Returns the selected vehicle position and orientation.

    Args:
        window: The main WidgetWindow instance

    Returns:
        tuple: (position, orientation) as numpy arrays, or (None, None) if not set
    """
    if len(window._vehicle_transform_models) == 6:
        vehicle_pos = np.array([window._vehicle_transform_models[i].get_value_as_float() for i in range(3)])
        vehicle_orientation = np.array(
            [window._vehicle_transform_models[i].get_value_as_float() for i in range(3, 6)]
        )
        return vehicle_pos, vehicle_orientation

    return None, None


def get_vehicles_spacing(window):
    """
    Returns the vehicles spacing values.

    Args:
        window: The main WidgetWindow instance

    Returns:
        numpy array of spacing values, or None if not set
    """
    if len(window._vehicles_spacing_models) != 2:
        return None

    vehicles_spacing = np.array([model.get_value_as_float() for model in window._vehicles_spacing_models])

    return vehicles_spacing