"""
File: scene_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Scene Selection frame for the UI
"""

__all__ = ["build_scene_frame"]

import omni.ui as ui

from pegasus.simulator.params import SIMULATION_ENVIRONMENTS, WORLD_THUMBNAIL


def build_scene_frame(window, delegate):
    """
    Builds the Scene Selection frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    with ui.CollapsableFrame("Scene Selection"):
        with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
            ui.Spacer(height=window.GENERAL_SPACING)

            with ui.HStack():
                ui.Label("World Assets", width=window.LABEL_PADDING, height=10.0)

                dropdown_menu = ui.ComboBox(0, height=10, name="environments")
                for environment in SIMULATION_ENVIRONMENTS:
                    dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(environment))

                delegate.set_scene_dropdown(dropdown_menu.model)

            ui.Spacer(height=0)

            with ui.CollapsableFrame("Geographic Coordinates", collapsed=False):
                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    with ui.HStack():
                        ui.Label("Latitude", name="label", width=window.LABEL_PADDING-50)
                        latitude_field = ui.FloatField(name="latitude", precision=6)
                        latitude_field.model.set_value(delegate._latitude)
                        delegate.set_latitude_field(latitude_field.model)
                        ui.Circle(name="transform", width=20, height=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)

                        ui.Label("Longitude", name="label", width=window.LABEL_PADDING-50)
                        longitude_field = ui.FloatField(name="longitude", precision=6)
                        longitude_field.model.set_value(delegate._longitude)
                        delegate.set_longitude_field(longitude_field.model)
                        ui.Circle(name="transform", width=20, height=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)

                        ui.Label("Altitude", name="label", width=window.LABEL_PADDING-50)
                        altitude_field = ui.FloatField(name="altitude", precision=6)
                        altitude_field.model.set_value(delegate._altitude)
                        delegate.set_altitude_field(altitude_field.model)
                        ui.Circle(name="transform", width=20, height=20, radius=3.5, size_policy=ui.CircleSizePolicy.FIXED)

                    with ui.HStack():
                        ui.Button("Set", enabled=True, clicked_fn=delegate.on_set_new_global_coordinates)
                        ui.Button("Reset", enabled=True, clicked_fn=delegate.on_reset_global_coordinates)
                        ui.Button("Make Default", enabled=True, clicked_fn=delegate.on_set_new_default_global_coordinates)

            ui.Spacer(height=0)

            with ui.HStack():
                with ui.ZStack(width=window.LABEL_PADDING * 1.5, height=window.BUTTON_HEIGHT * 3):
                    ui.Rectangle()
                    ui.Image(
                        WORLD_THUMBNAIL,
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.LEFT_CENTER,
                    )

                ui.Spacer(width=window.GENERAL_SPACING)

                with ui.VStack():
                    ui.Button(
                        "Load Scene",
                        height=window.BUTTON_HEIGHT,
                        clicked_fn=delegate.on_load_scene,
                        style=window.BUTTON_BASE_STYLE,
                    )

                    ui.Button(
                        "Load Ready Scene",
                        height=window.BUTTON_HEIGHT,
                        clicked_fn=delegate.on_load_ready_scene,
                        style=window.BUTTON_BASE_STYLE,
                    )

                    ui.Button(
                        "Clear Scene",
                        height=window.BUTTON_HEIGHT,
                        clicked_fn=delegate.on_clear_scene,
                        style=window.BUTTON_BASE_STYLE,
                    )