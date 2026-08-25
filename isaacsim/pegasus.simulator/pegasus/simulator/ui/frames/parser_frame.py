"""
File: parser_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Obstacles Parser frame for the UI
"""

__all__ = ["build_parser_frame"]

import omni.ui as ui


def build_parser_frame(window, delegate):
    """
    Builds the Obstacles Parser frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    helper = delegate._mission_helper
    with ui.CollapsableFrame("Obstacles Parser"):
        with ui.VStack(spacing=10):
            with ui.HStack():
                ui.Label("File Path", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                obstacles_file_path_field = ui.StringField(name="obstacles_file_path")
                obstacles_file_path_field.model.set_value(helper._obstacles_path)
                delegate.set_obstacles_file_path_field(obstacles_file_path_field.model)
            with ui.HStack():
                ui.Label("Mesh Name", name="label", width=window.LABEL_PADDING, alignment=ui.Alignment.TOP)
                obstacles_mesh_name_field = ui.StringField(name="mesh_name")
                obstacles_mesh_name_field.model.set_value(helper._mesh_name)
                delegate.set_obstacles_mesh_name_field(obstacles_mesh_name_field.model)
            ui.Button(
                "Parse",
                height=window.BUTTON_HEIGHT,
                clicked_fn=delegate.on_parse_obstacles,
            )