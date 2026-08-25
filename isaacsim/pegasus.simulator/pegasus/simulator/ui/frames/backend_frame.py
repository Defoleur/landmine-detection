"""
File: backend_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Streaming Backend selection frame for the UI
"""

__all__ = ["build_backend_frame"]

import omni.ui as ui
from omni.ui import color as cl

from pegasus.simulator.params import BACKENDS, BACKENDS_THUMBMAILS


def build_backend_frame(window, delegate):
    """
    Builds the Streaming Backend selection frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """

    def handle_backend_switch(
        px4_button,
        ardupilot_button,
        ros2_button,
        betaflight_button,
        button,
        logo_image,
        px4_menu=None,
        ardupilot_menu=None,
        betaflight_menu=None,
    ):
        if button == BACKENDS["px4"]:
            px4_button.enabled = False
            ardupilot_button.enabled = True
            ros2_button.enabled = True
            betaflight_button.enabled = True

            px4_button.set_style(window.BUTTON_SELECTED_STYLE)
            ardupilot_button.set_style(window.BUTTON_BASE_STYLE)
            ros2_button.set_style(window.BUTTON_BASE_STYLE)
            betaflight_button.set_style(window.BUTTON_BASE_STYLE)

            px4_menu.enabled = True
            px4_menu.visible = True
            ardupilot_menu.enabled = False
            ardupilot_menu.visible = False
            betaflight_menu.enabled = False
            betaflight_menu.visible = False

        elif button == BACKENDS["betaflight"]:
            px4_button.enabled = True
            ardupilot_button.enabled = True
            ros2_button.enabled = True
            betaflight_button.enabled = False

            px4_button.set_style(window.BUTTON_BASE_STYLE)
            ardupilot_button.set_style(window.BUTTON_BASE_STYLE)
            ros2_button.set_style(window.BUTTON_BASE_STYLE)
            betaflight_button.set_style(window.BUTTON_SELECTED_STYLE)

            px4_menu.enabled = False
            px4_menu.visible = False
            ardupilot_menu.enabled = False
            ardupilot_menu.visible = False
            betaflight_menu.enabled = True
            betaflight_menu.visible = True

        elif button == BACKENDS["ardupilot"]:
            px4_button.enabled = True
            ardupilot_button.enabled = False
            ros2_button.enabled = True
            betaflight_button.enabled = True

            px4_button.set_style(window.BUTTON_BASE_STYLE)
            ardupilot_button.set_style(window.BUTTON_SELECTED_STYLE)
            ros2_button.set_style(window.BUTTON_BASE_STYLE)
            betaflight_button.set_style(window.BUTTON_BASE_STYLE)

            px4_menu.enabled = False
            px4_menu.visible = False
            ardupilot_menu.enabled = True
            ardupilot_menu.visible = True
            betaflight_menu.enabled = False
            betaflight_menu.visible = False
        else:
            px4_button.enabled = True
            ardupilot_button.enabled = True
            ros2_button.enabled = False
            betaflight_button.enabled = True

            px4_button.set_style(window.BUTTON_BASE_STYLE)
            ardupilot_button.set_style(window.BUTTON_BASE_STYLE)
            ros2_button.set_style(window.BUTTON_SELECTED_STYLE)
            betaflight_button.set_style(window.BUTTON_BASE_STYLE)

            px4_menu.enabled = False
            px4_menu.visible = False
            ardupilot_menu.enabled = False
            ardupilot_menu.visible = False
            betaflight_menu.enabled = False
            betaflight_menu.visible = False

        logo_image.source_url = BACKENDS_THUMBMAILS[button]
        delegate.set_streaming_backend(button)

    with ui.CollapsableFrame(title="Streaming Backend"):
        ui.Spacer(height=0)
        with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
            ui.Spacer(height=window.GENERAL_SPACING)

            with ui.HStack():
                with ui.ZStack(width=window.LABEL_PADDING):
                    ui.Rectangle(
                        alignment=ui.Alignment.CENTER,
                        width=200,
                        height=window.BUTTON_HEIGHT * 3,
                    )
                    logo_image = ui.Image(
                        BACKENDS_THUMBMAILS["betaflight"],
                        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                        alignment=ui.Alignment.CENTER,
                    )

                with ui.VStack():
                    betaflight_button = ui.Button(
                        "Betaflight",
                        height=window.BUTTON_HEIGHT * 0.75,
                        style=window.BUTTON_SELECTED_STYLE,
                        enabled=True,
                        visible=True
                    )
                    px4_button = ui.Button(
                        "PX4",
                        height=window.BUTTON_HEIGHT * 0.75,
                        style=window.BUTTON_BASE_STYLE,
                        enabled=True,
                        visible=True
                    )
                    ardupilot_button = ui.Button(
                        "ArduPilot",
                        height=window.BUTTON_HEIGHT * 0.75,
                        style=window.BUTTON_BASE_STYLE,
                        enabled=True,
                        visible=True
                    )
                    ros2_button = ui.Button(
                        "ROS 2",
                        height=window.BUTTON_HEIGHT * 0.75,
                        style=window.BUTTON_BASE_STYLE,
                        enabled=True,
                        visible=True
                    )

            px4_menu = ui.CollapsableFrame("PX4 Configurations", collapsed=False)
            ardupilot_menu = ui.CollapsableFrame("Ardupilot Configurations", collapsed=False)
            betaflight_menu = ui.CollapsableFrame("Betaflight Configurations", collapsed=False)

            px4_button.set_clicked_fn(lambda: handle_backend_switch(
                px4_button, ardupilot_button, ros2_button, betaflight_button, BACKENDS["px4"], logo_image, px4_menu, ardupilot_menu, betaflight_menu)
            )
            ardupilot_button.set_clicked_fn(lambda: handle_backend_switch(
                px4_button, ardupilot_button, ros2_button, betaflight_button, BACKENDS["ardupilot"], logo_image, px4_menu, ardupilot_menu, betaflight_menu)
            )
            ros2_button.set_clicked_fn(lambda: handle_backend_switch(
                px4_button, ardupilot_button, ros2_button, betaflight_button, BACKENDS["ros2"], logo_image, px4_menu, ardupilot_menu, betaflight_menu)
            )
            betaflight_button.set_clicked_fn(lambda: handle_backend_switch(
                px4_button, ardupilot_button, ros2_button, betaflight_button, BACKENDS["betaflight"], logo_image, px4_menu, ardupilot_menu, betaflight_menu)
            )

            with px4_menu:
                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    ui.Spacer(height=window.GENERAL_SPACING)
                    with ui.HStack():
                        ui.Label("Auto-launch PX4", name="label", width=window.LABEL_PADDING - 20)
                        px4_checkbox = ui.CheckBox()
                        px4_checkbox.model.set_value(delegate._autostart_px4)
                        delegate.set_px4_autostart_checkbox(px4_checkbox.model)

                    with ui.HStack():
                        ui.Label("PX4 Path", name="label", width=window.LABEL_PADDING - 20)
                        px4_path_field = ui.StringField(name="px4_path", width=300)
                        px4_path_field.model.set_value(delegate._px4_dir)
                        delegate.set_px4_directory_field(px4_path_field.model)

                        ui.Button("Reset", enabled=True, clicked_fn=delegate.on_reset_px4_path)
                        ui.Button("Make Default", enabled=True, clicked_fn=delegate.on_set_new_default_px4_path)

                    with ui.HStack():
                        ui.Label("PX4 airframe", name="label", width=window.LABEL_PADDING - 20)
                        px4_airframe_field = ui.StringField(name="px4_model")
                        px4_airframe_field.model.set_value(delegate._px4_airframe)
                        delegate.set_px4_airframe_field(px4_airframe_field.model)

            with ardupilot_menu:
                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    ui.Spacer(height=window.GENERAL_SPACING)
                    with ui.HStack():
                        ui.Label("Auto-launch Ardupilot", name="label", width=window.LABEL_PADDING + 10)
                        ardupilot_checkbox = ui.CheckBox()
                        ardupilot_checkbox.model.set_value(delegate._autostart_ardupilot)
                        delegate.set_ardupilot_autostart_checkbox(ardupilot_checkbox.model)

                    with ui.HStack():
                        ui.Label("ArduPilot Path", name="label", width=window.LABEL_PADDING - 20)
                        ardupilot_path_field = ui.StringField(name="ardupilot_path", width=300)
                        ardupilot_path_field.model.set_value(delegate._ardupilot_dir)
                        delegate.set_ardupilot_directory_field(ardupilot_path_field.model)

                        ui.Button("Reset", enabled=True, clicked_fn=delegate.on_reset_ardupilot_path)
                        ui.Button("Make Default", enabled=True, clicked_fn=delegate.on_set_new_default_ardupilot_path)

                    with ui.HStack():
                        ui.Label("ArduPilot airframe", name="label", width=window.LABEL_PADDING)
                        ardupilot_airframe_field = ui.StringField(name="ardupilot_model")
                        ardupilot_airframe_field.model.set_value(delegate._ardupilot_airframe)
                        delegate.set_ardupilot_airframe_field(ardupilot_airframe_field.model)

            with betaflight_menu:
                with ui.VStack(height=0, spacing=10, name="frame_v_stack"):
                    ui.Spacer(height=window.GENERAL_SPACING)

                    with ui.HStack():
                        ui.Label("Connection IP", name="label", width=window.LABEL_PADDING - 20)
                        connection_ip_field = ui.StringField(name="betaflight_connection_ip")
                        connection_ip_field.model.set_value(delegate._betaflight_connection_ip)
                        delegate.set_betaflight_connection_ip_field(connection_ip_field.model)

                    with ui.HStack():
                        ui.Label("Motor Port", name="label", width=window.LABEL_PADDING - 20)
                        motor_port_field = ui.StringField(name="betaflight_motor_port")
                        motor_port_field.model.set_value(str(delegate._betaflight_motor_port))
                        delegate.set_betaflight_motor_port_field(motor_port_field.model)

                    with ui.HStack():
                        ui.Label("State Port", name="label", width=window.LABEL_PADDING - 20)
                        state_port_field = ui.StringField(name="betaflight_state_port")
                        state_port_field.model.set_value(str(delegate._betaflight_state_port))
                        delegate.set_betaflight_state_port_field(state_port_field.model)

            ardupilot_menu.visible = False
            px4_menu.visible = False