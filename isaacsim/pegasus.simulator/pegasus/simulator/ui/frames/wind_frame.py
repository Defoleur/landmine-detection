"""
File: wind_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Global Wind Settings frame for the UI
"""

__all__ = ["build_wind_frame"]

import omni.ui as ui


def build_wind_frame(window, delegate):
    """
    Builds the Global Wind Settings frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    with ui.CollapsableFrame("Global Wind Settings", collapsed=False):
        with ui.VStack(spacing=5, height=0):
            with ui.HStack():
                ui.Label("Enable Wind", width=ui.Percent(40))
                window.wind_enabled_model = ui.SimpleBoolModel(False)
                ui.CheckBox(window.wind_enabled_model)
                window.wind_enabled_model.add_value_changed_fn(
                    lambda m: delegate.on_wind_enabled_changed(m.as_bool)
                )

            with ui.HStack():
                ui.Label("Wind Speed", width=ui.Percent(40))
                window.wind_speed_model = ui.SimpleFloatModel(0.0)
                ui.FloatDrag(window.wind_speed_model, min=0.0, max=100.0, step=0.5)
                window.wind_speed_model.add_value_changed_fn(
                    lambda m: delegate.on_wind_speed_changed(m.as_float)
                )

            with ui.HStack():
                ui.Label("Direction (X, Y, Z)", width=ui.Percent(40))

                window.wind_dir_x = ui.SimpleFloatModel(1.0)
                window.wind_dir_y = ui.SimpleFloatModel(0.0)
                window.wind_dir_z = ui.SimpleFloatModel(0.0)

                def on_dir_changed(_):
                    delegate.on_wind_direction_changed(
                        window.wind_dir_x.as_float,
                        window.wind_dir_y.as_float,
                        window.wind_dir_z.as_float
                    )

                ui.FloatDrag(window.wind_dir_x, step=0.1)
                ui.Spacer(width=2)
                ui.FloatDrag(window.wind_dir_y, step=0.1)
                ui.Spacer(width=2)
                ui.FloatDrag(window.wind_dir_z, step=0.1)

                window.wind_dir_x.add_value_changed_fn(on_dir_changed)
                window.wind_dir_y.add_value_changed_fn(on_dir_changed)
                window.wind_dir_z.add_value_changed_fn(on_dir_changed)