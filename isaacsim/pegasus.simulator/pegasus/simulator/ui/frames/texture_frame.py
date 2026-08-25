"""
File: texture_frame.py
Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
Description: Thermal Texture selection frame for the UI
"""

__all__ = ["build_texture_frame"]

import omni.ui as ui


def build_texture_frame(window, delegate):
    """
    Builds the Thermal Texture selection frame.

    Args:
        window: The main WidgetWindow instance
        delegate: The UIDelegate instance
    """
    with ui.CollapsableFrame("Thermal Texture", collapsed=True):
        with ui.HStack(spacing=5):
            ui.Label("Select Type:", width=100)

            window.variant_combo = ui.ComboBox(0, "default", "thermal")

            window.variant_combo.model.add_item_changed_fn(
                delegate.on_variant_changed
            )