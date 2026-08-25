__all__ = [
    "build_daytime_frame",
]

import omni.ui as ui


def build_daytime_frame(window, delegate):
    helper = delegate._mission_helper

    with ui.CollapsableFrame("Time of Day"):
        with ui.VStack(spacing=10, height=0):
            with ui.HStack(spacing=10):
                ui.Label("Enabled", width=window.LABEL_PADDING)
                enabled_model = ui.SimpleBoolModel(helper.physics._daytime_enabled)
                ui.CheckBox(enabled_model)
                enabled_model.add_value_changed_fn(
                    lambda m: setattr(helper.physics, "_daytime_enabled", m.as_bool)
                )

            with ui.HStack(spacing=10):
                ui.Label("Update Frequency", width=window.LABEL_PADDING)
                freq_model = ui.SimpleIntModel(helper.physics._daytime_update_frequency)
                ui.IntDrag(freq_model, min=1, max=1000)
                freq_model.add_value_changed_fn(
                    lambda m: setattr(helper.physics, "_daytime_update_frequency", m.as_int)
                )

            with ui.HStack(spacing=10):
                ui.Label("Rate", width=window.LABEL_PADDING)
                rate_model = ui.SimpleFloatModel(helper.physics._daytime_rate)
                ui.FloatDrag(rate_model, min=0.0, max=10.0, step=0.001)
                rate_model.add_value_changed_fn(
                    lambda m: setattr(helper.physics, "_daytime_rate", m.as_float)
                )
