__all__ = [
    "build_environment_frame",
]

import omni.ui as ui
from pegasus.simulator.helpers.environment_manager import EnvironmentManager


SEASON_NAMES = ["Summer", "Winter", "Autumn", "Spring"]
DAYTIME_NAMES = ["Day", "Night", "Morning", "Evening"]


def _on_apply(window, env, delegate):
    season_index = window._season_combo.model.get_item_value_model().as_int
    daytime_index = window._daytime_combo.model.get_item_value_model().as_int
    replace_trees = window._replace_trees_checkbox.model.as_bool
    if 0 <= season_index < len(SEASON_NAMES):
        season_name = SEASON_NAMES[season_index].lower()
        env.apply_season(season_name, replace_trees=replace_trees)
        delegate._mission_helper.mines_helper.update_vegetation_for_season(season_name)
    if 0 <= daytime_index < len(DAYTIME_NAMES):
        env.apply_daytime(DAYTIME_NAMES[daytime_index].lower())


def build_environment_frame(window, delegate):
    if not hasattr(window, "_env_manager"):
        window._env_manager = EnvironmentManager()

    env = window._env_manager
    env.set_stage()

    with ui.CollapsableFrame("Season & Daytime"):
        with ui.VStack(spacing=10, height=0):
            with ui.HStack(spacing=10):
                ui.Label("Season", name="label", width=window.LABEL_PADDING)
                window._season_combo = ui.ComboBox(0, *SEASON_NAMES)

            with ui.HStack(spacing=10):
                ui.Label("Daytime", name="label", width=window.LABEL_PADDING)
                window._daytime_combo = ui.ComboBox(0, *DAYTIME_NAMES)

            with ui.HStack(spacing=10):
                ui.Label("Replace Trees", name="label", width=window.LABEL_PADDING)
                window._replace_trees_checkbox = ui.CheckBox()
                window._replace_trees_checkbox.model.set_value(True)

            ui.Button("Apply", height=window.BUTTON_HEIGHT, clicked_fn=lambda: _on_apply(window, env, delegate))
