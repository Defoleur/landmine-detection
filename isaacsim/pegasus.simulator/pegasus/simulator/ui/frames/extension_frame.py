__all__ = [
    "build_extension_frame",
]

import os
from datetime import datetime

import carb
import omni.ui as ui
import omni.kit.app

_AUTO_PREFIX = "/ext/my_extension/"

_TRAJ_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "..",
    "my_company.my_python_extension", "my_company", "my_python_extension",
    "config", "camera_trajectories_custom",
))


def _trajectory_names():
    names = []
    try:
        if os.path.isdir(_TRAJ_DIR):
            for f in sorted(os.listdir(_TRAJ_DIR)):
                if f.endswith(".json"):
                    names.append(os.path.splitext(f)[0])
    except Exception:
        pass
    return names


def _mine_items():
    names = []
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return names
        for prim in stage.Traverse():
            if prim.IsValid() and "obstacle_mine" in prim.GetPath().pathString and prim.IsA(UsdGeom.Imageable):
                name = prim.GetName()
                if name and name not in names:
                    names.append(name)
        return names
    except Exception:
        return names


def build_extension_frame(window, delegate):
    settings = carb.settings.get_settings()

    def get_bool(key, default):
        return settings.get_as_bool(key) if settings.get(key) is not None else default

    def get_str(key, default):
        return settings.get_as_string(key) or default

    with ui.CollapsableFrame("Extension Settings"):
        with ui.VStack(spacing=10, height=0):

            with ui.CollapsableFrame("Detection", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Mines Detection", name="label", width=window.LABEL_PADDING)
                        cb = ui.CheckBox()
                        cb.model.set_value(get_bool("/ext/my_extension/mines_detection", False))
                        cb.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/mines_detection": settings.set(k, m.as_bool)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Thermal Camera", name="label", width=window.LABEL_PADDING)
                        cb = ui.CheckBox()
                        cb.model.set_value(get_bool("/ext/my_extension/thermal_camera_enabled", False))
                        cb.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/thermal_camera_enabled": settings.set(k, m.as_bool)
                        )

            with ui.CollapsableFrame("Visualization", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Trajectory Draw", name="label", width=window.LABEL_PADDING)
                        cb = ui.CheckBox()
                        cb.model.set_value(get_bool("/ext/my_extension/trajectory_draw", False))
                        cb.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/trajectory_draw": settings.set(k, m.as_bool)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Chase Camera", name="label", width=window.LABEL_PADDING)
                        cb = ui.CheckBox()
                        cb.model.set_value(get_bool("/ext/my_extension/chase_camera_enabled", False))
                        cb.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/chase_camera_enabled": settings.set(k, m.as_bool)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Drone Trail", name="label", width=window.LABEL_PADDING)
                        cb = ui.CheckBox()
                        cb.model.set_value(get_bool("/ext/my_extension/drone_trail_enabled", False))
                        cb.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/drone_trail_enabled": settings.set(k, m.as_bool)
                        )

            with ui.CollapsableFrame("Camera Settings", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Camera Mode", name="label", width=window.LABEL_PADDING)
                        initial_mode = settings.get_as_int("/ext/my_extension/camera_mode") or 0
                        mode_combo = ui.ComboBox(initial_mode, "Front", "Ground")
                        mode_combo.model.add_item_changed_fn(
                            lambda m, item, k="/ext/my_extension/camera_mode": settings.set(k, m.get_item_value_model().as_int)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Stabilized", name="label", width=window.LABEL_PADDING)
                        cb = ui.CheckBox()
                        cb.model.set_value(get_bool("/ext/my_extension/camera_stabilized", False))
                        cb.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/camera_stabilized": settings.set(k, m.as_bool)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Resolution X", name="label", width=window.LABEL_PADDING)
                        rx = ui.IntField()
                        rx.model.set_value(settings.get_as_int("/ext/my_extension/camera_width") or 800)
                        settings.set("/ext/my_extension/camera_width", rx.model.as_int)
                        rx.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/camera_width": settings.set(k, m.as_int)
                        )
                    with ui.HStack(spacing=10):
                        ui.Label("Resolution Y", name="label", width=window.LABEL_PADDING)
                        ry = ui.IntField()
                        ry.model.set_value(settings.get_as_int("/ext/my_extension/camera_height") or 600)
                        settings.set("/ext/my_extension/camera_height", ry.model.as_int)
                        ry.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/camera_height": settings.set(k, m.as_int)
                        )
                    with ui.HStack(spacing=10):
                        ui.Label("Frequency", name="label", width=window.LABEL_PADDING)
                        freq = ui.IntField()
                        freq.model.set_value(settings.get_as_int("/ext/my_extension/camera_frequency") or 30)
                        settings.set("/ext/my_extension/camera_frequency", freq.model.as_int)
                        freq.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/camera_frequency": settings.set(k, m.as_int)
                        )
                    with ui.HStack(spacing=10):
                        ui.Label("Focal Length", name="label", width=window.LABEL_PADDING)
                        fl = ui.FloatField()
                        fl.model.set_value(settings.get_as_float("/ext/my_extension/camera_focal_length") or 3.0)
                        settings.set("/ext/my_extension/camera_focal_length", freq.model.as_float)
                        fl.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/camera_focal_length": settings.set(k, m.as_float)
                        )

            with ui.CollapsableFrame("Tracking", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Tag Path", name="label", width=window.LABEL_PADDING)
                        tag_path = ui.StringField()
                        tag_path.model.set_value(get_str("/ext/my_extension/tag_path", "/World/tag_02"))
                        tag_path.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/tag_path": settings.set(k, m.as_string)
                        )

            with ui.CollapsableFrame("Dataset Creation", collapsed=True):
                default_base = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../../../datasets")
                )
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Dataset Name", name="label", width=window.LABEL_PADDING)
                        ds_name = ui.StringField()
                        ds_name.model.set_value(get_str("/ext/my_extension/dataset_name", "dataset"))
                        ds_name.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/dataset_name": settings.set(k, m.as_string)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Capture Rate (Hz)", name="label", width=window.LABEL_PADDING)
                        ds_rate = ui.FloatField()
                        ds_rate.model.set_value(settings.get_as_float("/ext/my_extension/dataset_rate") or 2.0)
                        settings.set("/ext/my_extension/dataset_rate", ds_rate.model.as_float)
                        ds_rate.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/dataset_rate": settings.set(k, m.as_float)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("High-Res Size", name="label", width=window.LABEL_PADDING)
                        ds_high = ui.IntField()
                        ds_high.model.set_value(settings.get_as_int("/ext/my_extension/dataset_high_res") or 2160)
                        settings.set("/ext/my_extension/dataset_high_res", ds_high.model.as_int)
                        ds_high.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/dataset_high_res": settings.set(k, m.as_int)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Crop Size", name="label", width=window.LABEL_PADDING)
                        ds_crop = ui.IntField()
                        ds_crop.model.set_value(settings.get_as_int("/ext/my_extension/dataset_crop_size") or 640)
                        settings.set("/ext/my_extension/dataset_crop_size", ds_crop.model.as_int)
                        ds_crop.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/dataset_crop_size": settings.set(k, m.as_int)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Min Visibility (%)", name="label", width=window.LABEL_PADDING)
                        ds_vis = ui.FloatField()
                        ds_vis.model.set_value(
                            (settings.get_as_float("/ext/my_extension/dataset_min_visible")
                             if settings.get_as_float("/ext/my_extension/dataset_min_visible") is not None
                             else 0.3) * 100.0
                        )
                        ds_vis.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/dataset_min_visible":
                                settings.set(k, max(0.0, min(1.0, m.as_float / 100.0)))
                        )

                    vis_enabled = settings.get_as_bool("/ext/my_extension/visibility_filter_enabled") or False

                    def _on_vis_enabled(m, k="/ext/my_extension/visibility_filter_enabled"):
                        settings.set(k, m.as_bool)
                        ds_vis.enabled = bool(m.as_bool)

                    with ui.HStack(spacing=10):
                        ui.Label("Visibility Filter", name="label", width=window.LABEL_PADDING)
                        ds_vis_enabled = ui.CheckBox(model=ui.SimpleBoolModel(vis_enabled))
                        ds_vis_enabled.model.add_value_changed_fn(_on_vis_enabled)
                        ds_vis.enabled = vis_enabled

                    with ui.HStack(spacing=10):
                        ui.Label("Output Folder", name="label", width=window.LABEL_PADDING)
                        ds_base = ui.StringField()
                        ds_base.model.set_value(get_str("/ext/my_extension/dataset_base_dir", default_base))
                        ds_base.model.add_value_changed_fn(
                            lambda m, k="/ext/my_extension/dataset_base_dir": settings.set(k, m.as_string)
                        )

                    status_label = ui.Label("Current dataset: -", name="label")

                    with ui.HStack(spacing=10):
                        start_btn = ui.Button(
                            "Start Recording",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=lambda: _start_dataset(),
                        )
                        stop_btn = ui.Button(
                            "Stop Recording",
                            height=window.BUTTON_HEIGHT,
                            enabled=False,
                            clicked_fn=lambda: _stop_dataset(),
                        )

                    def _start_dataset():
                        name = ds_name.model.as_string or "dataset"
                        full_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        settings.set("/ext/my_extension/dataset_name", full_name)
                        settings.set("/ext/my_extension/dataset_rate", ds_rate.model.as_float)
                        settings.set("/ext/my_extension/dataset_high_res", ds_high.model.as_int)
                        settings.set("/ext/my_extension/dataset_crop_size", ds_crop.model.as_int)
                        settings.set("/ext/my_extension/dataset_min_visible",
                                     max(0.0, min(1.0, ds_vis.model.as_float / 100.0)))
                        settings.set("/ext/my_extension/visibility_filter_enabled",
                                     ds_vis_enabled.model.as_bool)
                        settings.set("/ext/my_extension/dataset_base_dir", ds_base.model.as_string or default_base)
                        status_label.text = f"Current dataset: {full_name}"
                        settings.set("/ext/my_extension/dataset_capture_enabled", True)
                        start_btn.enabled = False
                        stop_btn.enabled = True

                    def _stop_dataset():
                        settings.set("/ext/my_extension/dataset_capture_enabled", False)
                        status_label.text = "Current dataset: -"
                        start_btn.enabled = True
                        stop_btn.enabled = False

            with ui.CollapsableFrame("Trajectory Authoring", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Trajectory Name", name="label", width=window.LABEL_PADDING)
                        traj_name = ui.StringField()
                        traj_name.model.set_value(get_str(_AUTO_PREFIX + "traj_name", "trajectory"))
                        traj_name.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "traj_name": settings.set(k, m.as_string)
                        )

                    traj_hint = ui.Label(
                        "Fly the camera; position, orientation and time are recorded.",
                        name="label",
                    )

                    with ui.HStack(spacing=10):
                        traj_start_btn = ui.Button(
                            "Start Traj Record",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=lambda: _start_traj(),
                        )
                        traj_stop_btn = ui.Button(
                            "Stop & Save Traj",
                            height=window.BUTTON_HEIGHT,
                            enabled=False,
                            clicked_fn=lambda: _stop_traj(),
                        )

                    def _start_traj():
                        settings.set(_AUTO_PREFIX + "traj_name", traj_name.model.as_string or "trajectory")
                        settings.set(_AUTO_PREFIX + "traj_record_enabled", True)
                        traj_start_btn.enabled = False
                        traj_stop_btn.enabled = True
                        traj_hint.text = "Recording... (position, orientation, time)."

                    def _stop_traj():
                        settings.set(_AUTO_PREFIX + "traj_record_enabled", False)
                        traj_start_btn.enabled = True
                        traj_stop_btn.enabled = False
                        traj_hint.text = "Saved. Replay translates position to the mine and keeps orientation."

            with ui.CollapsableFrame("Trajectory Preview", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Interval (s)", name="label", width=window.LABEL_PADDING)
                        interval_field = ui.FloatField()
                        interval_field.model.set_value(
                            settings.get_as_float(_AUTO_PREFIX + "traj_preview_interval") or 5.0
                        )
                        interval_field.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "traj_preview_interval": settings.set(k, m.as_float)
                        )

                    def _populate_combo(combo, items, selected):
                        model = combo.model
                        for item in list(model.get_item_children()):
                            model.remove_item(item)
                        for value in items:
                            model.append_child_item(None, ui.SimpleStringModel(value))
                        idx = items.index(selected) if selected in items else 0
                        model.get_item_value_model().set_value(idx)

                    with ui.HStack(spacing=10):
                        ui.Label("Landmine", name="label", width=window.LABEL_PADDING)
                        mine_combo = ui.ComboBox(0, "Auto (nearest drone)")

                    with ui.HStack(spacing=10):
                        ui.Label("Trajectory", name="label", width=window.LABEL_PADDING)
                        traj_combo = ui.ComboBox(0, "All trajectories")

                    _mine_combo_items = []
                    _traj_combo_items = []

                    def _on_mine_changed(m, item, k=_AUTO_PREFIX + "traj_preview_mine"):
                        idx = m.get_item_value_model().as_int
                        if idx <= 0 or idx >= len(_mine_combo_items):
                            settings.set(k, "")
                        else:
                            settings.set(k, _mine_combo_items[idx])

                    def _on_traj_changed(m, item, k=_AUTO_PREFIX + "traj_preview_traj"):
                        idx = m.get_item_value_model().as_int
                        if idx <= 0 or idx >= len(_traj_combo_items):
                            settings.set(k, "")
                        else:
                            settings.set(k, _traj_combo_items[idx])

                    mine_combo.model.add_item_changed_fn(_on_mine_changed)
                    traj_combo.model.add_item_changed_fn(_on_traj_changed)

                    preview_status = ui.Label("Preview: -", name="label")

                    with ui.HStack(spacing=10):
                        create_cam_btn = ui.Button(
                            "Create Preview Camera",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=lambda: _create_preview_camera(),
                        )
                        reset_cam_btn = ui.Button(
                            "Reset Camera",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=lambda: _reset_camera(),
                        )

                    def _create_preview_camera():
                        settings.set(_AUTO_PREFIX + "traj_create_camera", True)
                        create_cam_btn.enabled = False

                    def _reset_camera():
                        settings.set(_AUTO_PREFIX + "traj_reset_camera", True)

                    with ui.HStack(spacing=10):
                        preview_play_btn = ui.Button(
                            "Play Preview",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=lambda: _play_preview(),
                        )
                        preview_stop_btn = ui.Button(
                            "Stop Preview",
                            height=window.BUTTON_HEIGHT,
                            enabled=False,
                            clicked_fn=lambda: _stop_preview(),
                        )

                    def _play_preview():
                        settings.set(_AUTO_PREFIX + "traj_preview_interval", interval_field.model.as_float)
                        _refresh_combos(force=True)
                        settings.set(_AUTO_PREFIX + "traj_preview_enabled", True)
                        create_cam_btn.enabled = False
                        preview_play_btn.enabled = False
                        preview_stop_btn.enabled = True
                        preview_status.text = "Preview: running"

                    def _stop_preview():
                        settings.set(_AUTO_PREFIX + "traj_preview_enabled", False)
                        create_cam_btn.enabled = True
                        preview_play_btn.enabled = True
                        preview_stop_btn.enabled = False
                        preview_status.text = "Preview: -"

                    def _refresh_preview_status():
                        text = settings.get_as_string(_AUTO_PREFIX + "traj_preview_status") or "-"
                        preview_status.text = f"Preview: {text}"
                        running = settings.get_as_bool(_AUTO_PREFIX + "traj_preview_enabled") or False
                        preview_play_btn.enabled = not running
                        preview_stop_btn.enabled = running
                        _refresh_combos()

                    _preview_combo_state = {"mines": "", "trajs": ""}

                    def _refresh_combos(force=False):
                        mine_items = ["Auto (nearest drone)"] + _mine_items()
                        traj_items = ["All trajectories"] + _trajectory_names()
                        mine_key = "|".join(mine_items)
                        traj_key = "|".join(traj_items)
                        if force or mine_key != _preview_combo_state["mines"]:
                            _mine_combo_items[:] = mine_items
                            _populate_combo(
                                mine_combo, mine_items,
                                get_str(_AUTO_PREFIX + "traj_preview_mine", ""),
                            )
                            _preview_combo_state["mines"] = mine_key
                        if force or traj_key != _preview_combo_state["trajs"]:
                            _traj_combo_items[:] = traj_items
                            _populate_combo(
                                traj_combo, traj_items,
                                get_str(_AUTO_PREFIX + "traj_preview_traj", ""),
                            )
                            _preview_combo_state["trajs"] = traj_key

                    _refresh_combos(force=True)

                    _preview_refresh_counter = {"n": 0}

                    def _preview_status_tick(_e, _c=_preview_refresh_counter):
                        _c["n"] += 1
                        if _c["n"] % 15 == 0:
                            _refresh_preview_status()

                    _app = omni.kit.app.get_app()
                    window._preview_status_sub = _app.get_update_event_stream().create_subscription_to_pop(
                        _preview_status_tick, name="Traj Preview Status"
                    )

            with ui.CollapsableFrame("Auto Dataset Recording", collapsed=True):
                with ui.VStack(spacing=10, height=0):
                    with ui.HStack(spacing=10):
                        ui.Label("Seasons", name="label", width=window.LABEL_PADDING)
                        seasons_field = ui.StringField()
                        seasons_field.model.set_value(get_str(_AUTO_PREFIX + "auto_record_seasons",
                                                              "summer,autumn,winter,spring"))
                        seasons_field.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_seasons": settings.set(k, m.as_string)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Daytimes", name="label", width=window.LABEL_PADDING)
                        daytimes_field = ui.StringField()
                        daytimes_field.model.set_value(get_str(_AUTO_PREFIX + "auto_record_daytimes",
                                                               "day,morning,evening"))
                        daytimes_field.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_daytimes": settings.set(k, m.as_string)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Shots / Mine", name="label", width=window.LABEL_PADDING)
                        shots_field = ui.IntField()
                        shots_field.model.set_value(settings.get_as_int(_AUTO_PREFIX + "auto_record_shots_per_mine") or 10)
                        settings.set(_AUTO_PREFIX + "auto_record_shots_per_mine", shots_field.model.as_int)
                        shots_field.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_shots_per_mine": settings.set(k, m.as_int)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Follow Viewport", name="label", width=window.LABEL_PADDING)
                        follow_cb = ui.CheckBox()
                        follow_cb.model.set_value(get_bool(_AUTO_PREFIX + "auto_record_follow_viewport", True))
                        follow_cb.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_follow_viewport": settings.set(k, m.as_bool)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Random Speed", name="label", width=window.LABEL_PADDING)
                        speed_cb = ui.CheckBox()
                        speed_cb.model.set_value(get_bool(_AUTO_PREFIX + "auto_record_speed_enabled", True))
                        speed_cb.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_speed_enabled": settings.set(k, m.as_bool)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Speed Min", name="label", width=window.LABEL_PADDING)
                        speed_min_field = ui.FloatField()
                        speed_min_field.model.set_value(
                            settings.get_as_float(_AUTO_PREFIX + "auto_record_speed_min") or 2.0
                        )
                        speed_min_field.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_speed_min": settings.set(k, m.as_float)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Speed Max", name="label", width=window.LABEL_PADDING)
                        speed_max_field = ui.FloatField()
                        speed_max_field.model.set_value(
                            settings.get_as_float(_AUTO_PREFIX + "auto_record_speed_max") or 2.0
                        )
                        speed_max_field.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_speed_max": settings.set(k, m.as_float)
                        )

                    with ui.HStack(spacing=10):
                        ui.Label("Close Sim on Finish", name="label", width=window.LABEL_PADDING)
                        close_cb = ui.CheckBox()
                        close_cb.model.set_value(get_bool(_AUTO_PREFIX + "auto_record_close_on_finish", True))
                        close_cb.model.add_value_changed_fn(
                            lambda m, k=_AUTO_PREFIX + "auto_record_close_on_finish": settings.set(k, m.as_bool)
                        )

                    auto_status = ui.Label("Auto Record: -", name="label")

                    with ui.HStack(spacing=10):
                        auto_start_btn = ui.Button(
                            "Start Auto Record",
                            height=window.BUTTON_HEIGHT,
                            clicked_fn=lambda: _start_auto(),
                        )
                        auto_stop_btn = ui.Button(
                            "Stop Auto Record",
                            height=window.BUTTON_HEIGHT,
                            enabled=False,
                            clicked_fn=lambda: _stop_auto(),
                        )

                    def _start_auto():
                        settings.set(_AUTO_PREFIX + "auto_record_seasons",
                                     seasons_field.model.as_string or "summer,autumn,winter,spring")
                        settings.set(_AUTO_PREFIX + "auto_record_daytimes",
                                     daytimes_field.model.as_string or "day,morning,evening")
                        settings.set(_AUTO_PREFIX + "auto_record_shots_per_mine", shots_field.model.as_int)
                        settings.set(_AUTO_PREFIX + "auto_record_follow_viewport", follow_cb.model.as_bool)
                        settings.set(_AUTO_PREFIX + "auto_record_speed_enabled", speed_cb.model.as_bool)
                        settings.set(_AUTO_PREFIX + "auto_record_speed_min", speed_min_field.model.as_float)
                        settings.set(_AUTO_PREFIX + "auto_record_speed_max", speed_max_field.model.as_float)
                        settings.set(_AUTO_PREFIX + "auto_record_close_on_finish", close_cb.model.as_bool)
                        settings.set(_AUTO_PREFIX + "auto_record_enabled", True)
                        auto_start_btn.enabled = False
                        auto_stop_btn.enabled = True
                        auto_status.text = "Auto Record: running"

                    def _stop_auto():
                        settings.set(_AUTO_PREFIX + "auto_record_enabled", False)
                        settings.set("/ext/my_extension/dataset_capture_enabled", False)

                        auto_start_btn.enabled = True
                        auto_stop_btn.enabled = False
                        auto_status.text = "Auto Record: -"


                    def _refresh_auto_status():
                        text = settings.get_as_string(_AUTO_PREFIX + "auto_record_status") or "-"
                        auto_status.text = f"Auto Record: {text}"
                        running = settings.get_as_bool(_AUTO_PREFIX + "auto_record_enabled") or False
                        auto_start_btn.enabled = not running
                        auto_stop_btn.enabled = running

                    _auto_refresh_counter = {"n": 0}

                    def _auto_status_tick(_e, _c=_auto_refresh_counter):
                        _c["n"] += 1
                        if _c["n"] % 15 == 0:
                            _refresh_auto_status()

                    _app = omni.kit.app.get_app()
                    window._auto_status_sub = _app.get_update_event_stream().create_subscription_to_pop(
                        _auto_status_tick, name="Auto Record Status"
                    )
                    _refresh_auto_status()
