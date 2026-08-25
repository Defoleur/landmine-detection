__all__ = [
    "build_schedule_frame",
]

import re
import omni.ui as ui

SEASON_NAMES = ["Summer", "Autumn", "Winter", "Spring"]
SEASON_VALUES = ["summer", "autumn", "winter", "spring"]
DAYTIME_NAMES = ["Day", "Evening", "Night", "Morning"]
DAYTIME_VALUES = ["day", "evening", "night", "morning"]
MODE_NAMES = ["Auto", "Custom"]


def _update_preview(window, helper):
    if window._preview_frame:
        window._preview_frame.rebuild()


def _build_preview_frame(window, helper):
    flat = helper._build_schedule_flat()
    with ui.VStack(spacing=2):
        if not flat:
            ui.Label("(empty)", name="label")
            return

        current_index = helper._schedule_current_index if helper.mission_time is not None else -1
        for i, (c, s, d) in enumerate(flat, 1):
            sn = SEASON_NAMES[SEASON_VALUES.index(s)] if s in SEASON_VALUES else s
            dn = DAYTIME_NAMES[DAYTIME_VALUES.index(d)] if d in DAYTIME_VALUES else d
            is_current = (i - 1) == current_index
            with ui.HStack(spacing=5):
                ui.Label(">" if is_current else "", width=14, name="label")
                ui.Label(f"#{i}", width=20, name="label")
                ui.Label(f"{c}x", width=40, name="label")
                ui.Label(sn, width=80, name="label")
                ui.Label(dn, width=80, name="label")


def _sync_custom_to_helper(window, helper):
    helper._schedule_custom = []
    for row in window._schedule_row_models:
        c = row["count"]
        s = SEASON_VALUES[row["season_idx"]]
        d = DAYTIME_VALUES[row["daytime_idx"]]
        helper._schedule_custom.append((c, s, d))


def _add_from_picker(window, helper):
    c = window._picker_count_model.as_int
    s = window._picker_season_combo.model.get_item_value_model().as_int
    d = window._picker_daytime_combo.model.get_item_value_model().as_int
    row_id = window._schedule_next_id
    window._schedule_next_id += 1
    window._schedule_row_models.append({
        "id": row_id,
        "count": c,
        "season_idx": s,
        "daytime_idx": d,
    })
    _on_rows_changed(window, helper)


def _remove_row(window, helper, row_id):
    window._schedule_row_models[:] = [r for r in window._schedule_row_models if r["id"] != row_id]
    _on_rows_changed(window, helper)


def _clear_rows(window, helper):
    window._schedule_row_models.clear()
    _on_rows_changed(window, helper)


def _on_rows_changed(window, helper):
    _sync_custom_to_helper(window, helper)
    _update_preview(window, helper)
    if window._custom_frame:
        window._custom_frame.rebuild()
    window._status_label.text = ""


def _parse_text_to_rows(window, helper):
    text = window._schedule_text_model.as_string
    lines = text.strip().split('\n')
    valid_rows = []
    invalid_lines = []
    next_id = window._schedule_next_id
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\d+)x\s+(\w+)/(\w+)$', line)
        if not m:
            invalid_lines.append(i)
            continue
        count = int(m.group(1))
        season_str = m.group(2).lower()
        daytime_str = m.group(3).lower()
        if season_str not in SEASON_VALUES:
            invalid_lines.append(i)
            continue
        if daytime_str not in DAYTIME_VALUES:
            invalid_lines.append(i)
            continue
        si = SEASON_VALUES.index(season_str)
        di = DAYTIME_VALUES.index(daytime_str)
        valid_rows.append({
            "id": next_id,
            "count": count,
            "season_idx": si,
            "daytime_idx": di,
        })
        next_id += 1
    if invalid_lines:
        window._status_label.text = f"Parse error on line(s): {', '.join(str(l) for l in invalid_lines)}"
        return
    window._schedule_row_models = valid_rows
    window._schedule_next_id = next_id
    _on_rows_changed(window, helper)
    window._status_label.text = f"Parsed {len(valid_rows)} entr{'y' if len(valid_rows) == 1 else 'ies'}"


def _copy_rows_to_text(window, helper):
    lines = []
    for row in window._schedule_row_models:
        s = SEASON_VALUES[row["season_idx"]]
        d = DAYTIME_VALUES[row["daytime_idx"]]
        lines.append(f"{row['count']}x {s}/{d}")
    window._schedule_text_model.as_string = "\n".join(lines)
    window._status_label.text = "Copied!"


def _build_custom_frame(window, helper):
    with ui.VStack(spacing=5):
        for row in window._schedule_row_models:
            with ui.HStack():
                text = f"{row['count']}x {SEASON_NAMES[row['season_idx']]} / {DAYTIME_NAMES[row['daytime_idx']]}"
                ui.Label(text, name="label", alignment=ui.Alignment.LEFT)
                ui.Button("X", width=20, clicked_fn=lambda rid=row["id"]: _remove_row(window, helper, rid))


def build_schedule_frame(window, delegate):
    helper = delegate._mission_helper

    if not hasattr(window, "_schedule_row_models"):
        window._schedule_row_models = []
        window._schedule_next_id = 0

    if not hasattr(window, "_schedule_text_model"):
        window._schedule_text_model = ui.SimpleStringModel("")

    window._custom_frame = None

    with ui.CollapsableFrame("Season Schedule"):
        with ui.VStack(spacing=10, height=0):
            with ui.HStack(spacing=10):
                ui.Label("Enabled", name="label", width=window.LABEL_PADDING)
                enable_model = ui.SimpleBoolModel(helper._schedule_enabled)
                ui.CheckBox(enable_model)
                enable_model.add_value_changed_fn(
                    lambda m: setattr(helper, "_schedule_enabled", m.as_bool)
                )

            with ui.HStack(spacing=10):
                ui.Label("Mode", name="label", width=window.LABEL_PADDING)
                window._mode_combo = ui.ComboBox(0, *MODE_NAMES)
                window._mode_combo.model.add_item_changed_fn(
                    lambda m, _it: _on_mode_changed(window, helper, m.get_item_value_model().as_int)
                )

            window._auto_section = ui.VStack(spacing=5, height=0)
            with window._auto_section:
                with ui.HStack(spacing=20):
                    with ui.VStack(spacing=5):
                        ui.Label("Seasons", alignment=ui.Alignment.LEFT, style={"color": 0xFFAAAAAA, "font_weight": "bold"})
                        for val, name in zip(SEASON_VALUES, SEASON_NAMES):
                            with ui.HStack():
                                m = ui.SimpleBoolModel(val in helper._schedule_seasons)
                                m.add_value_changed_fn(
                                    lambda _, v=val, md=m: (
                                        _on_season_changed(helper, v, md),
                                        _update_preview(window, helper),
                                    )
                                )
                                ui.CheckBox(m, width=20)
                                ui.Label(name, name="label", alignment=ui.Alignment.LEFT)

                    with ui.VStack(spacing=5):
                        ui.Label("Daytime", alignment=ui.Alignment.LEFT, style={"color": 0xFFAAAAAA, "font_weight": "bold"})
                        for val, name in zip(DAYTIME_VALUES, DAYTIME_NAMES):
                            with ui.HStack():
                                m = ui.SimpleBoolModel(val in helper._schedule_daytimes)
                                m.add_value_changed_fn(
                                    lambda _, v=val, md=m: (
                                        _on_daytime_changed(helper, v, md),
                                        _update_preview(window, helper),
                                    )
                                )
                                ui.CheckBox(m, width=20)
                                ui.Label(name, name="label", alignment=ui.Alignment.LEFT)

                ui.Separator()

                with ui.HStack(spacing=10):
                    ui.Label("Episodes per combo", name="label", width=window.LABEL_PADDING)
                    episodes_model = ui.SimpleIntModel(helper._schedule_episodes)
                    ui.IntDrag(episodes_model, min=1, max=1000)
                    episodes_model.add_value_changed_fn(
                        lambda m: (
                            setattr(helper, "_schedule_episodes", m.as_int),
                            _update_preview(window, helper),
                        )
                    )

            window._custom_section = ui.VStack(spacing=8, height=0)
            with window._custom_section:
                ui.Label("Bulk Input", alignment=ui.Alignment.LEFT, style={"font_weight": "bold"},
                         tooltip=(
                             "Format: <count>x <season>/<daytime>\n"
                             "\n"
                             "Example:\n"
                             "  5x summer/day\n"
                             "  3x winter/night\n"
                             "  2x autumn/morning\n"
                             "  7x spring/evening\n"
                             "\n"
                             "Valid seasons: summer, winter, autumn, spring\n"
                             "Valid daytimes: day, night, morning, evening"
                         ))
                ui.StringField(
                    window._schedule_text_model,
                    height=80,
                    multiline=True,
                )
                with ui.HStack(spacing=10):
                    ui.Button("Parse", width=80, clicked_fn=lambda: _parse_text_to_rows(window, helper))
                    ui.Button("Copy", width=80, clicked_fn=lambda: _copy_rows_to_text(window, helper))
                    window._status_label = ui.Label("", name="label", width=200, alignment=ui.Alignment.LEFT_CENTER)

                ui.Separator()

                ui.Label("Manual Entry", alignment=ui.Alignment.LEFT, style={"color": 0xFFAAAAAA, "font_weight": "bold"})
                with ui.HStack(spacing=10):
                    ui.Label("Count", name="label", width=48)
                    window._picker_count_model = ui.SimpleIntModel(1)
                    ui.IntDrag(window._picker_count_model, min=1, max=1000, width=60)
                    ui.Label("Season", name="label", width=48)
                    window._picker_season_combo = ui.ComboBox(0, *SEASON_NAMES, width=90)
                    ui.Label("Daytime", name="label", width=48)
                    window._picker_daytime_combo = ui.ComboBox(0, *DAYTIME_NAMES, width=90)
                ui.Button("+ Add to Schedule", clicked_fn=lambda: _add_from_picker(window, helper))

                ui.Separator()

                ui.Label("Schedule Rows", alignment=ui.Alignment.LEFT, style={"color": 0xFFAAAAAA, "font_weight": "bold"})
                window._custom_frame = ui.Frame()
                window._custom_frame.set_build_fn(
                    lambda: _build_custom_frame(window, helper)
                )
                ui.Button("Clear All", clicked_fn=lambda: _clear_rows(window, helper))

            ui.Separator()

            with ui.HStack(spacing=10):
                ui.Label("Current", name="label", width=window.LABEL_PADDING)
                window._schedule_status_label = ui.Label("--", name="label")
            
            ui.Separator()

            ui.Label("Preview", name="header", alignment=ui.Alignment.LEFT, style={"font_weight": "bold"})
            window._preview_frame = ui.Frame()
            window._preview_frame.set_build_fn(
                lambda: _build_preview_frame(window, helper)
            )

            window._auto_section.visible = True
            window._custom_section.visible = False

    # Load saved custom data
    if helper._schedule_custom is not None and helper._schedule_custom:
        for c, s, d in helper._schedule_custom:
            row_id = window._schedule_next_id
            window._schedule_next_id += 1
            si = SEASON_VALUES.index(s) if s in SEASON_VALUES else 0
            di = DAYTIME_VALUES.index(d) if d in DAYTIME_VALUES else 0
            window._schedule_row_models.append({
                "id": row_id,
                "count": c,
                "season_idx": si,
                "daytime_idx": di,
            })
        window._mode_combo.model.get_item_value_model().as_int = 1
        window._auto_section.visible = False
        window._custom_section.visible = True
        _sync_custom_to_helper(window, helper)
        lines = []
        for row in window._schedule_row_models:
            s = SEASON_VALUES[row["season_idx"]]
            d = DAYTIME_VALUES[row["daytime_idx"]]
            lines.append(f"{row['count']}x {s}/{d}")
        window._schedule_text_model.as_string = "\n".join(lines)
    _update_preview(window, helper)


def _on_mode_changed(window, helper, mode):
    window._status_label.text = ""
    if mode == 0:
        window._auto_section.visible = True
        window._custom_section.visible = False
        helper._schedule_custom = None
        _update_preview(window, helper)
    else:
        window._auto_section.visible = False
        window._custom_section.visible = True
        _sync_custom_to_helper(window, helper)
        _update_preview(window, helper)


def _on_season_changed(helper, val, model):
    if model.as_bool:
        if val not in helper._schedule_seasons:
            helper._schedule_seasons.append(val)
    else:
        if val in helper._schedule_seasons:
            helper._schedule_seasons.remove(val)


def _on_daytime_changed(helper, val, model):
    if model.as_bool:
        if val not in helper._schedule_daytimes:
            helper._schedule_daytimes.append(val)
    else:
        if val in helper._schedule_daytimes:
            helper._schedule_daytimes.remove(val)
