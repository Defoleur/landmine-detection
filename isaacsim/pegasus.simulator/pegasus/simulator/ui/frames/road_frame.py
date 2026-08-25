import omni.ui as ui
import omni
from pxr import UsdGeom
from pegasus.simulator.params import SPAWN_MAPS

ROAD_PRIMS = [
    "/World/layout/road2",
    "/World/layout/road_curved_adapted",
    "/World/layout/road_curved_adapted_01",
    "/World/layout/road_curved_adapted_02",
    "/World/layout/road_curved_adapted_03",
    "/World/layout/road_curved_adapted_04",
    "/World/layout/road_curved_adapted_05",
    "/World/layout/road2_04",
    "/World/layout/ground_road_07_04_new",
    "/World/layout/road_08_04",

    "/World/layout/brick_building",
    "/World/layout/burned_car_1"
]

ROAD_OBSTACLE_PRIMS = [
    "/World/layout/Plane_02",
]

ROAD_TAG_BASE = [206.3, -33.0, 5.38507]


def build_road_frame(window, delegate):
    width = getattr(window, "LABEL_PADDING", 120)
    helper = delegate._mission_helper

    with ui.CollapsableFrame("Map", width=ui.Fraction(1)):
        with ui.VStack(spacing=10, height=0):
            with ui.HStack(spacing=10):
                ui.Label("Road Enabled", width=width)
                model = ui.SimpleBoolModel(helper._road_enabled)
                ui.CheckBox(model)

                def on_road_changed(m):
                    enabled = m.as_bool
                    helper._road_enabled = enabled
                    _apply_road_state(delegate, enabled)
                    if hasattr(helper, '_spawn_dropdown') and helper._spawn_dropdown:
                        idx = 0 if enabled else 3
                        helper._spawn_dropdown.get_item_value_model().set_value(idx)

                model.add_value_changed_fn(on_road_changed)

            with ui.HStack(spacing=10):
                ui.Label("Image Map", name="label", width=width)
                img_map_enabled_checkbox = ui.CheckBox()
                img_map_enabled_checkbox.model.set_value(helper._img_map_enabled)
                delegate.set_img_map_enabled_checkbox(img_map_enabled_checkbox.model)

            with ui.HStack(spacing=10):
                ui.Label("Spawn Map Type", name="label", width=width,
                            alignment=ui.Alignment.TOP)
                spawn_dropdown_menu = ui.ComboBox(3 if not helper._road_enabled else 0, name="spawn_map")
                for spawn in SPAWN_MAPS:
                    spawn_dropdown_menu.model.append_child_item(None, ui.SimpleStringModel(spawn))
                delegate.set_spawn_dropdown(spawn_dropdown_menu.model)


def _apply_road_state(delegate, enabled):
    # helper = delegate._mission_helper
    stage = omni.usd.get_context().get_stage()
    if not stage:
        return

    for path in ROAD_PRIMS:
        prim = stage.GetPrimAtPath(_resolve_path(path))
        if prim and prim.IsValid():
            prim.SetActive(enabled)
            if enabled and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeVisible()

    for path in ROAD_OBSTACLE_PRIMS:
        prim = stage.GetPrimAtPath(_resolve_path(path))
        if prim and prim.IsValid():
            prim.SetActive(not enabled)

    if enabled:
        _on_road_enabled(delegate)
    else:
        _on_road_disabled(delegate)


def _on_road_enabled(delegate):
    helper = delegate._mission_helper
    if not helper.tags:
        return
    helper._saved_tag_positions = [list(tag.pos) for tag in helper.tags]
    positions = _compute_triangle_formation(ROAD_TAG_BASE)
    for i, tag in enumerate(helper.tags):
        if i < len(positions):
            tag.set_pos(positions[i])


def _on_road_disabled(delegate):
    helper = delegate._mission_helper
    if not helper.tags:
        return
    saved = getattr(helper, "_saved_tag_positions", None)
    if not saved:
        return
    for i, tag in enumerate(helper.tags):
        if i < len(saved):
            tag.set_pos(saved[i])
    helper._saved_tag_positions = None


def _compute_triangle_formation(base_pos):
    return [
        list(base_pos),
        [base_pos[0] - 2, base_pos[1] + 2, base_pos[2]],
        [base_pos[0] - 2, base_pos[1] - 2, base_pos[2]],
    ]


def _resolve_path(path):
    if path and path.startswith("/World/"):
        return "/World/layout" + path[6:]
    return path
