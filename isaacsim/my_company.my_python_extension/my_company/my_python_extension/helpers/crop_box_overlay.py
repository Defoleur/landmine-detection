import carb
from omni.ui import scene as sc, color as cl
from omni.kit.viewport.utility import get_active_viewport_and_window


class CropBoxOverlay:
    """Always-on-top viewport overlay drawing a red crosshair that marks the
    center and extent of the dataset image crop."""

    def __init__(self):
        self._scene_view = None
        self._h_line = None
        self._v_line = None
        self._attached = False

    def _ensure_attached(self):
        if self._attached and self._scene_view is not None:
            return True
        try:
            viewport_api, _ = get_active_viewport_and_window()
            if viewport_api is None:
                return False
            scene_view = sc.SceneView()
            with scene_view.scene:
                h_line = sc.Line(
                    start=(0, 0, 0), end=(1, 0, 0),
                    thickness=1.0, color=cl.red
                )
                v_line = sc.Line(
                    start=(0, 0, 0), end=(0, 1, 0),
                    thickness=1.0, color=cl.red
                )
            viewport_api.add_scene_view(scene_view)
            self._scene_view = scene_view
            self._h_line = h_line
            self._v_line = v_line
            self._attached = True
            return True
        except Exception as e:
            carb.log_warn(f"CropBoxOverlay: attach failed: {e}")
            return False

    def update_crosshair(self, center, half, right, up):
        if not self._ensure_attached():
            return
        cx, cy, cz = center
        rx, ry, rz = right
        ux, uy, uz = up
        self._h_line.start = (cx - rx * half, cy - ry * half, cz - rz * half)
        self._h_line.end = (cx + rx * half, cy + ry * half, cz + rz * half)
        self._v_line.start = (cx - ux * half, cy - uy * half, cz - uz * half)
        self._v_line.end = (cx + ux * half, cy + uy * half, cz + uz * half)

    def show(self):
        if self._ensure_attached():
            self._h_line.visible = True
            self._v_line.visible = True

    def hide(self):
        if self._h_line is not None:
            self._h_line.visible = False
            self._v_line.visible = False

    def clear(self):
        try:
            if self._scene_view is not None:
                from omni.kit.viewport.utility import get_active_viewport_and_window
                viewport_api, _ = get_active_viewport_and_window()
                if viewport_api is not None:
                    viewport_api.remove_scene_view(self._scene_view)
        except Exception:
            pass
        self._scene_view = None
        self._h_line = None
        self._v_line = None
        self._attached = False
