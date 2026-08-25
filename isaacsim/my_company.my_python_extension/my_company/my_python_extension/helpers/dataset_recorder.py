import os
import time
import math
import carb
import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom

# --- Category tables ---------------------------------------------------------

_ALL_MINE_TYPES = [
    "Mine_TM46",
    "Mine_TM57",
    "Mine_TM62",
    "Mine_TM62B",
    "Mine_TM62D",
    "Mine_TM62P",
    "Mine_TM62P2",
    "Mine_TM62P3",
    "Mine_TM62T",
    "Mine_TM72",
    "Mine_TM83",
    "Mine_TM89",
    "Mine_PTM1",
    "Mine_PTM3",
    "Mine_PTM4",
    "Mine_PTKM1R",
    "Mine_PMN",
    "Mine_PMN2",
    "Mine_PMN4",
    "Mine_PFM1",
    "Mine_PMD6",
    "Mine_POM2",
    "Mine_POMZ2",
    "Mine_POM3",
    "Mine_OZM72",
    "Mine_POB_Pilka",
    "Mine_MON50",
    "Mine_MON90",
    "Mine_MON100",
    "Mine_MON200",
    "Mine_MS3",
    "Mine_MS4",
    "Mine_ML7",
    "Mine_ML8",
    "Mine_Cloth",
    "Mine_MOB",
    "Mine_Dzika",
]

# Exact (per-type) categories, ids in alphabetical order for determinism.
EXACT_MINE_TYPES = sorted(_ALL_MINE_TYPES)
EXACT_IDS = {t: idx for idx, t in enumerate(EXACT_MINE_TYPES)}

# Merged categories, ids in the order listed below.
MERGED_GROUPS = {
    "mine_disc": [
        "Mine_TM46", "Mine_TM57", "Mine_TM62", "Mine_TM62B",
        "Mine_TM62P", "Mine_TM62P2", "Mine_TM62P3", "Mine_TM62T",
        "Mine_TM72", "Mine_TM89", "Mine_PMN", "Mine_PMN2",
        "Mine_PMN4", "Mine_MS3",
    ],
    "mine_rectangular_elongated": [
        "Mine_PTM1", "Mine_PTM3", "Mine_PTM4"
    ],
    "mine_square": [
        "Mine_TM62D", 
    ],
    "mine_rectangular":[
        "Mine_PMD6", "Mine_MS4", "Mine_ML7", "Mine_ML8",
    ],
    "mine_cylinder_vert": [
        "Mine_PTKM1R", "Mine_POM2", "Mine_POMZ2", "Mine_POM3",
        "Mine_OZM72", "Mine_POB_Pilka",
    ],
    "mine_butterfly": ["Mine_PFM1"],
    "mine_directional_standing": [
        "Mine_MON50", "Mine_MON90", "Mine_MOB",
    ],
    "mine_round_directional_standing": [
        "Mine_MON100", "Mine_MON200"
    ],
    "mine_other": ["Mine_Cloth", "Mine_Dzika", "Mine_TM83"],
}
MERGED_IDS = {name: idx for idx, name in enumerate(MERGED_GROUPS.keys())}
TYPE_TO_MERGED = {t: name for name, types in MERGED_GROUPS.items() for t in types}

# Semantic class names used by the mine prims (strip the "Mine_" prefix),
# e.g. "TM62D", "PMN2". Used by the thermal camera fallback.
MINE_LABEL_NAMES = {t[len("Mine_"):] for t in EXACT_MINE_TYPES}

_SETTINGS_PREFIX = "/ext/my_extension/dataset_"


class DatasetRecorder:
    def __init__(self, camera_prim_path=None):
        self.camera_prim_path = camera_prim_path
        self._recording = False
        self._session = None
        self._last_capture = 0.0

    # --- Public API ---------------------------------------------------------

    def tick(self):
        settings = carb.settings.get_settings()
        enabled = settings.get_as_bool(_SETTINGS_PREFIX + "capture_enabled") or False

        if enabled and not self._recording:
            self._start(settings)
        elif enabled and self._recording:
            self._maybe_capture(settings)
        elif not enabled and self._recording:
            self._stop()

    def stop(self):
        self._stop()

    @property
    def is_recording(self):
        return self._recording

    # --- Lifecycle ----------------------------------------------------------

    def _start(self, settings):
        if self._recording:
            return

        cam_path = (
            settings.get("/ext/my_extension/camera_path")
            or self.camera_prim_path
            or "/World/auto_record_camera"
        )
        if not cam_path:
            carb.log_warn("Dataset: no camera path configured.")
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(cam_path).IsValid():
            carb.log_warn(f"Dataset: camera prim not found at {cam_path}.")
            return

        name = settings.get(_SETTINGS_PREFIX + "name") or "dataset"
        base_dir = settings.get(_SETTINGS_PREFIX + "base_dir") or self._default_base_dir()
        high_res = settings.get_as_int(_SETTINGS_PREFIX + "high_res") or 4096
        crop_size = settings.get_as_int(_SETTINGS_PREFIX + "crop_size") or 640
        crop_size = min(crop_size, high_res)
        min_visible = settings.get_as_float(_SETTINGS_PREFIX + "min_visible")
        if min_visible is None:
            min_visible = 0.3
        min_visible = max(0.0, min(1.0, min_visible))
        if not settings.get_as_bool(_SETTINGS_PREFIX + "visibility_filter_enabled"):
            min_visible = 0.0

        out_root = os.path.join(base_dir, name)
        images_dir = os.path.join(out_root, "images")
        labels_dir = os.path.join(out_root, "labels")
        exact_dir = os.path.join(out_root, "exact_labels")

        try:
            for d in (images_dir, labels_dir, exact_dir):
                os.makedirs(d, exist_ok=True)
        except Exception as e:
            carb.log_error(f"Dataset: cannot create folders: {e}")
            return

        try:
            rp_path = rep.create.render_product(cam_path, (high_res, high_res), name="dataset_rp")
        except Exception as e:
            carb.log_error(f"Dataset: cannot create render product: {e}")
            return

        seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        seg.attach([rp_path])
        rgb.attach([rp_path])

        self._session = {
            "rp": rp_path,
            "seg": seg,
            "rgb": rgb,
            "images_dir": images_dir,
            "labels_dir": labels_dir,
            "exact_dir": exact_dir,
            "high_res": high_res,
            "crop": crop_size,
            "min_visible": min_visible,
            "name": name,
            "counter": 0,
        }
        self._aabb_cache = {}
        self._bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        self._proj_valid = None
        self._proj_warned = False
        self._last_capture = time.time()
        self._recording = True
        self._write_classes()
        carb.log_info(f"Dataset: recording started -> {out_root}")

    def _stop(self):
        session = self._session
        if session is None:
            return
        self._recording = False
        self._session = None

        try:
            session["seg"].detach()
            session["rgb"].detach()
        except Exception as e:
            carb.log_warn(f"Dataset: annotator detach failed ({e})")

        try:
            rp = rep.get.render_product(session["rp"])
            if rp is not None:
                rp.destroy()
        except Exception as e:
            carb.log_warn(f"Dataset: render product destroy failed ({e})")

        carb.log_info("Dataset: recording stopped.")

    def _write_classes(self):
        session = self._session
        with open(os.path.join(session["labels_dir"], "classes.txt"), "w") as f:
            f.write("\n".join(MERGED_GROUPS.keys()) + "\n")
        with open(os.path.join(session["exact_dir"], "classes.txt"), "w") as f:
            f.write("\n".join(EXACT_MINE_TYPES) + "\n")

    # --- Capture ------------------------------------------------------------

    def _maybe_capture(self, settings):
        rate = settings.get_as_float(_SETTINGS_PREFIX + "rate") or 2.0
        interval = 1.0 / rate if rate > 0 else 1.0
        now = time.time()
        if now - self._last_capture >= interval:
            self._last_capture = now
            if not self._instance_data_ready():
                session = self._session
                session["warmup_frames"] = session.get("warmup_frames", 0) + 1
                if session["warmup_frames"] == 30:
                    carb.log_warn(
                        "Dataset: instance segmentation not producing data yet "
                        "(scene semantics may be missing)."
                    )
                return
            self._capture()

    def _instance_data_ready(self):
        """True once the new render product has rendered a frame with instance data.

        The syntheticdata mapping node needs a few frames to build its label map
        after a render product is created, otherwise it logs
        'missing filteredLabelMap semLabelMap in the history' and returns empty data.
        """
        try:
            seg_data = self._session["seg"].get_data()
            if not seg_data or "data" not in seg_data or seg_data["data"].size == 0:
                return False
            instance_pixels = seg_data["data"].squeeze()
            if instance_pixels.size == 0:
                return False
            info = seg_data.get("info") or {}
            id_to_labels = info.get("idToLabels", {})
            if not id_to_labels:
                return False
            return int(np.max(instance_pixels)) > 0
        except Exception:
            return False

    def _capture(self):
        session = self._session
        if session is None:
            return
        try:
            seg_data = session["seg"].get_data()
            if not seg_data or "data" not in seg_data or seg_data["data"].size == 0:
                return

            instance_pixels = seg_data["data"].squeeze()
            id_to_labels = seg_data["info"].get("idToLabels", {})

            rgb = self._unwrap(session["rgb"].get_data())
            if rgb is None:
                return

            high = session["high_res"]
            crop = session["crop"]
            cx0 = (high - crop) // 2
            cy0 = (high - crop) // 2

            rgb = np.squeeze(rgb)
            if rgb.ndim != 3 or rgb.shape[0] != high or rgb.shape[1] != high:
                carb.log_warn("Dataset: unexpected rgb shape, skipping frame.")
                return

            img_crop = rgb[cy0:cy0 + crop, cx0:cx0 + crop]
            if img_crop.shape[2] >= 3:
                img_bgr = cv2.cvtColor(img_crop[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_crop[:, :, :3]

            lines_merged = []
            lines_exact = []

            settings = carb.settings.get_settings()
            proj = None
            if session["min_visible"] > 0.0 and not (self._proj_valid is False):
                try:
                    proj = self._camera_projection(settings, high)
                except Exception:
                    proj = None
            first = True

            for inst_id, prim_path in id_to_labels.items():
                try:
                    inst_id = int(inst_id)
                except (TypeError, ValueError):
                    continue
                if not isinstance(prim_path, str) or "obstacle_mine" not in prim_path:
                    continue

                mine_type = self._parse_mine_type(prim_path)
                if mine_type is None:
                    continue

                mask = instance_pixels == inst_id
                if not np.any(mask):
                    continue
                ys, xs = np.where(mask)

                if proj is not None:
                    if first and self._proj_valid is None:
                        ok = self._check_projection(proj, prim_path, xs, ys, high)
                        self._proj_valid = True if ok is None else ok
                        if not self._proj_valid and not self._proj_warned:
                            carb.log_warn(
                                "Dataset: camera projection mismatch detected, "
                                "disabling min-visibility filter."
                            )
                            self._proj_warned = True
                    if self._proj_valid and self._below_min_visible(
                        proj, session, prim_path, mask, cx0, cy0
                    ):
                        continue
                first = False

                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())

                x_min -= cx0
                x_max -= cx0
                y_min -= cy0
                y_max -= cy0

                x_min = max(0, min(x_min, crop - 1))
                x_max = max(0, min(x_max, crop - 1))
                y_min = max(0, min(y_min, crop - 1))
                y_max = max(0, min(y_max, crop - 1))
                if x_max <= x_min or y_max <= y_min:
                    continue

                quad = self._to_quad(x_min, y_min, x_max, y_max, crop)

                merged_id = self._merged_id(mine_type)
                exact_id = self._exact_id(mine_type)
                if merged_id is not None:
                    lines_merged.append(f"{merged_id} {quad}")
                if exact_id is not None:
                    lines_exact.append(f"{exact_id} {quad}")

            counter = session["counter"]
            scheduled = settings.get("/ext/my_extension/auto_record_mine") or ""
            mine = scheduled
            season = settings.get("/ext/my_extension/auto_record_season") or ""
            daytime = settings.get("/ext/my_extension/auto_record_daytime") or ""
            fname = "_".join(
                p for p in (
                    session["name"],
                    f"{counter:06d}",
                    mine,
                    season,
                    daytime,
                ) if p
            )
            cv2.imwrite(os.path.join(session["images_dir"], fname + ".png"), img_bgr)

            with open(os.path.join(session["labels_dir"], fname + ".txt"), "w") as f:
                f.write("\n".join(lines_merged) + ("\n" if lines_merged else ""))
            with open(os.path.join(session["exact_dir"], fname + ".txt"), "w") as f:
                f.write("\n".join(lines_exact) + ("\n" if lines_exact else ""))

            session["counter"] = counter + 1
        except Exception as e:
            carb.log_error(f"Dataset: capture error: {e}")

    # --- Helpers ------------------------------------------------------------

    def _camera_path(self, settings):
        return (
            settings.get("/ext/my_extension/camera_path")
            or self.camera_prim_path
            or "/World/auto_record_camera"
        )

    def _camera_projection(self, settings, high):
        """Build a view-projection for the recording camera, or None.

        Returns dict with a world->camera matrix (Gf.Matrix4d) and a focal
        length in pixels, assuming a square render product and USD camera
        convention (camera looks down -Z, image row 0 is the top of the frame).
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(self._camera_path(settings))
        if prim is None or not prim.IsValid():
            return None
        cam = UsdGeom.Camera(prim)
        focal = cam.GetFocalLengthAttr().Get() or 15.0
        haperture = cam.GetHorizontalApertureAttr().Get() or 20.944
        if focal <= 0.0 or haperture <= 0.0:
            return None
        hfov = 2.0 * math.atan(haperture / (2.0 * focal))
        f_px = (high / 2.0) / math.tan(hfov / 2.0)
        try:
            world = omni.usd.get_world_transform_matrix(prim)
        except Exception:
            return None
        return {
            "view": world.GetInverse(),
            "f_px": f_px,
            "cx": high / 2.0,
            "cy": high / 2.0,
        }

    @staticmethod
    def _project(proj, point):
        """Project a world-space point to high-res pixel coords, or None."""
        p = proj["view"].TransformAffine(Gf.Vec3d(point[0], point[1], point[2]))
        z = -p[2]
        if z <= 1e-6:
            return None
        x = proj["cx"] + proj["f_px"] * (p[0] / z)
        y = proj["cy"] - proj["f_px"] * (p[1] / z)
        return (x, y)

    def _mine_world_aabb(self, prim_path):
        """World-space AABB corners of a prim, cached (mines are static)."""
        if prim_path in self._aabb_cache:
            return self._aabb_cache[prim_path]
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            self._aabb_cache[prim_path] = None
            return None
        try:
            bbox = self._bbox_cache.ComputeWorldBound(prim)
            rng = bbox.GetRange()
            mn = rng.GetMin()
            mx = rng.GetMax()
            corners = [
                (mn[0], mn[1], mn[2]),
                (mx[0], mn[1], mn[2]),
                (mn[0], mx[1], mn[2]),
                (mx[0], mx[1], mn[2]),
                (mn[0], mn[1], mx[2]),
                (mx[0], mn[1], mx[2]),
                (mn[0], mx[1], mx[2]),
                (mx[0], mx[1], mx[2]),
            ]
        except Exception:
            corners = None
        self._aabb_cache[prim_path] = corners
        return corners

    def _footprint_area(self, proj, corners, cx0, cy0, crop):
        """Area (px) of the projected AABB intersected with the crop box."""
        xs = []
        ys = []
        for corner in corners:
            px = self._project(proj, corner)
            if px is None:
                continue
            xs.append(px[0])
            ys.append(px[1])
        if not xs:
            return 0
        x0 = max(cx0, min(xs))
        x1 = min(cx0 + crop, max(xs))
        y0 = max(cy0, min(ys))
        y1 = min(cy0 + crop, max(ys))
        w = x1 - x0
        h = y1 - y0
        if w <= 0 or h <= 0:
            return 0
        return w * h

    def _below_min_visible(self, proj, session, prim_path, mask, cx0, cy0):
        corners = self._mine_world_aabb(prim_path)
        if not corners:
            return False
        full = self._footprint_area(proj, corners, cx0, cy0, session["crop"])
        if full <= 0:
            return False
        crop = session["crop"]
        visible = int(mask[cy0:cy0 + crop, cx0:cx0 + crop].sum())
        return visible / float(full) < session["min_visible"]

    def _check_projection(self, proj, prim_path, xs, ys, high):
        """Sanity-check the projection against a visible mask, or None if unknown."""
        corners = self._mine_world_aabb(prim_path)
        if not corners:
            return None
        min_c = [min(c[i] for c in corners) for i in range(3)]
        max_c = [max(c[i] for c in corners) for i in range(3)]
        center = ((min_c[0] + max_c[0]) / 2.0,
                  (min_c[1] + max_c[1]) / 2.0,
                  (min_c[2] + max_c[2]) / 2.0)
        pp = self._project(proj, center)
        if pp is None:
            return None
        cxm = (int(xs.min()) + int(xs.max())) / 2.0
        cym = (int(ys.min()) + int(ys.max())) / 2.0
        tolerance = max(high * 0.25, 80.0)
        return math.hypot(pp[0] - cxm, pp[1] - cym) <= tolerance

    @staticmethod
    def _unwrap(annotator_data):
        if isinstance(annotator_data, dict):
            return annotator_data.get("data", annotator_data)
        return annotator_data

    @staticmethod
    def _parse_mine_type(prim_path):
        name = prim_path.lower()
        matches = [t for t in EXACT_MINE_TYPES if t.lower() in name]
        if not matches:
            return None
        return max(matches, key=len)

    @staticmethod
    def _merged_id(mine_type):
        merged_name = TYPE_TO_MERGED.get(mine_type)
        if merged_name is None:
            return None
        return MERGED_IDS[merged_name]

    @staticmethod
    def _exact_id(mine_type):
        return EXACT_IDS.get(mine_type)

    @staticmethod
    def _to_quad(x_min, y_min, x_max, y_max, size):
        x_min_n = x_min / size
        x_max_n = x_max / size
        y_min_n = y_min / size
        y_max_n = y_max / size
        return (
            f"{x_min_n:.6f} {y_min_n:.6f} "
            f"{x_max_n:.6f} {y_min_n:.6f} "
            f"{x_max_n:.6f} {y_max_n:.6f} "
            f"{x_min_n:.6f} {y_max_n:.6f}"
        )

    @staticmethod
    def _default_base_dir():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(script_dir, "../../../../../../datasets"))
