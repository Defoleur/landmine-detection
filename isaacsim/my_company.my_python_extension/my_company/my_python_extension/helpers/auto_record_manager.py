import os
import json
import time
import math
import random
from datetime import datetime

import carb
import omni.timeline
import omni.usd
import omni.kit.app
from pxr import Gf, Usd, UsdGeom

from .dataset_recorder import DatasetRecorder, _ALL_MINE_TYPES

_PREFIX = "/ext/my_extension/"
_AUTO_ENABLED = _PREFIX + "auto_record_enabled"
_AUTO_STATUS = _PREFIX + "auto_record_status"
_AUTO_SEASONS = _PREFIX + "auto_record_seasons"
_AUTO_DAYTIMES = _PREFIX + "auto_record_daytimes"
_AUTO_SHOTS = _PREFIX + "auto_record_shots_per_mine"
_AUTO_FOLLOW_VIEWPORT = _PREFIX + "auto_record_follow_viewport"
_AUTO_SPEED_ENABLED = _PREFIX + "auto_record_speed_enabled"
_AUTO_SPEED_MIN = _PREFIX + "auto_record_speed_min"
_AUTO_SPEED_MAX = _PREFIX + "auto_record_speed_max"
_AUTO_MINE = _PREFIX + "auto_record_mine"
_AUTO_MINES = _PREFIX + "auto_record_mines"
_AUTO_SEASON = _PREFIX + "auto_record_season"
_AUTO_DAYTIME = _PREFIX + "auto_record_daytime"
_AUTO_CLOSE_ON_FINISH = _PREFIX + "auto_record_close_on_finish"
_TRAJ_RECORDING = _PREFIX + "traj_record_enabled"
_TRAJ_NAME = _PREFIX + "traj_name"
_PREVIEW_ENABLED = _PREFIX + "traj_preview_enabled"
_PREVIEW_INTERVAL = _PREFIX + "traj_preview_interval"
_PREVIEW_STATUS = _PREFIX + "traj_preview_status"
_PREVIEW_MINE = _PREFIX + "traj_preview_mine"
_PREVIEW_TRAJ = _PREFIX + "traj_preview_traj"
_CREATE_CAMERA = _PREFIX + "traj_create_camera"
_RESET_CAMERA = _PREFIX + "traj_reset_camera"

DEFAULT_SEASONS = "summer,autumn,winter,spring"
DEFAULT_DAYTIMES = "day,morning,evening"

SETTLE_AFTER_SEASON = 3.0
SETTLE_AFTER_DAYTIME = 2.0

PREVIEW_INTERVAL_DEFAULT = 5.0
PREVIEW_DURATION = 10.0  # seconds per trajectory

SPEED_DEFAULT_MIN = 0.7
SPEED_DEFAULT_MAX = 1.3

# Known mine type strings (lowercased), used to recover a mine's type from its
# prim name via longest-prefix match. Thermal TM-62 is spawned under its own
# name "obstacle_mine_tm_62_thermal" (underscores between tm and 62).
_MINE_TYPE_STRINGS = sorted(
    {t.lower() for t in _ALL_MINE_TYPES}
    | {"Mine_TM62_Thermal".lower(), "mine_tm_62_thermal"}
)

MINE_NAME_HINT = "obstacle_mine"
LOOK_ABOVE = 0.10

AUTO_CAM_PATH = "/World/auto_record_camera"
AUTO_CAM_FOCAL_LENGTH = 15.0

# Spawn pose for the standalone preview camera: position (0, 0, 3) with
# orientation Euler (0, 0, -90) deg (rotation about Z by -90 deg), matching
# the recorded trajectory start pose.  Gf uses row-vector convention, so the
# world matrix is R * T: this makes ExtractTranslation() return the exact
# position (T * R would rotate the position by the orientation).
_AUTO_CAM_ORIENT = Gf.Quatd(0.7071067811865476, 0.0, 0.0, -0.7071067811865475)
AUTO_CAM_INITIAL_WORLD = Gf.Matrix4d().SetRotate(_AUTO_CAM_ORIENT) * \
    Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, 3.0))


class AutoRecordManager:
    """Drives automatic dataset recording.

    Loop: season -> daytime -> every landmine (random trajectory flight).
    The existing camera prim from ``/ext/my_extension/camera_path`` is flown
    freely through the world (no drone movement, no extra cameras) while the
    DatasetRecorder captures frames from the same prim.
    """

    def __init__(self):
        self._active = False
        self._phase = "idle"  # idle | plan | flight
        self._plan = []
        self._plan_index = 0
        self._mine_positions = []
        self._trajectories = []
        self._mine_traj_map = {}
        self._flight = None
        self._wait_until = 0.0
        self._prev_stabilized = False
        self._prev_camera_path = None
        self._recorded_season = None
        self._dataset_base = None

        self._traj_recording = False
        self._traj_samples = []
        self._traj_record_start = 0.0

        self._preview_active = False
        self._preview_trajectories = []
        self._preview_index = 0
        self._preview_wait_until = 0.0
        self._preview_mine_pos = None
        self._preview_log = []
        self._preview_log_fh = None
        self._preview_log_path = None
        self._last_pose_log = 0.0

        self._last_status = ""
        self._status_flush = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self, dt):
        settings = carb.settings.get_settings()
        dt = self._as_dt(dt)

        if settings.get_as_bool(_CREATE_CAMERA) or False:
            settings.set(_CREATE_CAMERA, False)
            self._ensure_auto_cam()
            self._set_preview_status(settings, "Preview camera created")

        if settings.get_as_bool(_RESET_CAMERA) or False:
            settings.set(_RESET_CAMERA, False)
            self._reset_camera(settings)

        self._handle_manual_trajectory_recording(settings)

        sim_active = self._is_simulation_active()
        preview_enabled = settings.get_as_bool(_PREVIEW_ENABLED) or False

        if preview_enabled:
            if settings.get_as_bool(_AUTO_ENABLED) or False:
                settings.set(_AUTO_ENABLED, False)
            settings.set("/ext/my_extension/dataset_capture_enabled", False)
            if not self._preview_active:
                self._begin_preview(settings)
            if self._preview_active:
                if sim_active:
                    try:
                        self._preview_step(settings, dt)
                    except Exception as e:
                        carb.log_error(f"TrajPreview: step error: {e}")
                        self._stop_preview(settings)
                else:
                    self._set_preview_waiting(settings)
            return

        if self._preview_active:
            self._stop_preview(settings)

        enabled = settings.get_as_bool(_AUTO_ENABLED) or False
        if enabled and not self._active:
            if sim_active:
                self._begin(settings)
            else:
                self._set_waiting_status(settings)
            return
        elif not enabled and self._active:
            self._abort(settings)

        if self._active:
            if not sim_active:
                self._set_waiting_status(settings)
                return
            try:
                self._step(settings, dt)
            except Exception as e:
                carb.log_error(f"AutoRecord: step error: {e}")
                self._abort(settings)

    def stop(self):
        settings = carb.settings.get_settings()
        self._abort(settings)
        self._stop_dataset()
        self._stop_preview(settings)

    def _is_simulation_active(self):
        timeline = omni.timeline.get_timeline_interface()
        settings = carb.settings.get_settings()
        return timeline.is_playing() or (settings.get_as_bool("/ext/pegasus/custom_30hz_active") or False)

    @staticmethod
    def _as_dt(dt, default=1.0 / 30.0):
        """Coerce a tick payload into a numeric dt.

        Update-stream callbacks hand over a ``carb.events.IEvent`` (or a bare
        float in older Kit versions); flight timers need a plain number.
        """
        if isinstance(dt, (int, float)) and not isinstance(dt, bool):
            return float(dt)
        payload = getattr(dt, "payload", None)
        if isinstance(payload, dict):
            value = payload.get("dt", None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return default

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _begin(self, settings):
        self._ensure_auto_cam()
        mines = self._discover_mine_prims()
        if not mines:
            carb.log_warn("AutoRecord: no mines found in stage, aborting.")
            settings.set(_AUTO_ENABLED, False)
            self._set_status(settings, "No mines found")
            return
        self._mine_positions = [m["pos"] for m in mines]
        self._mines = mines
        settings.set(_AUTO_MINES, ",".join(m["name"] for m in mines))

        seasons = self._parse_csv(settings.get(_AUTO_SEASONS) or DEFAULT_SEASONS)
        daytimes = self._parse_csv(settings.get(_AUTO_DAYTIMES) or DEFAULT_DAYTIMES)
        if not seasons or not daytimes:
            carb.log_warn("AutoRecord: seasons/daytimes empty, aborting.")
            settings.set(_AUTO_ENABLED, False)
            self._set_status(settings, "No seasons/daytimes")
            return

        self._trajectories = self._load_trajectories(settings)
        if not self._trajectories:
            carb.log_warn("AutoRecord: trajectory library empty, aborting.")
            settings.set(_AUTO_ENABLED, False)
            self._set_status(settings, "No trajectories")
            return

        self._mine_traj_map = self._assign_mine_trajectories(mines, seasons, daytimes)
        self._plan = self._build_plan(seasons, daytimes, len(self._mine_positions))

        self._plan_index = 0
        self._phase = "plan"
        self._active = True
        self._wait_until = 0.0
        self._recorded_season = None
        self._recorded_daytime = None
        self._current_mine = ""

        self._dataset_user_active = settings.get_as_bool("/ext/my_extension/dataset_capture_enabled") or False

        if settings.get(_AUTO_FOLLOW_VIEWPORT) is None:
            settings.set(_AUTO_FOLLOW_VIEWPORT, True)
        self._prev_stabilized = settings.get_as_bool("/ext/my_extension/camera_stabilized") or False
        settings.set("/ext/my_extension/camera_stabilized", False)

        self._prev_camera_path = settings.get("/ext/my_extension/camera_path") or "/World/quadrotor/body/camera_022"
        settings.set("/ext/my_extension/camera_path", AUTO_CAM_PATH)

        carb.log_info(
            f"AutoRecord: started, {len(self._mine_positions)} mines, "
            f"{len(seasons)} seasons x {len(daytimes)} daytimes."
        )
        top_n = sum(1 for t in self._mine_traj_map.values() if t["category"] == "top")
        ap_n = sum(1 for t in self._mine_traj_map.values() if t["ap"])
        carb.log_info(
            f"AutoRecord: trajectory assignment -> top {top_n}, "
            f"side {len(self._mine_traj_map) - top_n}, ap {ap_n} "
            f"({len(seasons) * len(daytimes)} context(s) per mine)."
        )
    

    # ------------------------------------------------------------------
    # Recording start/stop (mirrors UI buttons)
    # ------------------------------------------------------------------

    def _stop_dataset(self, settings):
        settings.set("/ext/my_extension/dataset_capture_enabled", False)

    def _publish_context(self, settings):
        """Expose current mine/season/daytime so DatasetRecorder can embed them
        in each captured file name. Empty values fall back to base naming."""
        settings.set(_AUTO_MINE, getattr(self, "_current_mine", "") or "")
        settings.set(_AUTO_SEASON, getattr(self, "_recorded_season", "") or "")
        settings.set(_AUTO_DAYTIME, getattr(self, "_recorded_daytime", "") or "")

    def _clear_context(self, settings):
        settings.set(_AUTO_MINE, "")
        settings.set(_AUTO_MINES, "")
        settings.set(_AUTO_SEASON, "")
        settings.set(_AUTO_DAYTIME, "")


    def _abort(self, settings):
        if not getattr(self, "_dataset_user_active", False):
            self._stop_dataset(settings)
        self._active = False
        self._phase = "idle"
        self._plan = []
        self._flight = None
        self._recorded_season = None
        self._recorded_daytime = None
        self._current_mine = ""
        self._clear_context(settings)
        settings.set("/ext/my_extension/camera_stabilized", self._prev_stabilized)
        if getattr(self, "_prev_camera_path", None):
            settings.set("/ext/my_extension/camera_path", self._prev_camera_path)
        self._set_status(settings, "Stopped")
        carb.log_info("AutoRecord: stopped.")

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def _step(self, settings, dt):
        if self._phase == "flight":
            if self._flight is not None and self._flight_step(settings, dt):
                self._flight = None
                self._phase = "plan"
                self._plan_index += 1
            return

        now = time.time()
        if now < self._wait_until:
            return

        if self._plan_index >= len(self._plan):
            self._finish(settings)
            return

        kind, value = self._plan[self._plan_index]
        if kind == "season":
            self._apply_season_action(settings, value)
        elif kind == "daytime":
            self._apply_daytime_action(settings, value)
        elif kind == "mine":
            self._start_mine_flight(settings, value)
        else:
            self._plan_index += 1

    def _apply_season_action(self, settings, season):
        try:
            from pegasus.simulator.helpers.environment_manager import EnvironmentManager

            env = EnvironmentManager()
            env.set_stage()
            env.apply_season(season, replace_trees=True)
        except Exception as e:
            carb.log_warn(f"AutoRecord: apply_season failed: {e}")

        # Dataset naming/folder logic is handled by the DatasetRecorder; the
        # per-image filename carries the season/timeofday/mine context.

        self._recorded_season = season
        self._publish_context(settings)
        self._set_status(settings, f"Season: {season} (ready)")
        self._wait_until = time.time() + SETTLE_AFTER_SEASON
        self._plan_index += 1
        carb.log_info(f"AutoRecord: season '{season}' -> dataset (recorder-managed)")

    def _apply_daytime_action(self, settings, daytime):
        try:
            from pegasus.simulator.helpers.environment_manager import EnvironmentManager

            env = EnvironmentManager()
            env.set_stage()
            env.apply_daytime(daytime)
        except Exception as e:
            carb.log_warn(f"AutoRecord: apply_daytime failed: {e}")

        self._recorded_daytime = daytime
        self._publish_context(settings)
        self._set_status(settings, f"Season: {self._recorded_season}, Daytime: {daytime}")
        self._wait_until = time.time() + SETTLE_AFTER_DAYTIME
        self._plan_index += 1

    def _start_mine_flight(self, settings, mine_idx):
        traj = self._mine_traj_map.get(
            (mine_idx, self._recorded_season or "", self._recorded_daytime or "")
        )
        if traj is None:
            traj = random.choice(self._trajectories)
        mine_pos = self._mine_positions[mine_idx]
        self._current_mine = ""
        if 0 <= mine_idx < len(self._mines):
            self._current_mine = self._mines[mine_idx].get("name", "")
        self._publish_context(settings)

        world_wps = []
        for wp in traj["waypoints"]:
            pos = [mine_pos[0] + wp["pos"][0], mine_pos[1] + wp["pos"][1], mine_pos[2] + wp["pos"][2]]
            world_wps.append({"pos": pos, "orient": list(wp["orient"]), "time": wp["time"]})

        world_wps = self._trim_waypoints(world_wps)

        rate = settings.get_as_float("/ext/my_extension/dataset_rate") or 2.0
        shots = max(1, settings.get_as_int(_AUTO_SHOTS) or 10)
        if len(world_wps) >= 2:
            duration = world_wps[-1]["time"]
        else:
            duration = 1.0
        if duration <= 0.0:
            duration = max(0.5, shots / max(rate, 0.1))

        speed = 1.0
        if settings.get_as_bool(_AUTO_SPEED_ENABLED) or False:
            lo = settings.get_as_float(_AUTO_SPEED_MIN) or SPEED_DEFAULT_MIN
            hi = settings.get_as_float(_AUTO_SPEED_MAX) or SPEED_DEFAULT_MAX
            if hi < lo:
                lo, hi = hi, lo
            speed = random.uniform(lo, hi)
        if speed > 0.0:
            duration = duration / speed

        self._flight = {
            "waypoints": world_wps,
            "duration": duration,
            "elapsed": 0.0,
        }
        self._phase = "flight"
        self._set_status(settings, f"Mine {mine_idx + 1}/{len(self._mine_positions)} - {traj['name']} ({speed:g}x)")

    @staticmethod
    def _trim_waypoints(wps):
        """Remove the stationary lead-in and tail-out from a trajectory.

        Recorded trajectories park the camera at a fixed pose for a while at
        the start and again at the end; that reads as a ~1s 'default position
        near the mine' hold between flights. Stripping those runs (and rebasing
        the remaining times to start at 0) makes each flight pure motion.
        """
        wps = [dict(w) for w in wps]
        if len(wps) < 2:
            return wps
        while len(wps) > 1 and wps[-1]["pos"] == wps[-2]["pos"]:
            wps.pop()
        if not wps:
            return wps
        start_pos = wps[0]["pos"]
        first = 1
        while first < len(wps) and wps[first]["pos"] == start_pos:
            first += 1
        if first >= len(wps):
            return [wps[-1]]
        wps = wps[first:]
        t0 = wps[0]["time"]
        for w in wps:
            w["time"] -= t0
        return wps

    def _flight_step(self, settings, dt):
        f = self._flight
        f["elapsed"] += dt
        if f["elapsed"] >= f["duration"]:
            final = f["waypoints"][-1]
            self._set_camera_pose(settings, final["pos"], final["orient"])
            return True

        pos, orient = self._sample_timed(f["waypoints"], f["elapsed"])
        self._set_camera_pose(settings, pos, orient)
        return False

    def _finish(self, settings):
        if not getattr(self, "_dataset_user_active", False):
            self._stop_dataset(settings)
        self._active = False
        self._phase = "idle"
        self._flight = None
        self._plan = []
        self._recorded_season = None
        self._recorded_daytime = None
        self._current_mine = ""
        self._clear_context(settings)
        settings.set("/ext/my_extension/camera_stabilized", self._prev_stabilized)
        if getattr(self, "_prev_camera_path", None):
            settings.set("/ext/my_extension/camera_path", self._prev_camera_path)
        settings.set(_AUTO_ENABLED, False)
        self._set_status(settings, "Finished")
        carb.log_info("AutoRecord: finished.")
        if settings.get_as_bool(_AUTO_CLOSE_ON_FINISH) or False:
            carb.log_info("AutoRecord: closing Isaac Sim as requested.")
            omni.kit.app.get_app().post_quit()

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def _resolve_cam_path(self, settings):
        stage = omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath(AUTO_CAM_PATH).IsValid():
            return AUTO_CAM_PATH
        return settings.get("/ext/my_extension/camera_path") or "/World/quadrotor/body/camera_022"

    def _camera_op_order(self, cam_path):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            return None
        return [str(op.GetOpType()).split(".")[-1] for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]

    def _ensure_auto_cam(self):
        """Create a standalone camera prim at world root for trajectory flights.

        Not parented under the drone body, so its world pose can be set
        directly without fighting the drone's own transform.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(AUTO_CAM_PATH)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            order = [op.GetOpType() for op in ops]
            expected = [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient]
            if order != expected:
                xformable.ClearXformOpOrder()
                t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                r_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                world = AUTO_CAM_INITIAL_WORLD
                t_op.Set(world.ExtractTranslation(), Usd.TimeCode.Default())
                r_op.Set(world.ExtractRotation().GetQuat(), Usd.TimeCode.Default())
                carb.log_info(f"AutoRecord: repaired xform op order of {AUTO_CAM_PATH}.")
            return
        from pxr import Sdf

        cam = UsdGeom.Camera.Define(stage, Sdf.Path(AUTO_CAM_PATH))
        cam.CreateFocalLengthAttr(AUTO_CAM_FOCAL_LENGTH)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))

        world = AUTO_CAM_INITIAL_WORLD

        xformable = UsdGeom.Xformable(cam.GetPrim())
        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        r_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        t_op.Set(world.ExtractTranslation(), Usd.TimeCode.Default())
        r_op.Set(world.ExtractRotation().GetQuat(), Usd.TimeCode.Default())
        carb.log_info(f"AutoRecord: created standalone camera at {AUTO_CAM_PATH}.")

    def _reset_camera(self, settings):
        """Snap the preview camera back to its default spawn pose."""
        self._ensure_auto_cam()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(AUTO_CAM_PATH)
        if not prim or not prim.IsValid():
            return
        world = AUTO_CAM_INITIAL_WORLD
        t = world.ExtractTranslation()
        r = world.ExtractRotation().GetQuat()
        orient = [r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2]]
        self._set_camera_pose(settings, list(t), orient)
        settings.set("/ext/my_extension/camera_path", AUTO_CAM_PATH)
        self._set_preview_status(settings, "Camera reset to default pose")
        carb.log_info("AutoRecord: camera reset to default pose.")

    def _set_camera_pose(self, settings, pos, orient):
        cam_path = self._resolve_cam_path(settings)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            return

        orient_list = list(orient)
        orient = Gf.Quatd(orient_list[0], Gf.Vec3d(orient_list[1], orient_list[2], orient_list[3]))
        # Gf is row-vector: R * T so that ExtractTranslation() == pos exactly.
        world = Gf.Matrix4d().SetRotate(orient) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*pos))

        parent = prim.GetParent()
        if parent is not None and parent.IsValid():
            parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        else:
            parent_world = Gf.Matrix4d().SetIdentity()
        local = parent_world.GetInverse() * world

        xformable = UsdGeom.Xformable(prim)
        t_op = None
        r_op = None
        ops = xformable.GetOrderedXformOps()
        order = [op.GetOpType() for op in ops]
        expected = [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient]
        if order != expected:
            carb.log_info(
                f"TrajPreview: fixing camera xform op order {order} -> [translate, orient] for {cam_path}."
            )
            xformable.ClearXformOpOrder()
            t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            r_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        else:
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    t_op = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    r_op = op
        if t_op is None:
            t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        if r_op is None:
            r_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        t = local.ExtractTranslation()
        r = local.ExtractRotation().GetQuat()
        # Author at the default time (static value), NOT a frame-0 sample, so
        # the Isaac Sim camera gizmo can still edit the camera afterwards.
        # Clear any stale time samples authored by older versions.
        t_op.GetAttr().Clear()
        r_op.GetAttr().Clear()
        time_code = Usd.TimeCode.Default()
        if t_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            t_op.Set(Gf.Vec3f(*t), time_code)
        else:
            t_op.Set(Gf.Vec3d(*t), time_code)
        if r_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            r_op.Set(Gf.Quatf(float(r.GetReal()), Gf.Vec3f(*r.GetImaginary())), time_code)
        else:
            r_op.Set(Gf.Quatd(r.GetReal(), r.GetImaginary()), time_code)

        if self._preview_active and self._preview_log_fh is not None:
            now = time.time()
            if now - self._last_pose_log < 0.25:
                return
            self._last_pose_log = now
            actual = self._camera_world_pos(settings)
            drone = self._drone_world_pos(settings)
            self._preview_log_entry({
                "event": "pose",
                "time": time.time(),
                "cam_path": cam_path,
                "parent": prim.GetPath().pathString.rsplit("/", 1)[0],
                "target_pos": list(pos),
                "target_orient": orient_list,
                "actual_world_pos": list(actual) if actual else None,
                "world_to_target_delta": [round(actual[k] - pos[k], 4) for k in range(3)] if actual else None,
                "drone_pos": list(drone) if drone else None,
            })

    # ------------------------------------------------------------------
    # Trajectory library
    # ------------------------------------------------------------------

    def _trajectory_dir(self):
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/camera_trajectories_custom"))

    def _load_trajectories(self, settings):
        traj_dir = self._trajectory_dir()
        os.makedirs(traj_dir, exist_ok=True)

        trajectories = []
        for fname in sorted(os.listdir(traj_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(traj_dir, fname)
            try:
                with open(path) as f:
                    data = json.load(f)
                wps = data.get("waypoints", [])
                if not wps:
                    continue
                parsed = []
                for wp in wps:
                    pos = list(wp.get("pos", [0.0, 0.0, 0.0]))
                    orient = list(wp.get("orient", [1.0, 0.0, 0.0, 0.0]))
                    t = float(wp.get("time", 0.0))
                    parsed.append({"pos": pos, "orient": orient, "time": t})
                name = os.path.splitext(fname)[0]
                trajectories.append({
                    "name": name,
                    "category": "top" if "top" in name else "side",
                    "ap": "ap" in name,
                    "waypoints": parsed,
                })
            except Exception as e:
                carb.log_warn(f"AutoRecord: failed loading trajectory {path}: {e}")

        carb.log_info(f"AutoRecord: loaded {len(trajectories)} trajectories from {traj_dir}.")
        return trajectories

    # ------------------------------------------------------------------
    # Manual trajectory authoring
    # ------------------------------------------------------------------

    def _handle_manual_trajectory_recording(self, settings):
        enabled = settings.get_as_bool(_TRAJ_RECORDING) or False
        if enabled and not self._traj_recording:
            self._traj_recording = True
            self._traj_samples = []
            self._traj_last_sample = 0.0
            self._traj_record_start = time.time()
            pose = self._read_viewport_camera_pose()
            if pose is None:
                carb.log_warn("AutoRecord: cannot read viewport camera for trajectory recording.")
                self._traj_recording = False
                settings.set(_TRAJ_RECORDING, False)
                return
            pos, orient = pose
            self._traj_samples = [{"pos": list(pos), "orient": list(orient), "time": 0.0}]
            self._traj_last_sample = time.time()
            carb.log_info("AutoRecord: trajectory recording started (pos + orient + time).")
        elif enabled and self._traj_recording:
            now = time.time()
            if now - self._traj_last_sample >= 0.1:
                pose = self._read_viewport_camera_pose()
                if pose is not None:
                    pos, orient = pose
                    self._traj_samples.append({
                        "pos": list(pos),
                        "orient": list(orient),
                        "time": now - self._traj_record_start,
                    })
                    self._traj_last_sample = now
        elif not enabled and self._traj_recording:
            self._traj_recording = False
            self._save_manual_trajectory(settings)

    def _read_viewport_camera_pose(self):
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                return None
            cam_path = viewport.camera_path
            if not cam_path:
                return None
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(cam_path)
            if not prim or not prim.IsValid():
                return None
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            orient = world.ExtractRotation().GetQuat()
            return (world.ExtractTranslation(), [orient.GetReal(), orient.GetImaginary()[0], orient.GetImaginary()[1], orient.GetImaginary()[2]])
        except Exception as e:
            carb.log_warn(f"AutoRecord: viewport camera read failed: {e}")
            return None

    def _save_manual_trajectory(self, settings):
        name = settings.get(_TRAJ_NAME) or "trajectory"
        name = "".join(c for c in name if c.isalnum() or c in "_-").strip("_-") or "trajectory"
        if not self._traj_samples:
            carb.log_warn("AutoRecord: nothing recorded, not saving.")
            return

        wps = []
        for s in self._traj_samples:
            wps.append({"pos": list(s["pos"]),
                        "orient": list(s["orient"]),
                        "time": s["time"]})
        data = {"name": name, "preset": False, "waypoints": wps}
        path = os.path.join(self._trajectory_dir(), name + ".json")
        os.makedirs(self._trajectory_dir(), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            carb.log_info(f"AutoRecord: saved trajectory -> {path} ({len(wps)} waypoints)")
        except Exception as e:
            carb.log_warn(f"AutoRecord: failed saving trajectory: {e}")

    # ------------------------------------------------------------------
    # Trajectory preview (test flights without dataset capture)
    # ------------------------------------------------------------------

    def _begin_preview(self, settings):
        self._ensure_auto_cam()
        self._open_preview_log()
        cam_path = self._resolve_cam_path(settings)
        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath(cam_path).IsValid():
            carb.log_warn(f"TrajPreview: camera prim not found at {cam_path}, aborting.")
            settings.set(_PREVIEW_ENABLED, False)
            self._set_preview_status(settings, "Camera prim not found")
            self._close_preview_log()
            return

        self._preview_trajectories = self._load_trajectories(settings)
        selected_traj = settings.get_as_string(_PREVIEW_TRAJ) or ""
        if selected_traj:
            self._preview_trajectories = [t for t in self._preview_trajectories if t["name"] == selected_traj]
        if not self._preview_trajectories:
            carb.log_warn("TrajPreview: trajectory library empty, aborting.")
            settings.set(_PREVIEW_ENABLED, False)
            self._set_preview_status(settings, "No trajectories found")
            self._close_preview_log()
            return

        self._preview_mine_pos = self._pick_preview_mine(settings)
        if self._preview_mine_pos is not None:
            carb.log_info(f"TrajPreview: centering on landmine at {self._preview_mine_pos}.")
        else:
            carb.log_warn("TrajPreview: no landmines found, flying trajectories at origin.")
        self._preview_log_entry({
            "event": "begin",
            "time": time.time(),
            "cam_path": cam_path,
            "cam_op_order": self._camera_op_order(cam_path),
            "mine_pos": list(self._preview_mine_pos) if self._preview_mine_pos else None,
            "selected_mine": settings.get_as_string(_PREVIEW_MINE) or "auto",
            "selected_traj": settings.get_as_string(_PREVIEW_TRAJ) or "all",
            "drone_pos": list(self._drone_world_pos(settings)) if self._drone_world_pos(settings) else None,
            "n_trajectories": len(self._preview_trajectories),
            "trajectory_names": [t["name"] for t in self._preview_trajectories],
        })

        self._preview_index = 0
        self._preview_wait_until = 0.0
        self._preview_active = True
        self._start_preview_flight(settings)
        carb.log_info(f"TrajPreview: started, {len(self._preview_trajectories)} trajectories.")

    def _pick_preview_mine(self, settings):
        mines = self._discover_mine_prims()
        if not mines:
            return None
        sel = settings.get_as_string(_PREVIEW_MINE) or ""
        for m in mines:
            if m["name"] == sel:
                return list(m["pos"])
        ref = self._drone_world_pos(settings)
        if ref is None:
            return list(mines[0]["pos"])
        return list(min(mines, key=lambda m: self._seg_len(m["pos"], list(ref)))["pos"])

    def _drone_world_pos(self, settings):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        for path in ("/World/quadrotor/body", "/World/quadrotor"):
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                try:
                    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    return tuple(world.ExtractTranslation())
                except Exception:
                    return None
        return None

    def _camera_world_pos(self, settings):
        cam_path = self._resolve_cam_path(settings)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        prim = stage.GetPrimAtPath(cam_path)
        if not prim or not prim.IsValid():
            return None
        try:
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            return tuple(world.ExtractTranslation())
        except Exception:
            return None

    def _stop_preview(self, settings):
        if not self._preview_active:
            return
        self._preview_active = False
        self._flight = None
        self._preview_trajectories = []
        self._preview_mine_pos = None
        settings.set("/ext/my_extension/dataset_capture_enabled", False)
        if getattr(self, "_prev_camera_path", None):
            settings.set("/ext/my_extension/camera_path", self._prev_camera_path)
        self._set_preview_status(settings, "Stopped")
        self._preview_log_entry({"event": "end", "time": time.time()})
        self._close_preview_log()
        carb.log_info("TrajPreview: stopped.")

    def _start_preview_flight(self, settings):
        traj = self._preview_trajectories[self._preview_index]
        mine_pos = self._preview_mine_pos
        wps = []
        for wp in traj["waypoints"]:
            if mine_pos is not None:
                pos = [mine_pos[0] + wp["pos"][0], mine_pos[1] + wp["pos"][1], mine_pos[2] + wp["pos"][2]]
            else:
                pos = list(wp["pos"])
            wps.append({"pos": pos, "orient": list(wp["orient"]), "time": wp["time"]})

        duration = wps[-1]["time"] if len(wps) >= 2 else PREVIEW_DURATION
        if duration <= 0.0:
            duration = PREVIEW_DURATION

        self._flight = {
            "waypoints": wps,
            "duration": duration,
            "elapsed": 0.0,
        }
        self._preview_log_entry({
            "event": "flight_start",
            "time": time.time(),
            "trajectory": traj["name"],
            "mine_pos": list(mine_pos) if mine_pos else None,
            "duration": duration,
            "waypoints": wps,
        })
        self._set_preview_status(
            settings,
            f"{self._preview_index + 1}/{len(self._preview_trajectories)}: {traj['name']}",
        )

    def _preview_step(self, settings, dt):
        if self._flight is not None:
            if self._flight_step(settings, dt):
                self._flight = None
                interval = settings.get_as_float(_PREVIEW_INTERVAL) or PREVIEW_INTERVAL_DEFAULT
                self._preview_wait_until = time.time() + interval
                self._set_preview_status(settings, f"Next trajectory in {interval:g}s...")
            return

        now = time.time()
        if now < self._preview_wait_until:
            return

        self._preview_index += 1
        if self._preview_index >= len(self._preview_trajectories):
            self._stop_preview(settings)
            self._set_preview_status(settings, "Finished")
            return
        self._start_preview_flight(settings)

    def _set_preview_status(self, settings, text):
        settings.set(_PREVIEW_STATUS, text)

    def _open_preview_log(self):
        self._preview_log = []
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "preview_logs")
        os.makedirs(log_dir, exist_ok=True)
        fname = datetime.now().strftime("preview_%Y%m%d_%H%M%S.json")
        self._preview_log_path = os.path.join(log_dir, fname)
        self._preview_log_fh = open(self._preview_log_path, "w")
        self._preview_log_fh.write("[\n")
        self._preview_log_fh.flush()
        carb.log_info(f"TrajPreview: logging camera positions to {self._preview_log_path}")

    def _preview_log_entry(self, entry):
        self._preview_log.append(entry)
        if self._preview_log_fh is not None and not self._preview_log_fh.closed:
            self._preview_log_fh.write(json.dumps(entry) + ",\n")
            self._preview_log_fh.flush()

    def _close_preview_log(self):
        if self._preview_log_fh is not None and not self._preview_log_fh.closed:
            self._preview_log_fh.write("{}\n]")
            self._preview_log_fh.close()
        self._preview_log_fh = None
        self._preview_log = []

    def _set_preview_waiting(self, settings):
        now = time.time()
        if now - self._status_flush < 1.0:
            return
        self._status_flush = now
        self._set_preview_status(settings, "Waiting for simulation to start...")

    # ------------------------------------------------------------------
    # Mine discovery
    # ------------------------------------------------------------------

    def _discover_mine_prims(self):
        stage = omni.usd.get_context().get_stage()
        mines = []
        if stage is None:
            return mines
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if MINE_NAME_HINT not in prim.GetPath().pathString:
                continue
            xf = UsdGeom.Xformable(prim)
            if not prim.IsA(UsdGeom.Imageable):
                continue
            world = xf.ComputeLocalToWorldTransform(0.0)
            t = world.ExtractTranslation()
            mines.append({
                "name": prim.GetName(),
                "type": self._mine_type_from_name(prim.GetName()),
                "path": prim.GetPath().pathString,
                "pos": [t[0], t[1], t[2]],
            })
        carb.log_info(f"AutoRecord: discovered {len(mines)} mines.")
        return mines

    @staticmethod
    def _mine_type_from_name(name):
        """Recover the mine type (e.g. \"Mine_TM62\", \"Mine_PMN2\") from the
        prim name (e.g. \"obstacle_mine_tm62_0\") via longest-prefix match."""
        if not name or not name.lower().startswith("obstacle_"):
            return None
        seg = name.lower()[len("obstacle_"):]
        best = None
        for t in _MINE_TYPE_STRINGS:
            if seg == t or seg.startswith(t + "_"):
                if best is None or len(t) > len(best):
                    best = t
        return best

    @staticmethod
    def _mine_is_anti_tank_type(mine_type):
        if not mine_type:
            return False
        t = str(mine_type).lower()
        return "tm" in t or "pt" in t

    def _pick_trajectory_for(self, category, ap):
        pools = [t for t in self._trajectories if t["category"] == category and t["ap"] == ap]
        if not pools:
            pools = [t for t in self._trajectories if t["category"] == category]
        if not pools:
            pools = list(self._trajectories)
        return random.choice(pools)

    def _assign_mine_trajectories(self, mines, seasons, daytimes):
        """Assign one trajectory per (mine, season, daytime) context.

        Each mine keeps its category (top/side group halves) and the anti-tank
        rule (``ap`` never used for AT mines), but every season/daytime combo
        gets an independent random pick from that mine's valid trajectory pool,
        so the same landmine is flown with varied paths across the run.
        """
        contexts = [(s, d) for s in seasons for d in daytimes]
        groups = {}
        for idx, m in enumerate(mines):
            groups.setdefault(m["type"], []).append(idx)

        mapping = {}
        for mine_type, indices in groups.items():
            n = len(indices)
            n_top = (n + 1) // 2
            top_indices = set(indices[:n_top])
            use_ap = not self._mine_is_anti_tank_type(mine_type)
            for idx in indices:
                category = "top" if idx in top_indices else "side"
                ap = use_ap
                for season, daytime in contexts:
                    mapping[(idx, season, daytime)] = self._pick_trajectory_for(category, ap)
                assigned = {mapping[(idx, s, d)]["name"] for s, d in contexts}
                if mine_type:
                    carb.log_info(
                        f"AutoRecord: mine {idx} ({mine_type}) -> "
                        f"{len(assigned)}/{len(contexts)} distinct trajectory(ies) "
                        f"across contexts [{category}{'/ap' if ap else ''}]"
                    )
        return mapping

    def _discover_mines(self):
        return [m["pos"] for m in self._discover_mine_prims()]

    def _nearest_mine(self, pos):
        positions = self._discover_mines()
        if not positions:
            return None
        return min(positions, key=lambda p: self._seg_len(p, pos))

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(value):
        if not value:
            return []
        return [v.strip().lower() for v in str(value).split(",") if v.strip()]

    @staticmethod
    def _build_plan(seasons, daytimes, n_mines):
        plan = []
        for season in seasons:
            plan.append(("season", season))
            for daytime in daytimes:
                plan.append(("daytime", daytime))
                for idx in range(n_mines):
                    plan.append(("mine", idx))
        return plan

    @staticmethod
    def _seg_len(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    @staticmethod
    def _slerp(qa, qb, t):
        qa = list(qa)
        qb = list(qb)
        dot = sum(qa[i] * qb[i] for i in range(4))
        if dot < 0.0:
            qb = [-v for v in qb]
            dot = -dot
        if dot > 0.9995:
            result = [qa[i] + t * (qb[i] - qa[i]) for i in range(4)]
        else:
            theta = math.acos(max(-1.0, min(1.0, dot)))
            sin_theta = math.sin(theta)
            wa = math.sin((1.0 - t) * theta) / sin_theta
            wb = math.sin(t * theta) / sin_theta
            result = [wa * qa[i] + wb * qb[i] for i in range(4)]
        norm = math.sqrt(sum(v * v for v in result))
        if norm > 0.0:
            return [v / norm for v in result]
        return list(qa)

    def _sample_timed(self, wps, elapsed):
        if elapsed <= 0.0 or len(wps) == 1:
            return list(wps[0]["pos"]), list(wps[0]["orient"])
        last_time = wps[-1]["time"]
        if elapsed >= last_time:
            return list(wps[-1]["pos"]), list(wps[-1]["orient"])
        for i in range(len(wps) - 1):
            t0 = wps[i]["time"]
            t1 = wps[i + 1]["time"]
            if elapsed <= t1 or i == len(wps) - 2:
                span = t1 - t0
                if span <= 0.0:
                    return list(wps[i + 1]["pos"]), list(wps[i + 1]["orient"])
                f = (elapsed - t0) / span
                f = max(0.0, min(1.0, f))
                pos = [wps[i]["pos"][k] + f * (wps[i + 1]["pos"][k] - wps[i]["pos"][k]) for k in range(3)]
                orient = self._slerp(wps[i]["orient"], wps[i + 1]["orient"], f)
                return pos, orient
        return list(wps[-1]["pos"]), list(wps[-1]["orient"])

    def _set_status(self, settings, text):
        settings.set(_AUTO_STATUS, text)
        if text != self._last_status:
            self._last_status = text
            carb.log_info(f"AutoRecord: {text}")

    def _set_waiting_status(self, settings):
        now = time.time()
        if now - self._status_flush < 1.0:
            return
        self._status_flush = now
        self._set_status(settings, "Waiting for simulation to start...")
