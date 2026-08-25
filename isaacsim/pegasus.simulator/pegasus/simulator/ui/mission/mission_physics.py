from __future__ import annotations
from typing import TYPE_CHECKING
import csv
import time
import carb
import numpy as np
from datetime import datetime
import omni.kit
import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom, Gf
from omni.physx import get_physx_simulation_interface

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper


class MissionPhysics:
    def __init__(self, helper: MissionHelper):
        self.helper = helper
        self._daytime_enabled = False
        self._daytime_rate = 0.02
        self._daytime_frame_counter = 0
        self._daytime_update_frequency = 5
        self._daytime_sky_check_accumulator = 0.0
        self._daytime_sky_check_interval = 1.0
        self._active_shader_path = None
        log_settings = carb.settings.get_settings()
        log_settings.set("/log/channelLevel/omni.hydra", 2)

    def toggle_250hz_loop(self):
        self.helper._custom_loop_running = not self.helper._custom_loop_running
        settings = carb.settings.get_settings()

        settings.set("/physics/useGpu", False)
        settings.set("/physics/deterministic", True)
        settings.set("/physics/positionIterationCount", 16)
        settings.set("/physics/velocityIterationCount", 4)
        settings.set("/persistent/simulation/minFrameRate", 0)
        settings.set("/physics/asyncFastKinetics", False)

        if self.helper._custom_loop_running:
            self.helper._accumulator = 0.0
            settings.set("/ext/pegasus/custom_30hz_active", True)
            if self.helper.timeline.is_playing():
                self.helper.timeline.stop()
            self.helper._sim_time = self.helper.timeline.get_current_time()
            carb.log_warn("--- Started Continuous 30Hz Physics Loop ---")
        else:
            carb.log_warn("--- Stopped Continuous 30Hz Physics Loop ---")
            settings.set("/ext/pegasus/custom_30hz_active", False)

    def start_motor_replay(self):
        if not self.helper.multirotors or not self.helper.multirotors[0]._backends:
            carb.log_warn("Cannot start motor replay: no backend available")
            return

        backend = self.helper.multirotors[0]._backends[0]
        if not hasattr(backend, 'motor_csv_data') or len(backend.motor_csv_data) == 0:
            carb.log_warn("Cannot start motor replay: motor CSV not loaded")
            return

        self.helper._motor_csv_data = backend.motor_csv_data
        self.helper._motor_csv_index = 0
        self.helper._motor_csv_loaded = True
        self.helper._motor_replay_log = []

        self.helper._motor_csv_first_time_ns = self.helper._motor_csv_data[0].get('time_ns', 0)
        self.helper._motor_replay_start_time_ns = time.time_ns()

        self.helper._motor_replay_mode = True
        self.helper._custom_loop_running = True

        self.helper._sim_time = 0.0
        self.helper._accumulator = 0.0

        carb.log_warn(f"Motor replay started. First timestamp: {self.helper._motor_csv_first_time_ns}, Total rows: {len(self.helper._motor_csv_data)}")

    def stop_motor_replay(self):
        self._save_motor_replay_log()
        self.helper._motor_replay_mode = False
        self.helper._motor_csv_loaded = False
        self.helper._motor_csv_index = 0

    def _save_motor_replay_log(self, folder="/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/replay_logs"):
        if not self.helper._motor_replay_log:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/motor_replay_log_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as f:
                if self.helper._motor_replay_log:
                    fieldnames = list(self.helper._motor_replay_log[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.helper._motor_replay_log)
            carb.log_warn(f"Saved motor replay log with {len(self.helper._motor_replay_log)} rows to {filename}")
        except Exception as e:
            carb.log_error(f"Failed to save motor replay log: {e}")

    def _find_active_shader_path(self):
        stage = self.helper.stage
        if not stage:
            return None
        sky_names = ["sky", "sky_night", "sky_morning", "sky_evening"]
        for name in sky_names:
            prim = stage.GetPrimAtPath(f"/World/layout/layout/{name}")
            # INFO: WARNINGS ARE DISABLED
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                if UsdGeom.Imageable(prim).GetVisibilityAttr().Get() != "invisible":
                    shader_path = f"/World/layout/layout/{name}/sky/Looks/SkyMaterial/Shader"
                    if stage.GetPrimAtPath(shader_path).IsValid():
                        return shader_path
        return None

    def _increment_time_of_day(self, frame_dt):
        if not self._active_shader_path:
            return
        shader = self.helper.stage.GetPrimAtPath(self._active_shader_path)
        attr = shader.GetAttribute("inputs:TimeOfDay")
        if not attr or not attr.IsValid():
            attr = shader.GetAttribute("TimeOfDay")
        if attr and attr.IsValid():
            current = attr.Get() or 0.0
            attr.Set(current + self._daytime_rate * frame_dt)

    def _on_app_update(self, e: carb.events.IEvent):
        helper = self.helper

        if not helper._custom_loop_running:
            return

        if helper.stage is None:
            helper.stage = omni.usd.get_context().get_stage()

        if helper.body_prims[0] is None:
            helper.body_prims[0] = helper.stage.GetPrimAtPath(f"{helper.multirotors[0].prim_path}/body")

        if helper.tuner is not None and helper.tuner.is_running():
            helper.tuner.tick(e.payload.get("dt", 0.0) if hasattr(e, "payload") else 0.0)

        backend = helper.multirotors[0]._backends[0]
        physx_sim = get_physx_simulation_interface()

        if helper._motor_replay_requested:
            helper._motor_replay_requested = False
            self.start_motor_replay()

        frame_dt = e.payload["dt"]

        if self._daytime_enabled:
            self._daytime_sky_check_accumulator += frame_dt
            if self._daytime_sky_check_accumulator >= self._daytime_sky_check_interval:
                self._daytime_sky_check_accumulator = 0.0
                self._active_shader_path = self._find_active_shader_path()

            self._daytime_frame_counter += 1
            if self._daytime_frame_counter >= self._daytime_update_frequency:
                self._daytime_frame_counter = 0
                self._increment_time_of_day(frame_dt * self._daytime_update_frequency)

        if helper._motor_replay_mode and helper._motor_csv_loaded:
            backend.update_disabled = True
            helper._physx_interface.force_load_physics_from_usd()
            helper._accumulator += frame_dt
            if helper._motor_csv_index >= len(helper._motor_csv_data) - 4:
                carb.log_warn("Motor CSV replay finished. Switching to normal mode.")
                self._save_motor_replay_log()
                helper._motor_replay_mode = False
                helper._motor_csv_index = 0
                helper._motor_csv_loaded = False
                helper._motor_started = False
                helper._motor_replay_log = []
                return
            while helper._accumulator >= helper._fixed_dt:
                start_time = time.time()
                helper.move_cycle()
                helper.mission_cycle()
                helper.check_mission_end()

                row = helper._motor_csv_data[helper._motor_csv_index]
                if helper._motor_csv_index + 1 < len(helper._motor_csv_data):
                    next_time_ns = helper._motor_csv_data[helper._motor_csv_index + 1]['time_ns']
                    current_time_ns = row.get('time_ns', 0)
                    if current_time_ns == 0:
                        current_time_ns = row.get('timestamp', 0)
                    dt = (next_time_ns - current_time_ns) / 1e9
                else:
                    dt = helper._fixed_dt

                motors = [row.get('motor0', 0), row.get('motor1', 0),
                          row.get('motor2', 0), row.get('motor3', 0)]

                helper._sim_time = row.get("sim_time", 0)

                backend.current_controls = motors

                if helper.use_physx_sim:
                    physx_sim.simulate(helper._fixed_dt, helper._sim_time)
                else:
                    helper._physx_interface.update_simulation(helper._fixed_dt, helper._sim_time)

                xform = UsdGeom.Xformable(helper.body_prims[0]).ComputeLocalToWorldTransform(0.0)
                position = xform.ExtractTranslation()
                current_pos_vec = np.array(list(position))

                helper._motor_replay_log.append({
                    'index': helper._motor_csv_index,
                    'timestamp': time.time(),
                    'motor0': row.get('motor0', 0),
                    'motor1': row.get('motor1', 0),
                    'motor2': row.get('motor2', 0),
                    'motor3': row.get('motor3', 0),
                    'pos_x': current_pos_vec[0],
                    'pos_y': current_pos_vec[1],
                    'pos_z': current_pos_vec[2],
                    'sim_time': helper._sim_time,
                })

                helper._motor_csv_index += 1
                helper._accumulator -= helper._fixed_dt

            if helper.use_physx_sim:
                physx_sim.fetch_results()
            else:
                helper._physx_interface.update_transformations(False, True)

            helper.timeline.set_current_time(helper._sim_time)
            return

        frame_dt = e.payload["dt"]
        helper._accumulator += frame_dt
        backend.update_disabled = False

        while helper._accumulator >= helper._fixed_dt:
            helper.move_cycle()
            helper.mission_cycle()
            helper.check_mission_end()

            if helper.use_physx_sim:
                physx_sim.simulate(helper._fixed_dt, helper._sim_time)
            else:
                helper._physx_interface.update_simulation(helper._fixed_dt, helper._sim_time)

            controls = backend.current_controls

            if any(c > 0 for c in controls):
                if not helper.motor_record_armed:
                    helper._sim_time = 0.0
                xform = UsdGeom.Xformable(helper.body_prims[0]).ComputeLocalToWorldTransform(0.0)

                position = xform.ExtractTranslation()
                current_pos_vec = np.array(list(position))

                helper.motor_record_armed = True

                if helper.motor_record_armed:
                    helper._motor_replay_log.append({
                        'index': helper.motor_record_index,
                        'timestamp': time.time(),
                        'motor0': controls[0],
                        'motor1': controls[1],
                        'motor2': controls[2],
                        'motor3': controls[3],
                        'pos_x': current_pos_vec[0],
                        'pos_y': current_pos_vec[1],
                        'pos_z': current_pos_vec[2],
                        'sim_time': helper._sim_time,
                    })

                    helper.motor_record_index += 1
            else:
                helper.motor_record_armed = False

            helper._sim_time += helper._fixed_dt
            helper._accumulator -= helper._fixed_dt

        if helper.use_physx_sim:
            physx_sim.fetch_results()
        else:
            helper._physx_interface.update_transformations(False, True)

        helper.timeline.set_current_time(helper._sim_time)

        if not helper.motor_record_armed and len(helper._motor_replay_log) != 0:
            helper.motor_record_index = 0
            self._save_motor_replay_log("/home/geekscode/ros2_ws/src/my_drone_pkg/my_drone_pkg/original_logs")
            helper._motor_replay_log = []
