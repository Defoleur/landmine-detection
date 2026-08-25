from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import carb
import omni.usd
from pxr import UsdGeom, Gf

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper


class MissionTagsVehicles:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def on_reset_tag(self):
        if self.helper._tag is not None:
            self.helper._tag.reset()
            self.helper.move_cycle_state = "STOPPED"

    def on_start_tag(self):
        if self.helper._tag is not None and self.helper._window is not None:
            velocities = self.helper._window.get_tag_velocities()
            self.helper._tag.set_speed(velocities[0], velocities[1], velocities[2])
            self.helper.move_cycle_state = "STARTED"

    def on_pause_tag(self):
        if self.helper._tag is not None:
            self.helper.move_cycle_state = "STOPPED"
            self.helper._tag.reset_time()

    def move_cycle(self):
        if self.helper.move_cycle_state == "STOPPED" or self.helper._tag is None:
            return

        if self.helper._dvd_enabled_checkbox and self.helper._dvd_enabled_checkbox.get_value_as_bool():
            self.helper._tag.dvd_move()
        else:
            self.helper._tag.move()

    def generate_formation(self, current_formation, mapper_enabled=True):
        if not self.helper.ultimate_mission_check:
            return [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]

        current_formation_settings = self.helper.current_ultimate_mission["formation"]
        formation_type = current_formation_settings["formation_type"]
        formation_distance = current_formation_settings["formation_distance"]
        first, second, third = current_formation, current_formation, current_formation
        match formation_type:
            case "triangle":
                second = [-formation_distance, formation_distance, 0.0]
                third = [-formation_distance, -formation_distance, 0.0]
            case "key":
                second = [-formation_distance, formation_distance, 0.0]
                third = [formation_distance, -formation_distance, 0.0]
            case "row":
                second = [0.0, formation_distance, 0.0]
                third = [0.0, -formation_distance, 0.0]

        if mapper_enabled:
            generated_formation = [
                [current_formation[i] + axis + self.helper.default_formation_mapper[0][i] for i, axis in enumerate(first)],
                [current_formation[i] + axis + self.helper.default_formation_mapper[1][i] for i, axis in enumerate(second)],
                [current_formation[i] + axis + self.helper.default_formation_mapper[2][i] for i, axis in enumerate(third)],
            ]
        else:
            generated_formation = [
                [axis for i, axis in enumerate(first)],
                [current_formation[i] + axis for i, axis in enumerate(second)],
                [current_formation[i] + axis for i, axis in enumerate(third)],
            ]
        return generated_formation

    def on_reset_tags(self):
        generated_formation = self.generate_formation(self.helper.current_tag_formation, mapper_enabled=False)
        for i, tag in enumerate(self.helper.tags):
            tag.set_pos(generated_formation[i])

    def on_reset_imu(self):
        custom_imu_enabled = self.helper._custom_imu_enabled_checkbox.get_value_as_bool()

        for multirotor in self.helper.multirotors:
            multirotor._sensors[1]._gyroscope_bias = np.zeros((3,))
            multirotor._sensors[1]._accelerometer_bias = np.zeros((3,))

    def on_reset_vehicles(self):
        generated_formation = self.generate_formation(self.helper.current_drone_formation)
        for i, vehicle in enumerate(self.helper.multirotors):
            self.on_reset_vehicle(vehicle, generated_formation[i])

    def on_reset_vehicle(self, multirotor, formation):
        stage = omni.usd.get_context().get_stage()
        body_prim = stage.GetPrimAtPath(f"{multirotor.prim_path}/body")

        if not body_prim.IsValid():
            carb.log_warn("Quadrotor body not found in stage — cannot reset it.")
            return

        xform = UsdGeom.Xformable(body_prim)
        ops = xform.GetOrderedXformOps()

        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                if self.helper.ultimate_mission_check:
                    pos = formation
                    op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                else:
                    op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatf(1.0, 0.002, -3.366, 0.04))
