from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from omni.usd import get_stage_next_free_path
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from pxr import Gf, UsdGeom
import time
import carb

class Tag:
    def __init__(
        self,
        stage_prefix: str,
        usd_path: str = None,
        init_pos=[0.0, 0.0, 0.0],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        box_min = (-12.0, 3.0),
        box_max = (12.0, 6.0),
        vx=1,
        vy=1,
        vz=1
    ):
        """
        Class that initializes a vehicle in the isaac sim's curent stage

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_path (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.0].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
        """

        # Get the current world at which we want to spawn the vehicle
        self._world = PegasusInterface().world
        self._current_stage = self._world.stage

        # Save the name with which the vehicle will appear in the stage
        # and the name of the .usd file that contains its description
        self._stage_prefix = get_stage_next_free_path(self._current_stage, stage_prefix, False)
        self._usd_file = usd_path

        # Get the vehicle name by taking the last part of vehicle stage prefix
        self._tag_name = self._stage_prefix.rpartition("/")[-1]

        # Spawn the vehicle primitive in the world's stage
        self._prim = define_prim(self._stage_prefix, "Mesh")
        self._prim = get_prim_at_path(self._stage_prefix)
        self._prim.GetReferences().AddReference(self._usd_file)

        self._xform = UsdGeom.Xformable(self._prim)

        ops = self._xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(init_pos[0], init_pos[1], init_pos[2]))
                self.translate_op = op
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatf(init_orientation[0], init_orientation[1], init_orientation[2], init_orientation[3]))

        self.pos = init_pos
        self.init_pos = init_pos.copy()

        self.box_min = box_min
        self.box_max = box_max
        self.vx, self.vy, self.vz = vx, vy, vz
        self.start_vx, self.start_vy, self.start_vz = vx, vy, vz

        self.vy_sign = 1
        self.vz_sign = 1

        self.vy_check = True
        self.vz_check = True

        self.vy_done = False
        self.drone_pos = None

        self.last_time = None

        self.decrease_coef = 0.98
        self.increase_coef = 1.1

    def dvd_move(self):
        if self.last_time is None:
            self.last_time = time.time()

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # self.pos[0] += 1 * dt
        # self.pos[1] += 1 * dt
        self.pos[0] += self.vx * dt
        self.pos[1] += self.vy * dt
        self.pos[2] += self.vz * dt
        self.vy_done = False

        vy_start_move = self.vy
        vz_start_move = self.vz

        # Bounce check
        if (self.pos[1] <= self.box_min[0] or self.pos[1] >= self.box_max[0]) and self.vy_check is True:
            if abs(vy_start_move) < 0.05:
                self.vy_sign = -self.vy_sign
                self.vy = -self.vy
                self.vy_check = False

            self.vy *= self.decrease_coef
            self.vz *= self.decrease_coef
        elif (self.pos[1] <= self.box_min[0] or self.pos[1] >= self.box_max[0]) and not self.vy_check:
            self.vy *= self.increase_coef
            self.vz *= self.increase_coef

            self.vz = self.vz if abs(self.vz) < abs(self.start_vz) else self.start_vz * self.vz_sign
            self.vy = self.vy if abs(self.vy) < abs(self.start_vy) else self.start_vy * self.vy_sign
        else:
            self.vy_check = True

        if (self.pos[2] <= self.box_min[1] or self.pos[2] >= self.box_max[1]) and self.vz_check is True:
            if abs(vz_start_move) < 0.05:
                self.vz_sign = -self.vz_sign
                self.vz = -self.vz
                self.vz_check = False

            self.vz *= self.decrease_coef
            self.vy *= self.decrease_coef
        elif (self.pos[2] <= self.box_min[1] or self.pos[2] >= self.box_max[1]) and not self.vz_check:
            self.vz *= self.increase_coef
            self.vy *= self.increase_coef

            self.vz = self.vz if abs(self.vz) < abs(self.start_vz) else self.start_vz * self.vz_sign
            self.vy = self.vy if abs(self.vy) < abs(self.start_vy) else self.start_vy * self.vy_sign
        else:
            self.vz_check = True

        self.translate_op.Set(Gf.Vec3d(self.pos[0], self.pos[1], self.pos[2]))
        # self.send_tag_pos((UDP_IP, UDP_TAG_PORT))

    def move(self):
        if self.last_time is None:
            self.last_time = time.time()

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        self.pos[0] += self.vx * dt
        self.pos[1] += self.vy * dt
        self.pos[2] += self.vz * dt

        self.translate_op.Set(Gf.Vec3d(self.pos[0], self.pos[1], self.pos[2]))

    def set_speed(self, vx, vy, vz):
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def reset(self):
        self.translate_op.Set(Gf.Vec3d(self.init_pos[0], self.init_pos[1], self.init_pos[2]))
        self.pos = self.init_pos.copy()
        self.reset_time()

    def reset_time(self):
        self.last_time = None

    def set_pos_and_orient(self, pos, euler_angles):
        self.init_pos = pos
        self.pos = pos
        ops = self._xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                self.translate_op = op
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatf(euler_angles[0], euler_angles[1], euler_angles[2], euler_angles[3]))

    def set_pos(self, pos):
        self.init_pos = pos
        self.pos = pos
        ops = self._xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                self.translate_op = op
