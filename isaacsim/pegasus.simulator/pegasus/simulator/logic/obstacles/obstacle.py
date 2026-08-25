from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from omni.usd import get_stage_next_free_path
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from pxr import Gf, UsdGeom
import time
import carb
import math

class Obstacle:
    def __init__(
        self,
        stage_prefix: str,
        usd_path: str = None,
        init_pos=[0.0, 0.0, 0.0],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        init_scale=[1.0, 1.0, 1.0],
    ):

        self._world = PegasusInterface().world
        self._current_stage = self._world.stage

        self._stage_prefix = get_stage_next_free_path(self._current_stage, stage_prefix, False)
        self._usd_file = usd_path

        self._tag_name = self._stage_prefix.rpartition("/")[-1]

        self._prim = define_prim(self._stage_prefix, "Xform")
        self._prim = get_prim_at_path(self._stage_prefix)
        
        # ---> THE FIX: DECLARE INSTANCE BEFORE ADDING REFERENCE <---
        # This tells USD: "Do not unpack the geometry. Just point to the master file."
        self._prim.SetInstanceable(True)

        # Now, when it loads, it uses ~0 MB of additional RAM
        self._prim.GetReferences().AddReference(self._usd_file)
        
        self._xform = UsdGeom.Xformable(self._prim)

        # 2. Clear existing ops to ensure a clean slate
        self._xform.ClearXformOpOrder()

        # 3. Explicitly ADD the ops in the correct order: Translate -> Rotate -> Scale
        self.translate_op = self._xform.AddTranslateOp()
        self.translate_op.Set(Gf.Vec3d(init_pos[0], init_pos[1], init_pos[2]))

        self.orient_op = self._xform.AddOrientOp()
        self.orient_op.Set(Gf.Quatf(init_orientation[0], init_orientation[1], init_orientation[2], init_orientation[3]))

        ops = self._xform.GetOrderedXformOps()

        self.scale_op = self._xform.AddXformOp(UsdGeom.XformOp.TypeScale, 
                                                isInverseOp=False, 
                                                opSuffix="scale")
                                                
        self.scale_op.Set(Gf.Vec3f(init_scale[0], init_scale[1], init_scale[2]))

        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(init_pos[0], init_pos[1], init_pos[2]))
                self.translate_op = op
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatf(init_orientation[0], init_orientation[1], init_orientation[2], init_orientation[3]))

        self.pos = init_pos
        self.init_pos = init_pos.copy()

    def reset(self):
        self.translate_op.Set(Gf.Vec3d(self.init_pos[0], self.init_pos[1], self.init_pos[2]))
        self.pos = self.init_pos.copy()
        self.reset_time()

    def reset_time(self):
        self.last_time = None

    def cache_wind_variables(self):
        """Pre-calculates static wind variables to save FPS."""
        # Cache the phase
        self._wind_phase = (self.pos[0] * 0.05) + (self.pos[1] * 0.05)
        
        # Cache the base rotation object (so we don't instantiate it every frame)
        self._base_rot_obj = Gf.Rotation(self.orient_op.Get())
        
        # Cache vectors to avoid creating them every frame
        self._vec_x = Gf.Vec3d(1, 0, 0)
        self._vec_y = Gf.Vec3d(0, 1, 0)

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
                
        # Recalculate cache when the tree physically moves
        self.cache_wind_variables()

    def apply_wind(self, current_time: float, wind_strength: float = 2.0, wind_speed: float = 1.5):
        # Ensure variables are cached
        if not hasattr(self, '_wind_phase'):
            self.cache_wind_variables()

        # Pure math (Very fast)
        sway_pitch = math.sin((current_time * wind_speed) + self._wind_phase) * wind_strength
        sway_roll = math.cos((current_time * wind_speed * 1.3) + self._wind_phase) * (wind_strength * 0.4)

        # Create rotations using cached vectors
        rot_pitch = Gf.Rotation(self._vec_x, sway_pitch)
        rot_roll = Gf.Rotation(self._vec_y, sway_roll)
        
        # Multiply by the cached base rotation
        final_rot = (rot_pitch * rot_roll) * self._base_rot_obj

        # Single API call
        self.orient_op.Set(Gf.Quatf(final_rot.GetQuat()))

    def set_pos(self, pos):
        self.init_pos = pos
        self.pos = pos

        if hasattr(self, '_base_quat'):
            delattr(self, '_base_quat')

        ops = self._xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                self.translate_op = op

    def destroy(self):
        try:
            if self._prim and self._prim.IsValid():
                prim_path = self._prim.GetPath()
                self._current_stage.RemovePrim(prim_path)
                carb.log_info(f"Destroyed obstacle at {prim_path}")
                self._prim = None
            else:
                carb.log_warn("Obstacle prim already invalid or removed.")
        except Exception as e:
            carb.log_error(f"Failed to destroy obstacle: {e}")
