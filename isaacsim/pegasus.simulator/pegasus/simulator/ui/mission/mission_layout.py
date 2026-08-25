from __future__ import annotations
from typing import TYPE_CHECKING
import carb
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdShade, Sdf, UsdPhysics

if TYPE_CHECKING:
    from ..mission_helper import MissionHelper


class MissionLayout:
    def __init__(self, helper: MissionHelper):
        self.helper = helper

    def spawn_lightweight_layout_copy(
        self,
        original_path="/World/layout",
        copy_path="/World/layout_mission",
        z_offset=-5000.0,
        custom_texture_path=""
    ):
        stage = omni.usd.get_context().get_stage()

        original_prim = stage.GetPrimAtPath(original_path)
        if not original_prim.IsValid():
            carb.log_warn(f"Original layout not found at {original_path}")
            return

        copy_xform = UsdGeom.Xform.Define(stage, copy_path)
        copy_prim = copy_xform.GetPrim()
        copy_prim.GetReferences().AddInternalReference(original_path)

        if z_offset != 0.0:
            copy_xform.ClearXformOpOrder()
            t_op = copy_xform.AddTranslateOp()
            t_op.Set(Gf.Vec3d(0.0, 0.0, z_offset))

        for prim in Usd.PrimRange(copy_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI(prim)
                collision_api.CreateCollisionEnabledAttr(False)

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(prim)
                rb_api.CreateRigidBodyEnabledAttr(False)

        orig_sdf_path = Sdf.Path(original_path)
        copy_sdf_path = Sdf.Path(copy_path)

        light_types = ["SphereLight", "RectLight", "DiskLight", "CylinderLight", "DistantLight", "DomeLight"]

        for prim in Usd.PrimRange(original_prim):
            prim_name = prim.GetName().lower()

            is_sky = (prim_name == "sky")

            if is_sky:
                copy_target_path = prim.GetPath().ReplacePrefix(orig_sdf_path, copy_sdf_path)

                target_override = stage.OverridePrim(copy_target_path)

                if target_override.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(target_override).MakeInvisible()

                carb.log_info(f"Successfully targeted and disabled Sky at: {copy_target_path}")

        box_path = f"{copy_path}_Enclosure"
        cube = UsdGeom.Cube.Define(stage, box_path)

        box_size = 900.0
        UsdGeom.XformCommonAPI(cube).SetScale((box_size, box_size * 5, box_size))
        UsdGeom.XformCommonAPI(cube).SetTranslate((0.0, 0.0, z_offset))

        cube.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

        existing_material_path = "/World/layout/layout/Looks/Material_05"

        if existing_material_path:
            material_prim = stage.GetPrimAtPath(existing_material_path)

            if material_prim.IsValid():
                material = UsdShade.Material(material_prim)
                UsdShade.MaterialBindingAPI(cube).Bind(material)
                carb.log_warn(f"Material was found at path: {existing_material_path}")
            else:
                carb.log_warn(f"Could not find existing material at path: {existing_material_path}")
        else:
            carb.log_warn(f"Material doesn't exist at path: {existing_material_path}")

        carb.log_warn(f"Spawned mission layout at {copy_path}. Enclosure box and independent light initialized.")

    def delete_lightweight_layout_copy(self, copy_path="/World/layout_mission"):
        stage = omni.usd.get_context().get_stage()

        copy_prim = stage.GetPrimAtPath(copy_path)
        if copy_prim.IsValid():
            stage.RemovePrim(copy_path)
            carb.log_warn(f"Deleted layout copy at {copy_path}")
        else:
            carb.log_warn(f"Could not find layout copy to delete at {copy_path}")

        box_path = f"{copy_path}_Enclosure"
        box_prim = stage.GetPrimAtPath(box_path)
        if box_prim.IsValid():
            stage.RemovePrim(box_path)
            carb.log_warn(f"Deleted enclosure box at {box_path}")
