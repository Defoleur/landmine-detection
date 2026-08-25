import carb
from pxr import Usd, Gf, UsdGeom, UsdShade, Sdf
from omni.isaac.core.utils.prims import define_prim

class SceneUtils:
    """
    A helper class for common USD scene operations like creating prims
    and querying information.
    """
    
    def __init__(self):
        self.stage = None

    def set_stage(self, stage):
        """Sets the USD stage for the utility class."""
        self.stage = stage

    def has_stage(self) -> bool:
        """Checks if the stage has been set."""
        return self.stage is not None

    def get_prim_size(self, prim_path: str):
        """
        Gets the computed world-space bounding box size of any prim.
        (This is the improved version of your get_size method).
        """
        if not self.has_stage():
            carb.log_error("SceneUtils: Stage not set. Cannot get prim size.")
            return None

        prim = self.stage.GetPrimAtPath(prim_path)

        if prim.IsValid():
            # Use UsdGeom.Boundable for a more robust method
            # that works on meshes, shapes, etc.
            boundable = UsdGeom.Boundable(prim)
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            )
            # ComputeWorldBound works on Boundable prims
            bbox = bbox_cache.ComputeWorldBound(boundable.GetPrim())
            size = bbox.GetRange().GetSize()
            
            print(f"Prim size at {prim_path} (X, Y, Z): {size}")
            return size
        else:
            print(f"Prim not found at path: {prim_path}")
            return None

    def create_cylinder(self, name, position, radius=50.0, height=100.0, color=(1.0, 0.0, 0.0)):
        """
        Create a cylinder in the scene for visualization.
        """
        if not self.has_stage():
            carb.log_error("SceneUtils: Stage not set. Cannot create cylinder.")
            return None
            
        prim = define_prim(name, "Cylinder")
        cylinder = UsdGeom.Cylinder(prim)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(height)

        x, y, z = position
        UsdGeom.XformCommonAPI(cylinder).SetTranslate(Gf.Vec3d(x, y, z))

        display_color = cylinder.CreateDisplayColorAttr()
        display_color.Set([Gf.Vec3f(*color)])
        return prim

    def create_preview_material(self, mat_path, color=(1.0, 0.0, 0.0), opacity=0.5):
        """
        Create a simple UsdPreviewSurface material.
        """
        if not self.has_stage():
            carb.log_error("SceneUtils: Stage not set. Cannot create material.")
            return None
            
        mat = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/PreviewShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        mat.CreateSurfaceOutput().ConnectToSource(shader, "surface")
        return mat