__all__ = [
    "EnvironmentManager",
]

import os
import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf


class EnvironmentManager:

    SUMMER_PRIMS = ['/World/PaintTool/cube_03fc9', '/World/PaintTool/Cube_013_50a8c', '/World/PaintTool/Cube_013_39087', '/World/PaintTool/Cube_0_6_29800', '/World/PaintTool/Cube_1_3_cbf20',
                              '/World/PaintTool/Cube_0_3_c21e6', '/World/PaintTool/Cube_1_5_0322e', '/World/PaintTool/Wild_Grass_1_fa160', '/World/layout/grass_new_random', 
                              '/World/layout/grass_new_3']
    
    AUTUMN_PRIMS = ['/World/PaintTool/cluster_2_651cb', '/World/PaintTool/cluster_1_bfe10', '/World/PaintTool/oak_2_ff3db', '/World/PaintTool/oak_1_9bb40', '/World/PaintTool/maple_10d95', 
                    '/World/layout/maple', '/World/layout/oak_1', '/World/layout/oak_2', '/World/layout/cluster_1', '/World/layout/cluster_2', "/World/PaintTool/Grass__lawn__procedural_b0053"]

    SPRING_PRIMS = ['/World/PaintTool/Cube_013_50a8c_01', '/World/PaintTool/Cube_0_6_29801', 
                    '/World/PaintTool/Cube_0_3_c21e6_01', '/World/PaintTool/Cube_1_5_0322e_01', '/World/PaintTool/Mesh_001_12e3d']

    WINTER_PRIMS = ['/World/PaintTool/Plane_006_ce9af']

    SEASON_SUFFIXES = {
        "summer": "",
        "winter": "_winter",
        "autumn": "_fall",
        "spring": "_spring",
    }

    SEASONS = {
        "summer": {
            "main_ground_texture": '/World/layout/Looks/Material_05',
            "additional_ground_texture": '/World/layout/Looks/Grass_Countryside_01',
            "trees": ["Big_Pine", "Yellow_Pine", "Shumard_Oak", "Lombardy_Poplar", "Red_Ash", "Fraxinus", "Largetooth_Aspen", "Common_Apple"],
            "ground_pos": [0, 0, 0],
            "mine_offset": [0, 0, 0.02],
            "at_mine_offset": [0, 0, 0],
            "hide_bushes": False,
            "prims_to_show": SUMMER_PRIMS,
            "prims_to_hide": AUTUMN_PRIMS + SPRING_PRIMS + WINTER_PRIMS,
        },
        "winter": {
            "main_ground_texture": "/World/layout/Looks/Material_Snow_02",
            "additional_ground_texture": "/World/layout/Looks/Material_Snow",
            "trees": ["Big_Pine", "Yellow_Pine", "Shumard_Oak", "Lombardy_Poplar", "Red_Ash", "Fraxinus", "Largetooth_Aspen", "Common_Apple"],
            "ground_pos": [0, 0, 0.75],
            "mine_offset": [0, 0, 0.063],
            "at_mine_offset": [0, 0, 0.02],
            "hide_bushes": True,
            "prims_to_show": WINTER_PRIMS,
            "prims_to_hide": SUMMER_PRIMS + AUTUMN_PRIMS + SPRING_PRIMS,
        },
        "autumn": {
            "main_ground_texture": '/World/layout/Looks/Material_05',
            "additional_ground_texture": '/World/layout/Looks/Grass_Countryside_01',
            "trees": ["Big_Pine", "Yellow_Pine", "Shumard_Oak", "Lombardy_Poplar", "Red_Ash", "Fraxinus", "Largetooth_Aspen", "Common_Apple"],
            "ground_pos": [0, 0, 0],
            "mine_offset": [0, 0, 0],
            "at_mine_offset": [0, 0, 0],
            "hide_bushes": True,
            "prims_to_show": AUTUMN_PRIMS, 
            "prims_to_hide": SUMMER_PRIMS + SPRING_PRIMS + WINTER_PRIMS,
        },
        "spring": {
            "main_ground_texture": '/World/layout/Looks/Material_05',
            "additional_ground_texture": '/World/layout/Looks/Grass_Countryside_01',
            "trees": ["Big_Pine", "Yellow_Pine", "Shumard_Oak", "Lombardy_Poplar", "Red_Ash", "Fraxinus", "Largetooth_Aspen", "Common_Apple"],
            "ground_pos": [0, 0, 0.07],
            "mine_offset": [0, 0, 0.04],
            "at_mine_offset": [0, 0, 0.01],
            "hide_bushes": False,
            "prims_to_show": SPRING_PRIMS,
            "prims_to_hide": AUTUMN_PRIMS + SUMMER_PRIMS + WINTER_PRIMS,
        },
    }

    DAYTIMES = {
        "day": "/World/layout/sky",
        "night": "/World/layout/sky_night",
        "morning": "/World/layout/sky_morning",
        "evening": "/World/layout/sky_evening",
    }

    # Anti-personnel landmines to lift with the season ground offset.
    # key: mine type, value: stage name prefix used when spawning mines
    # (the spawner names them f"obstacle_{mine_type.lower()}_{j}").
    AP_MINES = {
        "Mine_PMN": "obstacle_mine_pmn",
        "Mine_PMN2": "obstacle_mine_pmn2",
        "Mine_PMN4": "obstacle_mine_pmn4",
        "Mine_PFM1": "obstacle_mine_pfm1",
        "Mine_PMD6": "obstacle_mine_pmd6",
        "Mine_POM2": "obstacle_mine_pom2",
        "Mine_POMZ2": "obstacle_mine_pomz2",
        "Mine_POM3": "obstacle_mine_pom3",
        "Mine_OZM72": "obstacle_mine_ozm72",
        "Mine_POB_Pilka": "obstacle_mine_pob_pilka",
        "Mine_MON50": "obstacle_mine_mon50",
        "Mine_MON90": "obstacle_mine_mon90",
        "Mine_MON100": "obstacle_mine_mon100",
        "Mine_MON200": "obstacle_mine_mon200",
        "Mine_MS3": "obstacle_mine_ms3",
        "Mine_MS4": "obstacle_mine_ms4",
        "Mine_ML7": "obstacle_mine_ml7",
        "Mine_ML8": "obstacle_mine_ml8",
    }

    # Anti-tank landmines to lift with the season at_mine_offset.
    # key: mine type, value: stage name prefix used when spawning mines
    # (the spawner names them f"obstacle_{mine_type.lower()}_{j}").
    AT_MINES = {
        "Mine_TM46": "obstacle_mine_tm46",
        "Mine_TM57": "obstacle_mine_tm57",
        "Mine_TM62": "obstacle_mine_tm62",
        "Mine_TM62B": "obstacle_mine_tm62b",
        "Mine_TM62D": "obstacle_mine_tm62d",
        "Mine_TM62P": "obstacle_mine_tm62p",
        "Mine_TM62P2": "obstacle_mine_tm62p2",
        "Mine_TM62P3": "obstacle_mine_tm62p3",
        "Mine_TM62T": "obstacle_mine_tm62t",
        "Mine_TM72": "obstacle_mine_tm72",
        "Mine_TM83": "obstacle_mine_tm83",
        "Mine_TM89": "obstacle_mine_tm89",
        "Mine_PTM1": "obstacle_mine_ptm1",
        "Mine_PTM3": "obstacle_mine_ptm3",
        "Mine_PTM4": "obstacle_mine_ptm4",
        "Mine_PTKM1R": "obstacle_mine_ptkm1r",
    }

    # Attribute on each mine prim holding its true spawn position (recorded at
    # load time). The season lift is always applied relative to this origin, so
    # mines return to it when the offset is zero - regardless of how many
    # EnvironmentManager instances touch them.
    MINE_ORIGIN_ATTR = "my_extension:mine_origin"

    def __init__(self):
        self.stage = None
        self.main_ground_prim_path = '/World/layout/ground/Root/Plane_01/Plane_01_001/Plane_01'
        self.ground_prim_paths = ['/World/layout/layout/mesh/mesh', '/World/layout/layout/mesh/mesh_01', '/World/layout/layout/mesh/mesh_02', '/World/layout/layout/mesh/mesh_03']
        self._original_tree_refs = {}
        self._original_ground_translation = None
        self._original_seasonal_translations = {}

    def set_stage(self, stage=None):
        self.stage = stage or omni.usd.get_context().get_stage()

    def apply_season(self, name, replace_trees=False):
        self._ensure_stage()
        season = self.SEASONS.get(name)
        if season is None:
            carb.log_warn(f"Unknown season: {name}")
            return

        main_path = self._resolve_path(season.get("main_ground_texture", ""))
        addl_path = self._resolve_path(season.get("additional_ground_texture", ""))

        ground_offset = season.get("ground_pos", [0, 0, 0])

        if main_path and self.main_ground_prim_path:
            material_prim = self.stage.GetPrimAtPath(main_path)
            if material_prim and material_prim.IsValid():
                material = UsdShade.Material(material_prim)
                prim = self.stage.GetPrimAtPath(self._resolve_path(self.main_ground_prim_path))
                if prim and prim.IsValid():
                    UsdShade.MaterialBindingAPI(prim).Bind(material)

        if addl_path:
            material_prim = self.stage.GetPrimAtPath(addl_path)
            if material_prim and material_prim.IsValid():
                material = UsdShade.Material(material_prim)
                for prim_path in self.ground_prim_paths:
                    prim = self.stage.GetPrimAtPath(self._resolve_path(prim_path))
                    if prim and prim.IsValid():
                        UsdShade.MaterialBindingAPI(prim).Bind(material)

        prim = self.stage.GetPrimAtPath(self._resolve_path(self.main_ground_prim_path))
        if prim and prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    current = op.Get(Usd.TimeCode.Default())
                    if self._original_ground_translation is None:
                        self._original_ground_translation = Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]))
                    offset = Gf.Vec3d(*ground_offset)
                    op.Set(self._original_ground_translation + offset, Usd.TimeCode.Default())
                    break

        self._apply_ground_offset(ground_offset, season.get("prims_to_show", []))
        self._apply_mine_lift(season.get("mine_offset", [0, 0, 0]), self.AP_MINES)
        self._apply_mine_lift(season.get("at_mine_offset", [0, 0, 0]), self.AT_MINES)

        for path in season.get("prims_to_show", []):
            self._set_prim_visible(self._resolve_path(path), True)
        for path in season.get("prims_to_hide", []):
            self._set_prim_visible(self._resolve_path(path), False)

        hide_bushes = season.get("hide_bushes", False)
        trees_prim = self.stage.GetPrimAtPath("/World/layout/trees")
        if trees_prim and trees_prim.IsValid():
            for prim in Usd.PrimRange(trees_prim):
                if prim.IsValid() and "obstacle_bush" in str(prim.GetPath()):
                    self._set_prim_visible(str(prim.GetPath()), not hide_bushes)

        if replace_trees:
            self._swap_trees(season, self.SEASON_SUFFIXES.get(name, ""))

    def apply_daytime(self, name):
        self._ensure_stage()
        sky_path = self.DAYTIMES.get(name)
        if sky_path is None:
            carb.log_warn(f"Unknown daytime: {name}")
            return

        for other_name, other_path in self.DAYTIMES.items():
            resolved = self._resolve_path(other_path)
            if resolved:
                self._set_prim_visible(resolved, False)

        resolved_sky = self._resolve_path(sky_path)
        if resolved_sky:
            self._set_prim_visible(resolved_sky, True)

    def _set_prim_visible(self, path, visible):
        if not path:
            return
        prim = self.stage.GetPrimAtPath(path)
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            if visible:
                UsdGeom.Imageable(prim).MakeVisible()
            else:
                UsdGeom.Imageable(prim).MakeInvisible()

    def _apply_ground_offset(self, offset, prim_paths):
        if offset == [0, 0, 0] or '/World/PaintTool/Plane_006_ce9af' in prim_paths:
            return
        
        for path in prim_paths:
            self._set_prim_translation(self._resolve_path(path), offset)

    def _apply_mine_lift(self, offset, mine_map):
        """Lift landmines matching mine_map by the given offset so they stay on
        top of the lifted terrain, same as the seasonal prims. A zero offset
        restores the mines to their original spawn positions."""
        mines_scope = self.stage.GetPrimAtPath("/World/layout/mines")
        if not mines_scope or not mines_scope.IsValid():
            return
        prefixes = set(mine_map.values())
        for prim in Usd.PrimRange(mines_scope):
            name = prim.GetName()
            if "obstacle_mine" not in name:
                continue
            if not self._is_mine(name, prefixes):
                continue
            self._set_mine_translation(str(prim.GetPath()), offset)

    def _set_mine_translation(self, path, offset):
        """Apply the offset relative to the mine's recorded spawn origin."""
        if not path:
            return
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            return
        origin_attr = prim.CreateAttribute(self.MINE_ORIGIN_ATTR, Sdf.ValueTypeNames.Vector3d)
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        origin = origin_attr.Get()
        if origin is None:
            current = translate_op.Get(Usd.TimeCode.Default())
            if current is None:
                current = (0.0, 0.0, 0.0)
            origin = Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]))
            origin_attr.Set(origin)
        translate_op.Set(origin + Gf.Vec3d(*offset), Usd.TimeCode.Default())

    def lift_mines(self, season_name):
        """Apply the season's mine offsets to the currently loaded landmines
        (used after mines are spawned into the scene)."""
        season = self.SEASONS.get(season_name)
        if season is None:
            carb.log_warn(f"Unknown season: {season_name}")
            return
        self._apply_mine_lift(season.get("mine_offset", [0, 0, 0]), self.AP_MINES)
        self._apply_mine_lift(season.get("at_mine_offset", [0, 0, 0]), self.AT_MINES)

    @staticmethod
    def _is_mine(name, prefixes):
        low = name.lower()
        return any(low.startswith(p) for p in prefixes)

    def _set_prim_translation(self, path, offset):
        if not path:
            return
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            return
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        key = str(path)
        if key not in self._original_seasonal_translations:
            if translate_op is not None:
                current = translate_op.Get(Usd.TimeCode.Default())
                self._original_seasonal_translations[key] = Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]))
            else:
                self._original_seasonal_translations[key] = Gf.Vec3d(0.0, 0.0, 0.0)
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        translate_op.Set(self._original_seasonal_translations[key] + Gf.Vec3d(*offset), Usd.TimeCode.Default())

    def _resolve_path(self, path):
        if path and path.startswith("/World/"):
            return "/World/layout" + path[6:]
        return path

    def _swap_trees(self, season, suffix):
        tree_names = season.get("trees", [])
        if not tree_names:
            return

        trees_prim = self.stage.GetPrimAtPath("/World/layout/trees")
        if not trees_prim or not trees_prim.IsValid():
            return

        for prim in Usd.PrimRange(trees_prim):
            if not prim.IsValid():
                continue
            ref_op = prim.GetMetadata("references")
            if not ref_op:
                continue

            raw = None
            used_attr = None
            for attr in ['prependedItems', 'prepended_items', 'GetPrependedItems',
                         'explicitItems', 'explicit_items', 'appendedItems', 'appended_items',
                         'GetExplicitItems', 'GetAppendedItems']:
                val = getattr(ref_op, attr, None)
                if val is not None:
                    raw = val() if callable(val) else val
                    used_attr = attr
                    break
            if used_attr is None:
                continue

            if isinstance(raw, (list, tuple)):
                refs = list(raw)
            elif raw:
                refs = list(raw)
            else:
                refs = []

            if not refs:
                continue

            for ref in refs:
                asset_path = ref.assetPath
                if not asset_path:
                    continue
                prim_path = str(prim.GetPath())

                if prim_path not in self._original_tree_refs:
                    self._original_tree_refs[prim_path] = asset_path

                if not suffix:
                    orig = self._original_tree_refs.get(prim_path)
                    if orig and orig != asset_path:
                        prim.GetReferences().ClearReferences()
                        prim.GetReferences().AddReference(orig, ref.primPath)
                    continue

                original_asset = self._original_tree_refs[prim_path]
                filename = os.path.basename(original_asset)
                stem, ext = os.path.splitext(filename)

                matched_name = None
                match_idx = -1
                for tn in tree_names:
                    idx = stem.lower().find(tn.lower())
                    if idx != -1:
                        matched_name = tn
                        match_idx = idx
                        break

                if matched_name is None:
                    continue

                obs_idx = stem.find("_obstacle_")
                if obs_idx == -1:
                    continue

                before = stem[:match_idx]
                after_obs = stem[obs_idx:]
                dirname = os.path.dirname(asset_path)

                found = None
                sfx_variants = [suffix]
                if len(suffix) > 1 and suffix[1].islower():
                    sfx_variants.append(suffix[0] + suffix[1].upper() + suffix[2:])
                for sfx in sfx_variants:
                    seasonal_name = matched_name + sfx
                    new_stem = before + seasonal_name + after_obs
                    new_fn = new_stem + ext
                    new_ap = os.path.join(dirname, new_fn)
                    layer = Sdf.Layer.FindOrOpen(new_ap)
                    if layer:
                        found = new_ap
                        break

                if found:
                    prim.GetReferences().ClearReferences()
                    prim.GetReferences().AddReference(found, ref.primPath)

    def _ensure_stage(self):
        if self.stage is None:
            self.set_stage()
