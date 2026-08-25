"""
| File: params.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: File that defines the base configurations for the Pegasus Simulator.
"""
import os
from pathlib import Path

import isaacsim.storage.native as nucleus

# Extension configuration
EXTENSION_NAME = "Pegasus Simulator"
WINDOW_TITLE = "Pegasus Simulator"
MENU_PATH = "Window/" + WINDOW_TITLE
DOC_LINK = "https://docs.omniverse.nvidia.com"
EXTENSION_OVERVIEW = "This extension shows how to incorporate drones into Isaac Sim"

# Get the current directory of where this extension is located
EXTENSION_FOLDER_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT = str(EXTENSION_FOLDER_PATH.parent.parent.parent.resolve())

# Get the configurations file path
CONFIG_FILE = ROOT + "/pegasus.simulator/config/configs.yaml"

# Define the Extension Assets Path
ASSET_PATH = ROOT + "/pegasus.simulator/pegasus/simulator/assets"
ROBOTS_ASSETS = ASSET_PATH + "/Robots"

# Define the built in robots of the extension
ROBOTS = {"Iris": ROBOTS_ASSETS + "/Iris/iris.usd"} #, "Flying Cube": ROBOTS_ASSETS + "/iris_cube.usda"}

MISSIONS = ["Static Obstacles", "Static", "Static Random", "Movable",]

TAGS_ASSETS = ASSET_PATH + "/Apriltags"

TAGS = {"01" : TAGS_ASSETS + "/01/tag_01.usda"}

OBSTACLES_ASSETS = ASSET_PATH + "/Obstacles"

OBSTACLES = {
    # "Tree" : OBSTACLES_ASSETS + "/Trees/tree.usda", 
    "Tree_4K_New" : OBSTACLES_ASSETS + "/Trees_main/tree_new_4k_2.usd", 
    "Tree_4K_New_2" : OBSTACLES_ASSETS + "/Trees_main/tree_new_4k_4.usd", 
    "Tree_Oak_Scarlet" : OBSTACLES_ASSETS + "/Trees_main/Scarlet_Oak.usd", 
    "Tree_Oak_Shumard" : OBSTACLES_ASSETS + "/Trees_main/Shumard_Oak.usd", 
    "Tree_Big_Pine" : OBSTACLES_ASSETS + "/Trees_main/Big_Pine_updated.usd", 
    "Tree_Common_Apple" : OBSTACLES_ASSETS + "/Trees_main/Common_Apple.usd",
    "Tree_Fraxinus" : OBSTACLES_ASSETS + "/Trees_main/Fraxinus.usd",
    "Tree_Largetooth_Aspen" : OBSTACLES_ASSETS + "/Trees_main/Largetooth_Aspen.usd",
    "Tree_Lombardy_Poplar" : OBSTACLES_ASSETS + "/Trees_main/Lombardy_Poplar.usd",
    "Tree_Red_Ash" : OBSTACLES_ASSETS + "/Trees_main/Red_Ash.usd",
    "Tree_Yellow_Pine" : OBSTACLES_ASSETS + "/Trees_main/Yellow_Pine.usd",
    "Tree_Black_Oak_Fall" : OBSTACLES_ASSETS + "/Trees_main/Black_Oak_Fall.usd",
    "Tree_Gray_Birch_fall" : OBSTACLES_ASSETS + "/Trees_main/Gray_Birch_fall.usd",
    "Tree_Lombardy_Poplar_Fall" : OBSTACLES_ASSETS + "/Trees_main/Lombardy_Poplar_Fall.usd",
    "Tree_Scarlet_Oak_fall" : OBSTACLES_ASSETS + "/Trees_main/Scarlet_Oak_fall.usd",
    "Tree_Shumard_Oak_Fall" : OBSTACLES_ASSETS + "/Trees_main/Shumard_Oak_Fall.usd",
    "Tree_Red_Ash_fall" : OBSTACLES_ASSETS + "/Trees_main/Red_Ash_fall.usd",
    "Tree_Fraxinus_fall" : OBSTACLES_ASSETS + "/Trees_main/Fraxinus_fall.usd",
    "Tree_Largetooth_Aspen_fall" : OBSTACLES_ASSETS + "/Trees_main/Largetooth_Aspen_fall.usd",
    "Tree_Common_Apple_fall" : OBSTACLES_ASSETS + "/Trees_main/Common_Apple_fall.usd",
    "Tree_Sycamore" : OBSTACLES_ASSETS + "/Trees_main/Sycamore.usd",
    "Tree_Fraxinus_winter" : OBSTACLES_ASSETS + "/Trees_main/Fraxinus_winter.usd",
    "Tree_Largetooth_Aspen_winter" : OBSTACLES_ASSETS + "/Trees_main/Largetooth_Aspen_winter.usd",
    "Tree_Lombardy_Poplar_winter" : OBSTACLES_ASSETS + "/Trees_main/Lombardy_Poplar_winter.usd",
    "Tree_Red_Ash_winter" : OBSTACLES_ASSETS + "/Trees_main/Red_Ash_winter.usd",
    "Tree_Scarlet_Oak_bbc77_winter" : OBSTACLES_ASSETS + "/Trees_main/Scarlet_Oak_winter.usd",
    "Tree_Shumard_Oak_a354d_winter" : OBSTACLES_ASSETS + "/Trees_main/Shumard_Oak_winter.usd",
    "Tree_Sycamore_winter" : OBSTACLES_ASSETS + "/Trees_main/Sycamore_winter.usd",
    "Tree_Common_Apple_winter" : OBSTACLES_ASSETS + "/Trees_main/Common_Apple_winter.usd",
    "Tree_Fraxinus_spring" : OBSTACLES_ASSETS + "/Trees_main/Fraxinus_spring.usd",
    "Tree_Largetooth_Aspen_spring" : OBSTACLES_ASSETS + "/Trees_main/Largetooth_Aspen_spring.usd",
    "Tree_Lombardy_Poplar_spring" : OBSTACLES_ASSETS + "/Trees_main/Lombardy_Poplar_spring.usd",
    "Tree_Red_Ash_spring" : OBSTACLES_ASSETS + "/Trees_main/Red_Ash_spring.usd",
    "Tree_Shumard_Oak_spring" : OBSTACLES_ASSETS + "/Trees_main/Shumard_Oak_spring.usd",
    "Tree_Common_Apple_spring" : OBSTACLES_ASSETS + "/Trees_main/Common_Apple_spring.usd",
    "Bush_Acacia": OBSTACLES_ASSETS + "/Bushes/Acacia_42901_updated.usd",
    "Bush_Fountain_Grass": OBSTACLES_ASSETS + "/Bushes/Fountain_Grass_Short_9331d_updated.usd",
    "Bush_Pampas_Grass": OBSTACLES_ASSETS + "/Bushes/Pampas_Grass_200d4_updated.usd",
    "Bush_Holly": OBSTACLES_ASSETS + "/Bushes/holly.usd",
    "Bush_Privet": OBSTACLES_ASSETS + "/Bushes/privet.usd",
    # "Bush_Hydrangea_Without_Flowers": OBSTACLES_ASSETS + "/Bushes/hydrangea_without_flowers.usd",
    # "Bush_Rhododendron_Without_Flowers": OBSTACLES_ASSETS + "/Bushes/rhododendron_without_flowers.usd",
    "Mine_TM46" : OBSTACLES_ASSETS + "/Mines/mine_tm_46.usd",
    "Mine_TM57" : OBSTACLES_ASSETS + "/Mines/mine_tm_57.usd",
    "Mine_TM62" : OBSTACLES_ASSETS + "/Mines/mine.usd",
    "Mine_TM62B" : OBSTACLES_ASSETS + "/Mines/mine_tm_62b.usd",
    "Mine_TM62D" : OBSTACLES_ASSETS + "/Mines/mine_tm_62d.usd",
    "Mine_TM62P" : OBSTACLES_ASSETS + "/Mines/mine_tm_62p.usd",
    "Mine_TM62P2" : OBSTACLES_ASSETS + "/Mines/mine_tm_62p2.usd",
    "Mine_TM62P3" : OBSTACLES_ASSETS + "/Mines/mine_tm_62p3.usd",
    "Mine_TM62T" : OBSTACLES_ASSETS + "/Mines/mine_tm_62t.usd",
    "Mine_TM72" : OBSTACLES_ASSETS + "/Mines/mine_tm72_trellis_main_last.usd",
    "Mine_TM83" : OBSTACLES_ASSETS + "/Mines/mine_tm_83.usd",
    "Mine_TM89" : OBSTACLES_ASSETS + "/Mines/mine_tm_89_main.usd",
    "Mine_PTM1" : OBSTACLES_ASSETS + "/Mines/mine_ptm_1_rotated.usd",
    "Mine_PTM3" : OBSTACLES_ASSETS + "/Mines/mine_ptm_3_last.usd",
    "Mine_PTM4" : OBSTACLES_ASSETS + "/Mines/mine_ptm_4.usd",
    "Mine_PTKM1R" : OBSTACLES_ASSETS + "/Mines/mine_ptkm_1r.usd",
    "Mine_PMN" : OBSTACLES_ASSETS + "/Mines/mine_pmn.usd",
    "Mine_PMN2" : OBSTACLES_ASSETS + "/Mines/mine_pmn_2.usd",
    "Mine_PMN4" : OBSTACLES_ASSETS + "/Mines/mine_pmn_4.usd",
    "Mine_PFM1" : OBSTACLES_ASSETS + "/Mines/mine_pfm_1.usd",
    "Mine_PMD6" : OBSTACLES_ASSETS + "/Mines/mine_pmd_6.usd",
    "Mine_POM2" : OBSTACLES_ASSETS + "/Mines/mine_pom_2.usd",
    "Mine_POMZ2" : OBSTACLES_ASSETS + "/Mines/mine_pomz_2.usd",
    "Mine_POM3" : OBSTACLES_ASSETS + "/Mines/mine_pom_3.usd",
    "Mine_OZM72" : OBSTACLES_ASSETS + "/Mines/mine_ozm_72.usd",
    "Mine_POB_Pilka" : OBSTACLES_ASSETS + "/Mines/mine_pob_pilka.usd",
    "Mine_MON50" : OBSTACLES_ASSETS + "/Mines/mine_mon_50.usd",
    "Mine_MON90" : OBSTACLES_ASSETS + "/Mines/mine_mon_90.usd",
    "Mine_MON100" : OBSTACLES_ASSETS + "/Mines/mine_mon_100.usd",
    "Mine_MON200" : OBSTACLES_ASSETS + "/Mines/mine_mon_200.usd",
    "Mine_MS3" : OBSTACLES_ASSETS + "/Mines/mine_ms_3.usd",
    "Mine_MS4" : OBSTACLES_ASSETS + "/Mines/mine_ms_4.usd",
    "Mine_ML7" : OBSTACLES_ASSETS + "/Mines/mine_ml_7_main.usd",
    "Mine_ML8" : OBSTACLES_ASSETS + "/Mines/mine_ml_8.usd",
    "Mine_Cloth" : OBSTACLES_ASSETS + "/Mines/mine_cloth_main.usd",
    "Mine_MOB" : OBSTACLES_ASSETS + "/Mines/mine_mob.usd",
    "Mine_Dzika" : OBSTACLES_ASSETS + "/Mines/mine_dzika.usd",
}

GARBAGE = {
    "Garbage_Boot_Main": OBSTACLES_ASSETS + "/Garbage/boot_main.usd",
    "Garbage_Boot_Side": OBSTACLES_ASSETS + "/Garbage/boot_side.usd",
    "Garbage_Brick": OBSTACLES_ASSETS + "/Garbage/brick.usd",
    "Garbage_Can": OBSTACLES_ASSETS + "/Garbage/can.usd",
    "Garbage_Hedgehog": OBSTACLES_ASSETS + "/Garbage/hedgehog.usd",
    "Garbage_Junk_Tyre": OBSTACLES_ASSETS + "/Garbage/junk_tyre.usd",
    "Garbage_Paint_Bucket": OBSTACLES_ASSETS + "/Garbage/paint_bucket.usd",
    "Garbage_Rusty_Key": OBSTACLES_ASSETS + "/Garbage/rusty_key.usd",
    "Garbage_Sandal_Rotated": OBSTACLES_ASSETS + "/Garbage/sandal_rotated.usd",
    "Garbage_Sandal": OBSTACLES_ASSETS + "/Garbage/sandal.usd",
    "Garbage_Tin_Can": OBSTACLES_ASSETS + "/Garbage/tin_can.usd",
    "Garbage_Tyre": OBSTACLES_ASSETS + "/Garbage/tyre.usd",
    "Garbage_Wheel_Second": OBSTACLES_ASSETS + "/Garbage/wheel_second.usd",
    "Garbage_Wheel": OBSTACLES_ASSETS + "/Garbage/wheel.usd",
}

MINES_VEGETATION = OBSTACLES_ASSETS + "/Grass/grass_mine_final.usd"
MINES_SPRING_VEGETATION = OBSTACLES_ASSETS + "/Grass/grass_mine_spring_final.usd"
MINES_LEAVES = OBSTACLES_ASSETS + "/Grass/landmine_leaves_new_final.usd"
MINES_SNOW = OBSTACLES_ASSETS + "/Grass/landmine_snow.usd"

MINES_VEGETATION_TYPES = {
    "Grass": {"path": MINES_VEGETATION, "x": 0.0, "y": 0.0, "z": 0.075},
    "Grass_Spring": {"path": MINES_SPRING_VEGETATION, "x": 0.0, "y": 0.0, "z": 0.075},
    "Leaves": {"path": MINES_LEAVES, "x": -0.01235, "y": 0.05062, "z": -0.11391},
    "Snow": {"path": MINES_SNOW, "x": 0.0, "y": 0.0, "z": 0.073},
}

MINES_VEGETATION_BY_SEASON = {
    "summer": "Grass",
    "spring": "Grass_Spring",
    "autumn": "Leaves",
    "winter": "Snow",
}

# A mine is considered "on the road" (and gets no vegetation) when it sits this
# much above the terrain directly below it. The probe ray starts just below the
# mine's surface so it passes under an elevated road deck and hits true ground.
MINES_VEGETATION_ROAD_ELEVATION_DELTA = 0.10
MINES_VEGETATION_ROAD_PROBE_DELTA = 0.05

# Setup the default simulation environments path
NVIDIA_ASSETS_PATH = str(nucleus.get_assets_root_path())
ISAAC_SIM_ENVIRONMENTS = "/Isaac/Environments"
NVIDIA_SIMULATION_ENVIRONMENTS = {
    "Default Environment": "Grid/default_environment.usd",
    "Black Gridroom": "Grid/gridroom_black.usd",
    "Curved Gridroom": "Grid/gridroom_curved.usd",
    "Hospital": "Hospital/hospital.usd",
    "Office": "Office/office.usd",
    "Simple Room": "Simple_Room/simple_room.usd",
    "Warehouse": "Simple_Warehouse/warehouse.usd",
    "Warehouse with Forklifts": "Simple_Warehouse/warehouse_with_forklifts.usd",
    "Warehouse with Shelves": "Simple_Warehouse/warehouse_multiple_shelves.usd",
    "Full Warehouse": "Simple_Warehouse/full_warehouse.usd",
    "Flat Plane": "Terrains/flat_plane.usd",
    "Rough Plane": "Terrains/rough_plane.usd",
    "Slope Plane": "Terrains/slope.usd",
    "Stairs Plane": "Terrains/stairs.usd",
}

OMNIVERSE_ENVIRONMENTS = {
    "Exhibition Hall": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/NVIDIA/Assets/Scenes/Templates/Interior/ZetCG_ExhibitionHall.usd"
}


SIMULATION_ENVIRONMENTS = {}

SCENES_PATH = ASSET_PATH + "/Scenes"

SIMULATION_ENVIRONMENTS["Multiseasonal Scene"] = "/home/geekscode/Documents/multiseasonal_test_without_road.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Sky Ground Cameras Without Road)"] = "/home/geekscode/Documents/calibration_test_sky_ground_cameras_without_road.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Sky Ground Cameras)"] = "/home/geekscode/Documents/calibration_test_road_curved_sky_ground_cameras.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Sky Ground)"] = "/home/geekscode/Documents/calibration_test_road_curved_sky_ground.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Sky Grass Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved_grass_sky.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Building Grass Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved_labeled_building.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Building Thermal Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved_building_thermal_mine.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Grass Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved_labeled_grass.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Texture Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved_textured.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Labeled Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved_labeled_new.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Road Curved)"] = "/home/geekscode/Documents/calibration_test_road_curved.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Road Straight)"] = "/home/geekscode/Documents/calibration_test_road_straight_main.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Road New)"] = "/home/geekscode/Documents/calibration_test_road_new.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Road Texture)"] = "/home/geekscode/Documents/calibration_test_road.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Grass Texture)"] = "/home/geekscode/Documents/calibration_test_grass_texture.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Low Density Grass)"] = "/home/geekscode/Documents/calibration_test_grass_low_density.usd"
SIMULATION_ENVIRONMENTS["Calibration Scene (Test)"] = "/home/geekscode/Documents/calibration_test.usda"
SIMULATION_ENVIRONMENTS["Calibration Scene (High Density Grass)"] = "/media/geekscode/Developer1/calibration_test_grass_high_density.usd"
SIMULATION_ENVIRONMENTS["Simple Landscape (Updated)"] = "/media/geekscode/Developer1/simple_landscape_backup.usdc"
SIMULATION_ENVIRONMENTS["Forest Scene (New)"] = "/media/geekscode/Developer1/forest_new.usd"
SIMULATION_ENVIRONMENTS["Forest Scene"] = "/media/geekscode/Developer1/forest.usdc"
SIMULATION_ENVIRONMENTS["Simple Landscape"] = "/media/geekscode/Developer1/simple_landscape.usdc"
SIMULATION_ENVIRONMENTS["House Scene"] = "/media/geekscode/Developer1/house_new.usdc"
SIMULATION_ENVIRONMENTS["Calibration Scene"] = "/home/geekscode/Documents/calibration.usda"
SIMULATION_ENVIRONMENTS["Custom World Without Vehicle (usda)"] = "/home/geekscode/Documents/scene_without_vehicle.usda"

# Add the Isaac Sim assets to the list
for asset in NVIDIA_SIMULATION_ENVIRONMENTS:
    SIMULATION_ENVIRONMENTS[asset] = (
        NVIDIA_ASSETS_PATH + ISAAC_SIM_ENVIRONMENTS + "/" + NVIDIA_SIMULATION_ENVIRONMENTS[asset]
    )

# Add the omniverse assets to the list
for asset in OMNIVERSE_ENVIRONMENTS:
    SIMULATION_ENVIRONMENTS[asset] = OMNIVERSE_ENVIRONMENTS[asset]

SIMULATION_ENVIRONMENTS["Custom World"] = "/home/geekscode/Documents/scene.usd"
SIMULATION_ENVIRONMENTS["Custom World Without Vehicle"] = "/home/geekscode/Documents/scene_without_vehicle.usd"
SIMULATION_ENVIRONMENTS["Custom World (Default)"] = "/home/geekscode/Documents/scene_default.usd"
SIMULATION_ENVIRONMENTS["World Not Exist"] = "/home/geekscode/Document321s/scene_default.usd"

BACKENDS = {
    "px4": "px4",
    "ardupilot": "ardupilot",
    "ros2": "ros2",
    "betaflight": "betaflight",
}

# Define the default settings for the simulation environment
WORLD_SETTINGS = {
    'px4': {
        "physics_dt": 1.0 / 250.0,
        "stage_units_in_meters": 1.0,
        "rendering_dt": 1.0 / 60.0,
        "device": "cpu"
    },
    'ardupilot': {
        "physics_dt": 1.0 / 800.0, # Reach communication of 250hz with ardupilot sitl
        "stage_units_in_meters": 1.0,
        "rendering_dt": 1.0 / 100.0,
        "device": "cpu"
    },
    'ros2': {
        "physics_dt": 1.0 / 250.0,
        "stage_units_in_meters": 1.0,
        "rendering_dt": 1.0 / 60.0,
        "device": "cpu"
    },
    'betaflight': {
        "physics_dt": 1.0 / 250.0,
        "stage_units_in_meters": 1.0,
        "rendering_dt": 1.0 / 60.0,
        "device": "cpu"
    },
}
DEFAULT_WORLD_SETTINGS = WORLD_SETTINGS['px4']

# Define where the thumbnail of the vehicle is located
THUMBNAIL = ROBOTS_ASSETS + "/Iris/iris_thumbnail.png"

# Define where the thumbail of the world is located
WORLD_THUMBNAIL = ASSET_PATH + "/Worlds/Empty_thumbnail.png"

TAG_THUMBNAIL = TAGS_ASSETS + "/apriltag_thumbnail.png"

OBSTACLE_THUMBNAIL = OBSTACLES_ASSETS + "/obstacle_thumbnail.png"

BACKENDS_THUMBMAILS_PATH = ASSET_PATH + "/Backends"
BACKENDS_THUMBMAILS = {
    "px4": BACKENDS_THUMBMAILS_PATH + "/px4_logo.png",
    "ardupilot": BACKENDS_THUMBMAILS_PATH + "/ardupilot_logo.png",
    "ros2": BACKENDS_THUMBMAILS_PATH + "/ros2_logo.png",
    "betaflight": BACKENDS_THUMBMAILS_PATH + "/betaflight_logo.png",
}

SPAWN_MAPS = {
    "Building Curved Road": {
        "mines": "mines_map_curved_building.png",
        "trees": "trees_map_curved_building.png",
    },
    "Curved Road": {
        "mines": "mines_map_curved.png",
        "trees": "trees_map_curved.png",
    },
    "Road": {
        "mines": "mines_map_straight.png",
        "trees": "trees_map_straight.png",
    },
    "Empty": {
        "mines": "empty_map.png",
        "trees": "empty_map.png",
    },
}