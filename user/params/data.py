# FARLAB: UrbanECG 
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 10/08/2023

# This script contains variables that refer to dataset features used in analysis. 

import os 
import sys

LONGITUDE_COL = 'gps_info.longitude'
LATITUDE_COL = 'gps_info.latitude'
TIME_COL = 'captured_at'

ORIENTATION_COL = 'camera_heading'
DIRECTION_COL = 'direction'

COORD_CRS = 'EPSG:4326'
PROJ_CRS = 'EPSG:2263'

TZ = 'America/New_York'

IMG_ID = 'frame_id'

TOP_LEVEL_DIR = "/share/ju/nexar_data/2023"

# Perspective Settings 
VIEW_DISTANCE = 100 # feet
VIEW_CONE = 90 # degrees (on each side from center of perspective)