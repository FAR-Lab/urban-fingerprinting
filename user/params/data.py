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
NYC_311_DATA_CSV_PATH = "/share/ju/urbanECG/data/geo/311_Service_Requests_from_2010_to_Present.csv"

# Perspective Settings 
VIEW_DISTANCE = 100 # feet
VIEW_CONE = 67.35 # Nexar cameras have 170 diagonal FOV, so cone should be about 168.5/2 degrees

# 311 Fields 
ID_COL_311 = 'Unique Key'
LONGITUDE_COL_311 = 'Longitude'
LATITUDE_COL_311 = 'Latitude'

XSTREET_1_311 = 'Cross Street 1'
XSTREET_2_311 = 'Cross Street 2'

AGENCY_311 = 'Agency'
AGENCY_NAME_311 = 'Agency Name'

START_DATE_311 = 'Created Date'
END_DATE_311 = 'Closed Date'

DESC_COL_311 = 'Descriptor'
TYPE_COL_311 = 'Complaint Type'
