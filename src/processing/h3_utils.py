# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub CoPilot 
# Last Modified: 10/29/2023

# This script houses utility functions for cropping geodataframes with H3 hexagons

import os 
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user.params.data import *

import geopandas as gpd 
import h3.api.basic_str as h3s 
import h3.api.numpy_int as h3n

from shapely.geometry import Polygon, MultiPolygon, Point

def h3_to_polygon(id): 
    match id: 
        case str(): 
            polygon = h3s.h3_to_geo_boundary(id)
        case np.int64():
            polygon = h3n.h3_to_geo_boundary(id)

        
        
    polygon = tuple(coord[::-1] for coord in polygon)
    polygon = Polygon(polygon)
    # flip the order of points 
    
    polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs=COORD_CRS)
    
    polygon = polygon.to_crs(PROJ_CRS)

    return polygon 

def crop_within_polygon(gdf, polygon):
    return gpd.clip(gdf, polygon)
    
    

