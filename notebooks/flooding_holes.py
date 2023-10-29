# %%
import os 
import sys

# %%
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


# %%
from user.params.data import *

# %%
import pandas as pd 
import geopandas as gpd 
import matplotlib.pyplot as plt
from shapely import Polygon
import osmnx as ox 
import contextily as ctx
import xyzservices.providers as xyz
# %%
import h3.api.basic_str as h3

# %%
frames_df = pd.read_csv('../output/df/2023-09-29/md.csv', engine='pyarrow')

# %%
flooding = pd.read_csv('../data/coords/sep29_flooding.csv', engine='pyarrow')

# %%
frames_gdf = gpd.GeoDataFrame(frames_df, geometry=gpd.points_from_xy(frames_df[LONGITUDE_COL], frames_df[LATITUDE_COL]), crs=COORD_CRS)
frames_gdf = frames_gdf.to_crs(PROJ_CRS)

# %%
flooding_gdf = gpd.GeoDataFrame(flooding, geometry=gpd.points_from_xy(flooding[LONGITUDE_COL_311], flooding[LATITUDE_COL_311]), crs=COORD_CRS)
flooding_gdf = flooding_gdf.to_crs(PROJ_CRS)

# %%
def h3_to_polygon(h3_id):
    polygon = h3.h3_to_geo_boundary(h3_id)
    polygon = tuple(coord[::-1] for coord in polygon)
    polygon = Polygon(polygon)
    # flip the order of points 
    
    polygon = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs=COORD_CRS)
    
    polygon = polygon.to_crs(PROJ_CRS)
    
    return polygon
    

# %%
def crop_gdf(gdf, geometry):
    return gpd.clip(gdf, geometry)

# %%

# %%
frames_gdf_crop = crop_gdf(frames_gdf, h3_to_polygon('862a10747ffffff'))
#print(len(frames_gdf_crop))
flooding_gdf_crop = crop_gdf(flooding_gdf, h3_to_polygon('862a10747ffffff'))

nyc_roads = ox.io.load_graphml('../data/geo/nyc.graphml')
nyc_roads_gdf = ox.graph_to_gdfs(nyc_roads, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
nyc_roads_gdf = nyc_roads_gdf.to_crs(PROJ_CRS)

nyc_roads_gdf_crop = crop_gdf(nyc_roads_gdf, h3_to_polygon('862a10747ffffff'))

# %%
fig, ax = plt.subplots(figsize=(20,20))


frames_gdf_crop.plot(ax=ax, color='blue', markersize=1)
flooding_gdf_crop.plot(ax=ax, color='red', markersize=2.5)
nyc_roads_gdf_crop.plot(ax=ax, color='black', linewidth=0.5)

# satellite base
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)


plt.axis('off')

# %%

plt.savefig('./output/flooding-holes.png', dpi=300, bbox_inches='tight')

