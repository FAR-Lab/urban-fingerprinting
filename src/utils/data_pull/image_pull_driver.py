# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot 
# Last Edited: 10/06/2023 

# This script is used to randomly sample N images from a selected day of coverage in the Nexar data. 


from random_sample_DoC import ImagePull
from fire import Fire
import pandas as pd 
import geopandas as gpd


if __name__ == '__main__':
    instance = ImagePull("/share/ju/nexar_data/nexar-scraper","2023-09-29")

    flooding = pd.read_csv("../../../data/coords/sep29_flooding.csv", engine='pyarrow')
    flooding = gpd.GeoDataFrame(flooding, geometry=gpd.points_from_xy(flooding.Longitude, flooding.Latitude), crs="EPSG:4326")
    flooding = flooding.to_crs("EPSG:2263")

    instance.pull_images(1000, "flooding_filtered_on_311_w_time_prox", coords=flooding, proximity=100)
    





