# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 10/06/2023

# This script is used to randomly sample N images from a selected day of coverage in the Nexar data.


from random_sample_DoC import ImagePull
from fire import Fire
import pandas as pd
import geopandas as gpd


if __name__ == "__main__":
    instance = ImagePull("/share/ju/nexar_data/2023", "2023-08-18")

    # flooding = pd.read_csv("../../../data/coords/sep29_flooding.csv", engine='pyarrow')
    # flooding = gpd.GeoDataFrame(flooding, geometry=gpd.points_from_xy(flooding.Longitude, flooding.Latitude), crs="EPSG:4326")
    # flooding = flooding.to_crs("EPSG:2263")

    instance.pull_images(100000, "ped_cropping_test")
