# FARLAB - UrbanECG Project
# Developer: @mattwfranchi, with help from GitHub CoPilot 
# Last Modified: 10/29/2023 

# This script pulls Nexar frames near 311 complaints. 

import os 
import sys 
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.data_pull.random_sample_DoC import ImagePull
from user.params.data import *

import pandas as pd
import geopandas as gpd

if __name__ == '__main__': 

    nyc_311_complaints_subsets = glob("../../data/coords/Descriptor*.csv")


    kwds_to_remove = ["Noise", "Talking", "Banging", "Music", "Loud", "Air", "Taste"]

    for kw in kwds_to_remove:
        nyc_311_complaints_subsets = [x for x in nyc_311_complaints_subsets if kw.lower() not in x.lower()]

    nyc_311_complaints_names = [x.split('_')[1] for x in nyc_311_complaints_subsets]

    

 

    DoCs = ["2023-08-18", "2023-08-20", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24"]

    for day in DoCs: 
        image_pull = ImagePull("/share/ju/nexar_data/2023", day)
        image_pull.log.info(f"Pulling images from {day}.")

        os.makedirs('../../data/nyc_311', exist_ok=True)

        for i in range(len(nyc_311_complaints_subsets)):
            os.makedirs('../../data/nyc_311/{}'.format(nyc_311_complaints_names[i]), exist_ok=True)

            try: 
                nyc_311_complaints = pd.read_csv(nyc_311_complaints_subsets[i], engine='pyarrow')
            except Exception as e: 
                image_pull.log.error(e)
                continue

            nyc_311_complaints = gpd.GeoDataFrame(nyc_311_complaints, geometry=gpd.points_from_xy(nyc_311_complaints[LONGITUDE_COL_311], nyc_311_complaints[LATITUDE_COL_311]), crs="EPSG:4326")
            nyc_311_complaints = nyc_311_complaints.to_crs(PROJ_CRS)

            image_pull.pull_images(1000,f"/share/ju/urbanECG/training_datasets/{nyc_311_complaints_names[i]}", coords=nyc_311_complaints, proximity=30, time_delta=15)

