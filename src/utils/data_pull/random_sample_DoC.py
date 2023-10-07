# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot 
# Last Edited: 10/06/2023

# This script is used to randomly sample N images from a selected day of coverage in the Nexar data. 

# Module Imports 
import os 
from glob import glob
import random
import logging 
from tqdm import tqdm
from h3 import h3
import pandas as pd 
import geopandas as gpd 
from pathlib import Path
import numpy as np

class ImagePull:
    def __init__(self, proj_path, DoC): 
        self.log = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.N = 1000
        self.proj_path = proj_path
        self.DoC = DoC
        self.csvs = glob(os.path.join(self.proj_path, self.DoC, "*", "*.csv"))

        # hex should be in parent directory of csv file
        # filter csvs list to only include csv files with a parent directory that is a valid h3 hexagon 
        self.csvs = [csv for csv in self.csvs if h3.h3_is_valid(hex(np.int64(Path(csv).parent.name)))]
        self.log.info(f"Read {len(self.csvs)} csvs corresponding to valid h3 hexagons from {self.proj_path}/{self.DoC}")
        self.image_list = []
        for csv in tqdm(self.csvs, desc=f"Reading image metadata csvs from {self.proj_path}/{self.DoC}"):
            self.image_list.append(pd.read_csv(csv, engine='pyarrow'))
        
        self.log.info(f"Concatenating image metadata csvs from {self.proj_path}/{self.DoC}...")
        self.image_list = pd.concat(self.image_list)
        self.log.info(f"Read {len(self.image_list.index)} images from {self.proj_path}/{self.DoC}")

        images_on_disk = glob(os.path.join(self.proj_path, self.DoC, "*", "*", "*.jpg"))
        self.frames_on_disk = [Path(image).stem for image in images_on_disk]
        del images_on_disk 

        len_before_filter = len(self.image_list.index)
        self.image_list = self.image_list[self.image_list['frame_id'].isin(self.frames_on_disk)]
        self.log.info(f"Filtered out {len_before_filter - len(self.image_list.index)} images that are not on disk")

        self.log.info(f"Initialized ImagePull with DoC={self.DoC}")

    
    def pull_images(self, N, output_dir, coords=None, proximity=1000, time_delta=-1):
        self.N = N 
        # Prepend 'output' to output_dir
        output_dir = os.path.join("output", output_dir)
        # Add number of images to output_dir
        output_dir = f"{output_dir}_{self.N}"
        # Add DoC to output_dir
        output_dir = f"{output_dir}_{self.DoC}"

        # Make output dir, exists_ok=True
        os.makedirs(output_dir, exist_ok=True)


        # Coordinate filtering, if applicable 
        if len(coords) > 0:

            # coords needs to be a geodataframe, if not raise error 
            if not isinstance(coords, gpd.GeoDataFrame):
                self.log.error("'coords' must be a geodataframe")
                raise TypeError("coords must be a geodataframe")

            self.image_list = gpd.GeoDataFrame(self.image_list,
                geometry=gpd.points_from_xy(
                    self.image_list["gps_info.longitude"], 
                    self.image_list["gps_info.latitude"]), crs="EPSG:4326")
                    
            self.image_list = self.image_list.to_crs("EPSG:2263")
            
            # only pull images within proximity of coords
            self.image_list = gpd.sjoin_nearest(self.image_list, coords,how='left',max_distance=proximity, distance_col='distance')
            self.image_list = self.image_list[self.image_list['distance'] <= proximity]

            if time_delta > 0:
                self.log.info(f"Filtering images to only include images within {time_delta} minutes of nearest event in coords...")
                # now, only keep images within 30 minutes of the flooding event 
                self.image_list['captured_at'] = pd.to_datetime(self.image_list['captured_at'], unit='ms')
                self.image_list['Created Date'] = pd.to_datetime(self.image_list['Created Date'])

                self.image_list['time_diff'] = self.image_list['captured_at'] - self.image_list['Created Date']
                self.image_list = self.image_list[self.image_list['time_diff'] <= pd.Timedelta(minutes=time_delta)]


        

        # Randomly sample N images from image_list
        sample = self.image_list.sample(n=self.N, random_state=1)

        dropped_files = 0
        # Copy sampled images to output_dir
        for image in tqdm(sample['frame_id'].apply(lambda x: f"{x}.jpg"), desc=f"Copying images to {output_dir}"):
            try:
                img_path = glob(os.path.join(self.proj_path, self.DoC, "*", "*", image))[0]
            except IndexError:
                self.log.warning(f"Could not find image {image} in {self.proj_path}/{self.DoC}")
                dropped_files += 1
            os.system(f"cp {img_path} {output_dir}")

        self.log.info(f"Successfully copied {self.N} images to {output_dir}")
        if dropped_files > 0:
            self.log.warning(f"Could not find {dropped_files} images in {self.proj_path}/{self.DoC}")


        # Return number of images copied
        return len(os.listdir(output_dir))

    def __run__(self, N, output_dir, coords=None, proximity=1000):
        self.N = N
        self.output_dir = output_dir
        self.coords = coords
        self.proximity = proximity
        self.pull_images(self.N, self.output_dir, self.coords, self.proximity)



