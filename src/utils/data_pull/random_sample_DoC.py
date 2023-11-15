# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 10/06/2023

# This script is used to randomly sample N images from a selected day of coverage in the Nexar data.

# Module Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join("../..", "src")))

from src.processing.geometric_utils import Frame, Perspective
from src.utils.logger import setup_logger
from glob import glob
import random
import logging
from tqdm import tqdm
from h3 import h3
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np

from tqdm.contrib.concurrent import process_map


class ImagePull:
    def __init__(self, proj_path, DoC):
        self.log = setup_logger("ImagePull")
        self.log.setLevel(logging.DEBUG)
        self.N = 1000
        self.proj_path = proj_path
        self.DoC = DoC
        self.csvs = glob(os.path.join(self.proj_path, self.DoC, "*", "*.csv"))
        self.pull_from_all_when_no_coords = True

        # hex should be in parent directory of csv file
        # filter csvs list to only include csv files with a parent directory that is a valid h3 hexagon
        self.csvs = [
            csv
            for csv in self.csvs
            if h3.h3_is_valid(hex(np.int64(Path(csv).parent.name)))
        ]
        self.log.info(
            f"Read {len(self.csvs)} csvs corresponding to valid h3 hexagons from {self.proj_path}/{self.DoC}"
        )
        self.image_list = []
        for csv in tqdm(
            self.csvs,
            desc=f"Reading image metadata csvs from {self.proj_path}/{self.DoC}",
        ):
            self.image_list.append(pd.read_csv(csv, engine="pyarrow"))

        self.log.info(
            f"Concatenating image metadata csvs from {self.proj_path}/{self.DoC}..."
        )
        self.image_list = pd.concat(self.image_list)
        self.log.info(
            f"Read {len(self.image_list.index)} images from {self.proj_path}/{self.DoC}"
        )

        images_on_disk = glob(
            os.path.join(self.proj_path, self.DoC, "*", "*", "*.jpg")
        )
        self.frames_on_disk = [Path(image).stem for image in images_on_disk]
        del images_on_disk

        len_before_filter = len(self.image_list.index)
        self.image_list = self.image_list[
            self.image_list["frame_id"].isin(self.frames_on_disk)
        ]
        self.log.info(
            f"Filtered out {len_before_filter - len(self.image_list.index)} images that are not on disk"
        )

        self.log.info(f"Initialized ImagePull with DoC={self.DoC}")

    def pull_images(
        self, N, output_dir, coords=pd.DataFrame(), proximity=50, time_delta=-1
    ):
        self.N = N
        # Prepend 'output' to output_dir
        #output_dir = os.path.join("output", str(output_dir))
        # Add number of images to output_dir
        output_dir = f"{output_dir}_{self.N}"
        # Add DoC to output_dir
        #output_dir = f"{output_dir}_{self.DoC}"

        # Make output dir, exists_ok=True
        os.makedirs(output_dir, exist_ok=True)

        # Coordinate filtering, if applicable
        if coords and len(coords) > 0:
            self.log.info(
                f"Filtering images to only include images within {proximity} feet of coords..."
            )

            # coords needs to be a geodataframe, if not raise error
            if not isinstance(coords, gpd.GeoDataFrame):
                self.log.error("'coords' must be a geodataframe")
                raise TypeError("coords must be a geodataframe")

            # if image list is not an instance of a geodataframe, make it one
            if not isinstance(self.image_list, gpd.GeoDataFrame):
                self.log.info(
                    "First run: converting image_list to geodataframe..."
                )
                self.image_list = gpd.GeoDataFrame(
                    self.image_list,
                    geometry=gpd.points_from_xy(
                        self.image_list["gps_info.longitude"],
                        self.image_list["gps_info.latitude"],
                    ),
                    crs="EPSG:4326",
                )

                self.image_list = self.image_list.to_crs("EPSG:2263")

            # make sure 'index_left' and 'index_right' columns don't exist
            if "index_left" in self.image_list.columns:
                self.image_list = self.image_list.drop(columns=["index_left"])
            if "index_right" in self.image_list.columns:
                self.image_list = self.image_list.drop(columns=["index_right"])

            if "index_left" in coords.columns:
                coords = coords.drop(columns=["index_left"])
            if "index_right" in coords.columns:
                coords = coords.drop(columns=["index_right"])

            # only pull images within proximity of coords
            close_images = gpd.sjoin_nearest(
                self.image_list,
                coords,
                how="left",
                max_distance=proximity,
                distance_col="distance",
            )
            close_images = close_images[close_images["distance"] <= proximity]

            # out of these images, only keep those that actually depict the coordinates
            self.log.info(f"{len(close_images.index)} images within proximity")
            close_images = close_images[
                close_images.apply(
                    lambda x: Frame(x).depicts_coordinates(
                        (x["Longitude"], x["Latitude"])
                    ),
                    axis=1,
                )
            ]
            self.log.info("Filtered out images that do not depict coords, now have {} images".format(len(close_images.index)))

            if time_delta > 0:
                self.log.info(
                    f"Filtering images to only include images within {time_delta} minutes of nearest event in coords..."
                )
                # now, only keep images within 30 minutes of the flooding event
                close_images["captured_at"] = pd.to_datetime(
                    close_images["captured_at"], unit="ms"
                )

                close_images['captured_at'] = close_images['captured_at'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

                close_images["Created Date"] = pd.to_datetime(
                    close_images["Created Date"]
                )
                close_images['Created Date'] = close_images['Created Date'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

                close_images["time_diff"] = (
                    abs(close_images["captured_at"] - close_images["Created Date"])
                )
                close_images = close_images[
                    close_images["time_diff"]
                    <= pd.Timedelta(minutes=time_delta)
                ]

        else:
            if self.pull_from_all_when_no_coords:
                self.log.info(
                    "No coords provided, using all images in image_list"
                )
                close_images = self.image_list
            else:
                self.log.error(
                    "No coords provided, and pull_from_all_when_no_coords is False, exiting..."
                )
                return
           
        if len(close_images.index) == 0:
            self.log.error("No images found within proximity of coords, exiting...")
            return

        # Randomly sample N images from image_list
        if len(close_images.index) > self.N:
            sample = close_images.sample(n=self.N, random_state=1)
        else:
            self.log.warning(
                f"Number of images in image_list is less than {self.N}, returning all images in image_list"
            )
            self.N = len(close_images.index)
            sample = close_images

        dropped_files = 0
        # Copy sampled images to output_dir
        # for image in process_map(lambda x: f"{x}.jpg", sample['frame_id'], desc=f"Copying images to {output_dir}"):
        sample.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
        sample['path'] = sample['frame_id'].apply(lambda x: f"{x}.jpg")
        for idx, image in tqdm(
            sample.iterrows(),
            desc=f"Copying images to {output_dir}",
        ):
            try:
                img_path = glob(
                    os.path.join(self.proj_path, self.DoC, "*", "*", image["path"])
                )[0]
            except IndexError:
                self.log.warning(
                    f"Could not find image {image['path']} in {self.proj_path}/{self.DoC}"
                )
                dropped_files += 1

            # snew_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}
            try:
                if coords and len(coords) > 0:
                    output_path = f"{image['frame_id']}_{image['distance']:.3f}_{str(image['time_diff'])}_{image['Complaint Type'].replace('/','-')}_{image['Descriptor'].replace('/','-')}.jpg"
                else: 
                    output_path = f"{image['frame_id']}.jpg"

                os.symlink(
                    img_path,
                    os.path.join(
                        output_dir,
                        output_path
                    )
                )
            except FileExistsError:
                self.log.warning(
                    f"Image {image} already exists in {output_dir}"
                )
                continue 
            except Exception as e:
                self.log.error(e)
                continue
 

        self.log.info(f"Successfully copied {self.N} images to {output_dir}")
        if dropped_files > 0:
            self.log.warning(
                f"Could not find {dropped_files} images in {self.proj_path}/{self.DoC}"
            )

        del sample
        del close_images

        # Return path of output_dir
        return output_dir

    def __call__(self, N, output_dir, coords=None, proximity=1000):
        self.N = int(N)
        self.output_dir = output_dir
        self.coords = coords
        self.proximity = proximity
        self.pull_images(self.N, self.output_dir, self.coords, self.proximity)
