# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/21/2023

# This script houses a class to visualize sampling order across space.

# Import packages
import sys
import os

import pandas as pd
import numpy as np
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs

from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from user.params.io import INSTALL_DIR, PROJECT_NAME

from user.params.data import (
    LONGITUDE_COL,
    LATITUDE_COL,
    TIME_COL,
    COORD_CRS,
    PROJ_CRS,
    TZ,
    IMG_ID,
)

from utils.logger import setup_logger

import glob
import pytz
import datetime

import os

import matplotlib
import matplotlib.pyplot as plt
import contextlib
from PIL import Image

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from shapely import wkt

import osmnx as ox

import mapclassify as mc


from matplotlib import rc
import matplotlib


class SampleOverTime(object):
    """Class to visually plot sampling order over time."""

    def __init__(self, sample_csv_path, md_tld):
        self.log = setup_logger()

        self.sample_csv_path = sample_csv_path
        self.sample = pd.read_csv(sample_csv_path, engine="pyarrow")

        self.md_paths = glob.glob(os.path.join(md_tld, "*/metadata.csv"))
        self.md = pd.concat(
            [
                pd.read_csv(md_path, engine="pyarrow")
                for md_path in self.md_paths
            ]
        )
        self.md = self.md.drop_duplicates(subset=[IMG_ID])

        self.md = self.md[[IMG_ID, TIME_COL, LATITUDE_COL, LONGITUDE_COL]]
        self.sample = self.sample.merge(self.md, on=IMG_ID, how="left")

        self.sample = gpd.GeoDataFrame(
            self.sample,
            geometry=gpd.points_from_xy(
                self.sample[LONGITUDE_COL], self.sample[LATITUDE_COL]
            ),
            crs=COORD_CRS,
        )
        self.sample = self.sample.to_crs(PROJ_CRS)

        self.sample[TIME_COL] = pd.to_datetime(self.sample[TIME_COL], utc=True)

        # enable LaTeX rendering
        rc("text", usetex=True)
        rc("font", family="serif")

        self.log.success("Successfully loaded sample data.")

    def add_background(self, gdf=None):
        if not gdf:
            self.background = gpd.read_file(gpd.datasets.get_path("nybb"))
            self.background = self.background.to_crs(PROJ_CRS)
            self.log.success("Successfully added default background data.")
        else:
            self.background = gdf
            self.background = self.background.to_crs(PROJ_CRS)
            self.log.success("Successfully added user background data.")

    def add_roads(self, gdf=None):
        if not gdf:
            self.roads = ox.io.load_graphml("../../data/geo/nyc.graphml")
            self.roads = ox.graph_to_gdfs(self.roads, nodes=False)
            self.roads = self.roads.to_crs(PROJ_CRS)
            self.log.success("Successfully added default roads data.")
        else:
            self.roads = gdf
            self.roads = self.roads.to_crs(PROJ_CRS)
            self.log.success("Successfully added user roads data.")

    def generate_frame(self, idx, df):
        fig, ax = plt.subplots(figsize=(20, 20))

        self.background.plot(
            ax=ax, color="white", edgecolor="black", alpha=0.5
        )
        self.roads.plot(ax=ax, color="gray", alpha=0.15, linewidth=1)
        df.plot(ax=ax, color="red", markersize=1, alpha=0.5)

        ax.set_title(f"Sample Over Time: {idx}", fontsize=20)
        # padding above title
        plt.subplots_adjust(top=0.90)

        plt.axis("off")

        plt.savefig(
            f"{INSTALL_DIR}/{PROJECT_NAME}/sample_over_time/{self.id}/{idx}.png",
            bbox_inches="tight",
            pad_inches=0,
        )

        self.log.debug(f"Saved frame {idx}.")

        plt.clf()
        plt.close()

    def generate_frames(self, num_points_per_frame=1000):
        # split metadata sequentially into chunks of size num_points_per_frame
        frames = np.array_split(
            self.sample, len(self.sample) // num_points_per_frame
        )

        self.log.info(
            f"Split sample into {len(frames)} frames of {num_points_per_frame} points each."
        )

        # create output directory
        self.id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.mkdir(f"{INSTALL_DIR}/{PROJECT_NAME}/sample_over_time/{self.id}")

        self.log.info(f"Generating frames for {self.id}.")
        # generate frames in parallel
        Parallel(n_jobs=16)(
            delayed(self.generate_frame)(idx, df)
            for idx, df in tqdm(
                enumerate(frames), desc="Generating frames.", total=len(frames)
            )
        )

        self.log.success(f"Successfully generated frames for {self.id}.")

    def generate_gif(self):
        self.log.info(f"Generating GIF for {self.id}.")

        frames = glob.glob(f"{INSTALL_DIR}/{PROJECT_NAME}/sample_over_time/{self.id}/*.png")
        frames = sorted(
            frames, key=lambda x: int(x.split("/")[-1].split(".")[0])
        )

        frames = [Image.open(frame) for frame in frames]

        frames[0].save(
            f"{INSTALL_DIR}/{PROJECT_NAME}/sample_over_time/{self.id}.gif",
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=50,
            loop=0,
        )

        self.log.success(f"Successfully generated GIF for {self.id}.")

    def __call__(self):
        self.add_background()
        self.add_roads()
        self.generate_frames(num_points_per_frame=200000)
        self.generate_gif()
