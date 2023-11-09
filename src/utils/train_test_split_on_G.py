# FARLAB - UrbanECG Project
# Developer: @mattwfranchi
# Last Edited: 11/07/2023

# This script houses a class to run a train/test split on a list of DaysOfCoverage, randomly sampling from a list of geographic areas G.

# Module Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)

from random import shuffle
from glob import glob

from src.utils.logger import setup_logger
from src.utils.timer import timer

from user.params.geo import ROAD_GRAPH_PATH
from user.params.io import TOP_LEVEL_FRAMES_DIR, TRAIN_TEST_SPLITS_PATH
from user.params.data import (
    IMG_ID,
    LONGITUDE_COL,
    LATITUDE_COL,
    COORD_CRS,
    PROJ_CRS,
)

from src.analysis.graph import G

import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm


class TTSplit_G:
    def __init__(
        self,
        days_of_coverage,
        geo_path="../../data/geo/nyc_ct/nyct2020.shp",
        grouping_col="CTLabel",
        output_prefix="train_test_sample",
    ):
        self.log = setup_logger("Train/Test Split on G")
        self.log.setLevel("INFO")

        self.output_prefix = output_prefix
        self.output_dir = f"{TRAIN_TEST_SPLITS_PATH}/{self.output_prefix}"

        # make output_dir, including parents, if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.roads = ox.io.load_graphml(ROAD_GRAPH_PATH)

        self.days_of_coverage = days_of_coverage

        self.G = gpd.read_file(geo_path, crs=COORD_CRS)
        self.G = self.G.to_crs(PROJ_CRS)

        self.grouping_col = grouping_col

        self.data_graph = G(TOP_LEVEL_FRAMES_DIR, ROAD_GRAPH_PATH)

        for day in self.days_of_coverage:
            self.data_graph.init_day_of_coverage(day)

        self.all_data = self.data_graph.join_days(days_of_coverage).frames_data
        self.log.info(f"Head of all_data:\n{self.all_data.head().to_string()}")

        # Spatial merge of G onto all_data.md
        if not isinstance(self.all_data, gpd.GeoDataFrame):
            self.log.warning(
                "all_data.frames_data is not a GeoDataFrame. Creating one now."
            )
            self.all_data = gpd.GeoDataFrame(
                self.all_data,
                geometry=gpd.points_from_xy(
                    self.all_data[LONGITUDE_COL],
                    self.all_data[LATITUDE_COL],
                    crs=COORD_CRS,
                ),
            )
            self.all_data = self.all_data.to_crs(PROJ_CRS)
        else:
            if self.all_data.crs != PROJ_CRS:
                self.log.warning(
                    f"all_data.frames_data is a GeoDataFrame, but its CRS is not {PROJ_CRS}. Reprojecting now."
                )
                self.all_data = self.all_data.to_crs(PROJ_CRS)

        self.frames_data_plot()

        self.log.info(
            f"{len(self.G[grouping_col].unique())} geographic areas in G."
        )

        self.validation_plot()

        self.all_data = gpd.sjoin(
            self.all_data, self.G, how="left", predicate="within"
        )
        # log head of all_data pretty
        self.log.info(f"Head of all_data:\n{self.all_data.head().to_string()}")

        self.log.success(
            f"{len(self.all_data)} frames joined to geographic areas."
        )

        self.log.info("Successfully initialized TTSplit_G")

    def __call__(self, train_pct=0.8, write=True):

        self.log.info(
            f"Splitting {len(self.all_data)} frames into train and test data with {train_pct * 100}% of frames in training data and {100 - (train_pct * 100)}% of frames in testing data."
        )

        g_ids = list(set(self.all_data[self.grouping_col].values))
        shuffle(g_ids)

        self.log.info(f"Randomly shuffled {len(g_ids)} geographic areas.")

        train_ids = g_ids[: int(train_pct * len(g_ids))]
        test_ids = g_ids[int(train_pct * len(g_ids)) :]

        self.log.info(
            f"Split {len(g_ids)} geographic areas into {len(train_ids)} training areas and {len(test_ids)} testing areas."
        )

        train_data = self.all_data[
            self.all_data[self.grouping_col].isin(train_ids)
        ]
        test_data = self.all_data[
            self.all_data[self.grouping_col].isin(test_ids)
        ]

        # make sure no frames are in both train and test data
        if (
            len(
                set(train_data[IMG_ID].values).intersection(
                    set(test_data[IMG_ID].values)
                )
            )
            > 0
        ):
            self.log.warning(
                "Frames are in both train and test data. Removing duplicates from test data."
            )
            test_data = test_data[
                ~test_data[IMG_ID].isin(train_data[IMG_ID].values)
            ]

        # make sure output dir exists
        os.makedirs(f"{self.output_dir}/{self.output_prefix}", exist_ok=True)

        # write train and test data (if write=True)
        if write:
            train_data.to_csv(
                f"{self.output_dir}/{self.output_prefix}_train.csv"
            )
            self.log.success(f"Wrote train data to {self.output_dir}.")
            test_data.to_csv(
                f"{self.output_dir}/{self.output_prefix}_test.csv"
            )
            self.log.success(f"Wrote test data to {self.output_dir}.")

            # symlink frames from train and test data
            self.symlink_frames(train_data, prefix="train")
            self.symlink_frames(test_data, prefix="test")

            self.log.success(
                f"Symlinked frames from train and test data to {self.output_dir}."
            )

            self.split_plot(train_data, test_data)

        return train_data, test_data

    def frames_data_plot(self):
        fig, ax = plt.subplots(figsize=(20, 20))

        self.all_data.plot(ax=ax, color="red", alpha=0.5, markersize=0.5)

        plt.savefig(f"{self.output_dir}/frames_data_plot.png")
        plt.clf()
        self.log.success(f"Wrote frames_data plot to {self.output_dir}.")

    def validation_plot(self):
        fig, ax = plt.subplots(figsize=(20, 20))

        self.G.plot(
            ax=ax, color="white", edgecolor="black", alpha=0.5, linewidth=1
        )
        self.all_data.plot(ax=ax, color="red", alpha=0.5, markersize=0.5)

        plt.savefig(f"{self.output_dir}/validation_plot.png")
        plt.clf()
        self.log.success(f"Wrote validation plot to {self.output_dir}.")

    def split_plot(self, train, test): 
        fig, ax = plt.subplots(figsize=(20, 20))

        self.G.plot(
            ax=ax, color="white", edgecolor="black", alpha=0.5, linewidth=1
        )

        train.plot(ax=ax, color="blue", alpha=0.5, markersize=0.5)
        test.plot(ax=ax, color="red", alpha=0.5, markersize=0.5)

        plt.legend(["Groupings", "Train", "Test"])

        plt.savefig(f"{self.output_dir}/split_plot.png")
        plt.clf() 

        self.log.success(f"Wrote split plot to {self.output_dir}.")


    def symlink_frames(self, data, prefix="split"):
        # Empty directory if it exists
        if os.path.exists(f"{self.output_dir}/{prefix}"):
            self.log.warning(
                f"{self.output_dir}/{prefix} already exists. Emptying directory."
            )
            os.system(f"rm -rf {self.output_dir}/{prefix}/*")

        os.makedirs(f"{self.output_dir}/{prefix}", exist_ok=True)

        """Symlink frames from data to self.output_dir"""
        for frame_id in tqdm(
            data[IMG_ID].values, desc=f"Symlinking {prefix} frames"
        ):
            # find img path first
            for day in self.days_of_coverage:
                try:
                    img_path = glob(
                        f"{TOP_LEVEL_FRAMES_DIR}/{day}/*/frames/{frame_id}.jpg"
                    )[0]
                except IndexError:
                    continue

            os.symlink(img_path, f"{self.output_dir}/{prefix}/{frame_id}.jpg")
