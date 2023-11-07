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


from src.utils.logger import setup_logger
from src.utils.timer import timer

from user.params.geo import ROAD_GRAPH_PATH
from user.params.io import TOP_LEVEL_FRAMES_DIR, TRAIN_TEST_SPLITS_PATH

from src.analysis.graph import G

import pandas as pd 
import osmnx as ox
import geopandas as gpd


class TTSplit_G:
    def __init__(self, days_of_coverage, geo_path="../../data/geo/nyc_ct/nyct2020.shp", grouping_col="CT2020", output_prefix="train_test_sample"):
        self.log = setup_logger("Train/Test Split on G")
        self.log.setLevel("INFO")

        self.output_prefix = output_prefix
        self.output_dir = f"{TRAIN_TEST_SPLITS_PATH}/{self.output_prefix}"

        # make output_dir, including parents, if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.roads = ox.io.load_graphml(ROAD_GRAPH_PATH)

        self.days_of_coverage = days_of_coverage
        
        self.G = gpd.read_file(geo_path)

        self.data_graph = G(TOP_LEVEL_FRAMES_DIR, ROAD_GRAPH_PATH)

        for day in self.days_of_coverage:
            self.data_graph.add_day_of_coverage(day)

        self.all_data = self.data_graph.join_days(days_of_coverage)

        # Spatial merge of G onto all_data.md 
        self.all_data = gpd.sjoin(self.all_data, self.G, how="left", op="within")

        self.all_data_grouped = self.all_data.groupby(grouping_col)

        self.log.info("Successfully initialized TTSplit_G")

    def __call__(self, train_pct=0.8, write=True):

        # get all names in self.all_data_grouped
        all_names = list(self.all_data_grouped.groups.keys())

        # shuffle all_names
        shuffle(all_names)

        # get train_names and test_names
        train_names = all_names[:int(train_pct * len(all_names))]
        test_names = all_names[int(train_pct * len(all_names)):]

        # get train and test data
        train_data = self.all_data_grouped.get_group(train_names)
        test_data = self.all_data_grouped.get_group(test_names)

        # write train and test data (if write=True)
        if write:
            train_data.to_csv(f"{self.output_prefix}_train.csv")
            test_data.to_csv(f"{self.output_prefix}_test.csv")
        
        return train_data, test_data


