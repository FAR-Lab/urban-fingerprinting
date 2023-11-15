# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 11/01/2023

# This script is used to take the roads of a graph G, and two sets of frames F, and compare traffic densities across G between the two sets F.

import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
import os
import sys
from glob import glob
from PIL import Image


import contextlib

from fire import Fire
from tqdm import tqdm

tqdm.pandas()
from tqdm.contrib.concurrent import process_map

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import contextily as ctx

mplstyle.use(["ggplot", "fast"])
import uuid
from joblib import Parallel, delayed
from itertools import chain
import datetime
import pytz


import gc

# Set up logging
import logging

logging.basicConfig(level=logging.CRITICAL)

from shapely import wkt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)

from src.utils import coco_mappings as cm
from src.utils.logger import setup_logger
from src.visualization import animated_map as am
from src.analysis.day_of_coverage import DayOfCoverage
from src.processing.h3_utils import h3_to_polygon, crop_within_polygon

from user.params.io import INSTALL_DIR, PROJECT_NAME

from user.params.data import (
    LONGITUDE_COL,
    LATITUDE_COL,
    TIME_COL,
    COORD_CRS,
    PROJ_CRS,
    IMG_ID,
    TZ,
)

# Constant for number of workers to use in parallel, should be equal to number of cores on machine
NUM_CORES = os.getenv("SLURM_CPUS_ON_NODE")
# Check if none
if NUM_CORES is None:
    # Set to 8
    NUM_CORES = 8
else:
    NUM_CORES = int(NUM_CORES)

NUM_CORES = 8


class G:
    """
    A class representing a graph object.

    Attributes:
    - DEBUG_MODE (bool): A flag indicating whether debug mode is on or off.
    - WRITE_MODE (bool): A flag indicating whether write mode is on or off.
    - log (logging.Logger): A logger object for logging messages.
    - PROJ_PATH (str): The path to the project directory.
    - days_of_coverage (list): A list of DayOfCoverage objects representing the days of coverage.
    - geo (networkx.MultiDiGraph): A MultiDiGraph object representing the graph.
    - gdf_nodes (geopandas.GeoDataFrame): A GeoDataFrame object representing the nodes of the graph.
    - gdf_edges (geopandas.GeoDataFrame): A GeoDataFrame object representing the edges of the graph.

    Methods:
    - toggle_debug(): Toggles the debug mode flag.
    - toggle_latex_font(): Toggles the LaTeX font flag.
    - get_frames_worker(folder): Returns a list of all .jpg files in the given folder.
    - get_md_worker(md_csv): Returns a pandas DataFrame object representing the metadata from the given CSV file.
    - get_data(day_of_coverage, num_workers): Returns a GeoDataFrame object representing the metadata for the given day of coverage.
    - load_day_of_coverage(day_of_coverage): Returns a GeoDataFrame object representing the metadata for the given day of coverage.
    - add_day_of_coverage(day_of_coverage): Adds a DayOfCoverage object representing the given day of coverage to the graph.
    - get_day_of_coverage(day_of_coverage): Returns a DayOfCoverage object representing the given day of coverage.
    - nearest_road_worker(subset): Returns a list of the nearest edges for each row in the given subset of metadata.
    - coverage_to_nearest_road(day_of_coverage): Calculates the nearest road for each frame in the given day of coverage.
    - add_detections(day_of_coverage): Adds a GeoDataFrame object representing the detections for the given day of coverage to the graph.
    - plot_edges(): Plots the edges of the graph.
    - plot_coverage(day_of_coverage): Plots the coverage of the given day of coverage.
    - plot_detections(day_of_coverage, class_id): Plots the detections of the given class for the given day of coverage.
    - join_days(DoCs): Concatenates the metadata, nearest edges, and detections of multiple days of coverage into a single DayOfCoverage object.
    - coco_agg_mappings(operation='mean', data=pd.DataFrame()): Returns a dictionary with class IDs as keys and the specified operation as values.
    - density_per_road_segment(DoCs, dtbounds=(datetime.datetime(1970,1,1,0,0,0), datetime.datetime(2024,1,1,0,0,0))): Calculates the density of detections per road segment within a given time range.
    - generate_gif(DoCs): Generates an animated chloropleth GIF using the given DoCs (days of coverage) data.
    - merge_days(DoCs): Merges the metadata, nearest edges, and detections of multiple days of coverage into a single DayOfCoverage object.
    - plot_density_per_road_segment(DoCs, dtbounds=(datetime.datetime(1970,1,1,0,0,0), datetime.datetime(2024,1,1,0,0,0))): Plots the density of detections per road segment within a given time range.
    """

    def __init__(self, proj_path, graphml_input, crop=False, crop_id=None, write=False):
        self.DEBUG_MODE = False
        self.WRITE_MODE = write
        self.log = setup_logger("Graph")
        self.log.setLevel(logging.INFO)
        self.log.info(f"Loading graph at path {graphml_input}")
        self.PROJ_PATH = proj_path
        self.days_of_coverage_0 = []
        self.days_of_coverage_1 = []
        self.geo = ox.io.load_graphml(graphml_input)
        self.gdf_nodes = ox.utils_graph.graph_to_gdfs(self.geo, edges=False)
        self.gdf_edges = ox.utils_graph.graph_to_gdfs(self.geo, nodes=False)

        self.crop = crop
        self.crop_id = crop_id

        if crop and crop_id is not None:
            self.log.info(f"Cropping graph to {crop_id}")
            self.log.info(f"Graph currently has {len(self.gdf_edges.index)} edges and {len(self.gdf_nodes.index)} nodes.")
            self.gdf_edges = crop_within_polygon(self.gdf_edges, h3_to_polygon(crop_id))
            self.gdf_nodes = crop_within_polygon(self.gdf_nodes, h3_to_polygon(crop_id))
            self.geo = ox.graph_from_gdfs(self.gdf_nodes, self.gdf_edges)
            self.log.info(f"Graph now has {len(self.gdf_edges.index)} edges and {len(self.gdf_nodes.index)} nodes.")


        self.gc_counter = 0

        self.log.info("Graph loaded.")

    def toggle_debug(self):
        """
        Toggles the debug mode flag.
        """
        self.DEBUG_MODE = not self.DEBUG_MODE
        self.log.info(f"Debug mode set to {self.DEBUG_MODE}.")

    def toggle_latex_font(self):
        """
        Toggles the LaTeX font flag.
        """
        plt.rcParams["text.usetex"] = not plt.rcParams["text.usetex"]
        self.log.info(f"Latex font set to {plt.rcParams['text.usetex']}.")

    def get_frames_worker(self, folder):
        """
        Returns a list of all .jpg files in the given folder.
        """
        # Check if folder exists
        if not os.path.exists(folder):
            raise ValueError("Folder does not exist.")

        # Check if folder is a directory
        if not os.path.isdir(folder):
            raise ValueError(f"Folder {folder} is not a directory.")

        # Glob 'folder' for all .jpg files
        files = glob(os.path.join(folder, "*/*.jpg"))

        return files

    def get_md_worker(self, md_csv):
        """
        Returns a pandas DataFrame object representing the metadata from the given CSV file.
        """

        # If md_csv is None, return 0
        if md_csv is None:
            return 0

        # Check if md_csv exists
        if not os.path.exists(md_csv):
            self.log.warning(f"Metadata CSV: {md_csv} does not exist.")

        # Read CSV
        df = pd.read_csv(md_csv, engine="pyarrow")

        return df

    def get_data(self, day_of_coverage, num_workers=24):
        """
        Returns a GeoDataFrame object representing the metadata for the given day of coverage.
        """

        # Check if md file has already been written to output
        if os.path.exists(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/md.csv"):
            # read md file from output
            md = pd.read_csv(
                f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/md.csv", engine="pyarrow"
            )
            md = gpd.GeoDataFrame(
                md, geometry=wkt.loads(md["geometry"]), crs=COORD_CRS
            )
            # return md
            self.log.info(
                f"Loading metadata from output for day of coverage {day_of_coverage}."
            )
            return md

        # Glob all h3-6 hexagon directories within the given day of coverage
        hex_dirs = glob(os.path.join(self.PROJ_PATH, day_of_coverage, "*"))
        # remove any non-directories
        hex_dirs = [x for x in hex_dirs if os.path.isdir(x)]

        if self.DEBUG_MODE:
            hex_dirs = hex_dirs[:1]

        self.log.info(f"Number of hex_dirs: {len(hex_dirs)}")

        # Allocate a ProcessPoolExecutor with num_workers

        # create copy of hex_dirs that points to metadata CSVs
        hex_dirs_md = hex_dirs.copy()
        # glob the csv in each hex_dir
        for i, hex_dir in enumerate(hex_dirs_md):
            # get list of csvs in hex_dir
            csvs = glob(
                os.path.join(self.PROJ_PATH, day_of_coverage, hex_dir, "*.csv")
            )
            # check if there is more than one csv
            if len(csvs) > 1:
                # raise error
                raise ValueError("More than one CSV in hex_dir.")
            # check if there is no csv
            elif len(csvs) == 0:
                # log warning
                self.log.warning(f"No CSV in hex_dir: {hex_dir}")
                # set hex_dirs_md[i] to None
                hex_dirs_md[i] = None
            else:
                # grab path of first csv
                hex_dirs_md[i] = csvs[0]

        self.log.info("Loading frames and metadata...")
        frames = Parallel(n_jobs=NUM_CORES,  timeout=999999)(
            delayed(self.get_frames_worker)(folder)
            for folder in tqdm(hex_dirs)
        )
        frames = list(chain.from_iterable(frames))
        self.log.info(f"All frames loaded. Number of frames: {len(frames)}")

        # Map get_md_counts_worker to all hex_dirs_md
        md = Parallel(n_jobs=NUM_CORES,  timeout=999999)(
            delayed(self.get_md_worker)(md_csv) for md_csv in tqdm(hex_dirs_md)
        )
        md = pd.concat(md)
        self.log.info(
            f"All metadata loaded. Number of rows in md: {len(md.index)}"
        )

        # get frame ids from frames list
        frame_ids = [x.split("/")[-1].split(".")[0] for x in frames]
        # filter md list to only include rows in frame_ids
        md = md[md[IMG_ID].isin(frame_ids)]

        # turn md into GeoDataFrame
        md = gpd.GeoDataFrame(
            md,
            geometry=gpd.points_from_xy(
                md[LONGITUDE_COL], md[LATITUDE_COL], crs=COORD_CRS
            ),
        )
        md = md.to_crs(PROJ_CRS)

        if self.WRITE_MODE:
            os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}", exist_ok=True)
            md.to_csv(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/md.csv")
            self.log.info(
                f"Wrote metadata to output for day of coverage {day_of_coverage}."
            )

        # return md for day of coverage
        return md

    def load_day_of_coverage(self, day_of_coverage):
        """
        Returns a GeoDataFrame object representing the metadata for the given day of coverage.
        """
        return self.get_data(day_of_coverage)

    def add_day_of_coverage(self, day_of_coverage, group=0):
        """
        Adds a DayOfCoverage object representing the given day of coverage to the graph.
        """
        DoC = DayOfCoverage(day_of_coverage)
        DoC.frames_data = self.get_data(day_of_coverage)
        if group == 0:
            self.days_of_coverage_0.append(DoC)
        elif group == 1:
            self.days_of_coverage_1.append(DoC)
        else:
            self.log.error(f"Invalid frame group: {group}")
        
        self.log.info(
            f"Added day of coverage {day_of_coverage} to graph group {group}, with {len(DoC.frames_data.index)} rows."
        )

    def get_day_of_coverage(self, day_of_coverage, group=0):
        """
        Returns a DayOfCoverage object representing the given day of coverage.
        """
        if group == 0:
            for DoC in self.days_of_coverage_0:
                if DoC.date == day_of_coverage:
                    return DoC
            else:
                self.log.error(
                    f"Day of coverage {day_of_coverage} not stored in graph."
                )
                return 4
        elif group == 1:
            for DoC in self.days_of_coverage_1:
                if DoC.date == day_of_coverage:
                    return DoC
            else:
                self.log.error(
                    f"Day of coverage {day_of_coverage} not stored in graph."
                )
                return 4
        else:
            self.log.error(f"Invalid frame group: {group}")
            return 4

    def nearest_road_worker(self, subset):
        """
        Returns a list of the nearest edges for each row in the given subset of metadata.
        """
        # Get nearest edge for each row in md
        nearest_edges = ox.distance.nearest_edges(
            self.geo, subset.geometry.x, subset.geometry.y, return_dist=True
        )

        return nearest_edges

    def coverage_to_nearest_road(self, day_of_coverage, group=0):
        """
        Calculates the nearest road for each frame in the given day of coverage.
        """
        # Check if day of coverage is in graph
        DoC = self.get_day_of_coverage(day_of_coverage, group)

        # Get day of coverage data
        md = self.get_day_of_coverage(day_of_coverage, group).frames_data
        md = md.to_crs(PROJ_CRS)

        # Check if nearest edges have already been written to output
        if os.path.exists(
            f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/nearest_edges.csv"
        ):
            # read nearest edges from output
            nearest_edges = pd.read_csv(
                f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/nearest_edges.csv",
                engine="pyarrow",
                index_col=IMG_ID,
            )

            
            # return nearest edges
            self.log.info(
                f"Loading nearest edges from output for day of coverage {day_of_coverage}."
            )

            if self.crop and self.crop_id is not None:
                # only keep nearest edges within crop_id 
                try:
                    
                    self.log.info(f"Nearest edges has {len(nearest_edges.index)} rows before cropping.")
                    nearest_edges = nearest_edges.merge(self.gdf_edges, how='left', left_on=['u','v','key'], right_index=True)
                    
                    nearest_edges.dropna(subset=['osmid'], inplace=True)
                    self.log.info(f"Nearest edges has {len(nearest_edges.index)} rows after cropping.")

                except KeyError as e:
                    self.log.error(f"KeyError in nearest edges for day of coverage {day_of_coverage}: {str(e)}")
                    return 4
                

            DoC.nearest_edges = nearest_edges
            return 0

        md_split = np.array_split(md, 10 * NUM_CORES)

        nearest = Parallel(n_jobs=NUM_CORES)(
            delayed(self.nearest_road_worker)(subset)
            for subset in tqdm(md_split)
        )
        nearest_edges, nearest_edges_dist = zip(*nearest)

        nearest_edges = [item for sublist in nearest_edges for item in sublist]
        nearest_edges_dist = [
            item for sublist in nearest_edges_dist for item in sublist
        ]

        nearest_edges = pd.DataFrame(nearest_edges, columns=["u", "v", "key"])
        nearest_edges["dist"] = nearest_edges_dist
        nearest_edges[IMG_ID] = md[IMG_ID].tolist()
        nearest_edges = nearest_edges.set_index(IMG_ID)


        if self.crop and self.crop_id is not None:
            # only keep nearest edges within crop_id 
                try:
                    
                    self.log.info(f"Nearest edges has {len(nearest_edges.index)} rows before cropping.")
                    nearest_edges = nearest_edges.merge(self.gdf_edges, how='left', left_on=['u','v','key'], right_index=True)
                    
                    nearest_edges.dropna(subset=['osmid'], inplace=True)
                    self.log.info(f"Nearest edges has {len(nearest_edges.index)} rows after cropping.")

                except KeyError as e:
                    self.log.error(f"KeyError in nearest edges for day of coverage {day_of_coverage}: {str(e)}")
                    return 4

        DoC.nearest_edges = nearest_edges

        if self.WRITE_MODE & (not self.crop):
            os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}", exist_ok=True)
            nearest_edges.to_csv(
                f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/nearest_edges.csv"
            )
            self.log.info(
                f"Wrote nearest edges to output for day of coverage {day_of_coverage}."
            )

        self.log.info(
            f"Added nearest edges to day of coverage {day_of_coverage}."
        )
        return 0

    def add_detections(self, day_of_coverage, group=0):
        """
        Adds a GeoDataFrame object representing the detections for the given day of coverage to the graph.
        """
        detections = pd.read_csv(
            f"{INSTALL_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/detections.csv",
            engine="pyarrow",
            index_col=0,
        )
        DoC = self.get_day_of_coverage(day_of_coverage, group)
        DoC.detections = detections
        self.log.info(
            f"Added detections to day of coverage {day_of_coverage}."
        )
        return 0

    # Starting by optimizing the gaussian_kernel function
    def gaussian_kernel_optimized(self, row, density, precomputed_neighbors):
        try:
            # Retrieve u, v from row index
            u, v = row.name

            # Use precomputed neighbors for u and v
            neighbors_u = precomputed_neighbors.get(u, [])
            neighbors_v = precomputed_neighbors.get(v, [])

            # Combine all neighbors, remove duplicates
            neighbors = set(neighbors_u + neighbors_v)

            # Check if neighbors exist in the DataFrame index
            existing_neighbors = list(neighbors.intersection(density.index))

            # Use .loc for faster DataFrame slicing with existing neighbors
            density_neighbors = density.loc[existing_neighbors]
            density_row = density.loc[(u, v)]

            # Vectorized operation to update density for neighbors (in-place operation)
            density_neighbors += density_row / len(existing_neighbors)

            # Return rows that need to be updated
            density.update(density_neighbors)

        except KeyError as e:
            self.log.error(
                f"KeyError in smoothing for edge ({u}, {v}): {str(e)}"
            )

    # Modify the smoothing function to include precomputed neighbors
    def smoothing_optimized(self, density, idx, precomputed_neighbors):
        if not density.empty:
            # Apply self.gaussian_kernel_optimized to each row of density
            tqdm.pandas(
                desc=f"Smoothing density for {idx}",
                position=idx,
                leave=False,
                total=len(density.index),
            )
            density.progress_apply(
                self.gaussian_kernel_optimized,
                axis=1,
                args=(density, precomputed_neighbors),
            )

        return density

    # Function to precompute all neighbors for each node in the graph
    def precompute_neighbors(self):
        nodes = list(self.geo.nodes())
        precomputed_neighbors = {}
        for node in tqdm(
            nodes, desc="precomputing nodes of all neighbors in G..."
        ):
            in_neighbors = list(self.geo.in_edges(node))
            out_neighbors = list(self.geo.out_edges(node))
            precomputed_neighbors[node] = in_neighbors + out_neighbors
        return precomputed_neighbors

    def smoothing(self, density, idx):
        # only smooth if density dataframe is not empty
        if not density.empty:
            # apply self.gaussian_kernel to each row of density
            tqdm.pandas(
                desc=f"Smoothing density for {idx}",
                position=idx,
                leave=True,
                total=len(density.index),
            )
            density = density.progress_apply(
                self.gaussian_kernel, axis=1, args=(density,)
            )

        return density

    def gaussian_kernel(self, row, density):
        # get u,v from row index
        u, v = row.name

        # get all neighboring edges of u
        u_in_neighbors = list(self.geo.in_edges(u))
        u_out_neighbors = list(self.geo.out_edges(u))
        u_neighbors = u_in_neighbors + u_out_neighbors

        # get all neighboring edges of v
        v_in_neighbors = list(self.geo.in_edges(v))
        v_out_neighbors = list(self.geo.out_edges(v))
        v_neighbors = v_in_neighbors + v_out_neighbors

        # combine all neighbors, remove duplicates
        neighbors = list(set(u_neighbors + v_neighbors))

        # get subset of density's edges that are in neighbors
        if density is not None:
            density_neighbors = density[density.index.isin(neighbors)]

        # get density of row
        density_row = density[density.index == (u, v)]

        # distribute density across neighbors
        density_neighbors = density_neighbors + (density_row / len(neighbors))
        self.log.debug(
            f"Distributed density of edge ({u}, {v}) to {len(density_neighbors.index)} neighbors."
        )

        # update density with density_neighbors
        density.update(density_neighbors)

        del density_neighbors
        del density_row
        del neighbors
        del u_neighbors

        return density

    def plot_edges(self):
        """
        Plots the edges of the graph.
        """
        _, ax = plt.subplots(figsize=(20, 20))
        self.gdf_edges.plot(ax=ax, color="black", linewidth=0.5)
        rID = uuid.uuid4().hex[:8]
        plt.savefig(f"{INSTALL_DIR}/{PROJECT_NAME}/plots/edges_{rID}.png")
        plt.close()

    def plot_coverage(self, day_of_coverage, group=0):
        """
        Plots the coverage of the given day of coverage.
        """
        DoC = self.get_day_of_coverage(day_of_coverage, group)
        _, ax = plt.subplots(figsize=(30, 30))

        # group by nearest edge, plot chloropleth

        coverage = (
            DoC.nearest_edges.groupby(["u", "v"])
            .size()
            .reset_index(name="counts")
        )

        coverage["binned"] = pd.qcut(
            coverage["counts"], 100, labels=False, duplicates="drop"
        )

        self.gdf_edges.plot(ax=ax, color="lightcoral", linewidth=0.5)
        self.gdf_edges.merge(coverage, on=["u", "v"], how="left").plot(
            ax=ax,
            column="binned",
            cmap="cividis",
            linewidth=0.5,
            legend=True,
            legend_kwds={
                "label": "Number of frames",
                "orientation": "horizontal",
            },
        )

        ax.set_axis_off()
        ax.margins(0)
        ax.set_title(f"Coverage for {day_of_coverage}")
        ax.title.set_size(50)

        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(
            f"{INSTALL_DIR}/{PROJECT_NAME}/plots/coverage_{rID}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    def plot_detections(self, day_of_coverage, class_id, group=0):
        """
        Plots the detections of the given class for the given day of coverage.
        """

        DoC = self.get_day_of_coverage(day_of_coverage, group)
        _, ax = plt.subplots(figsize=(30, 30), frameon=True)

        
        
        
        coverage = DoC.detections.merge(DoC.nearest_edges, left_index=True, right_on=IMG_ID, how="left")
        
        coverage = coverage.dropna(subset=['osmid'], axis=0)


        try:
            coverage = (
                coverage.groupby(["u", "v"])
                .agg({str(class_id): "mean"})
                .reset_index()
            )
        except Exception as e:
            self.log.error(f"Error grouping by nearest edge: {str(e)}")
            return 4


        # coverage['binned'] = pd.qcut(coverage[str(class_id)], 100, labels=False, duplicates='drop')

        self.gdf_edges.plot(
            ax=ax, color="lightcoral", linewidth=2, alpha=0.2
        )

        ctx.add_basemap(ax, crs=PROJ_CRS, source=ctx.providers.OpenStreetMap.Mapnik)

        try:
            self.gdf_edges.merge(coverage, on=["u", "v"], how="left").plot(
                ax=ax,
                column=str(class_id),
                cmap="cividis",
                linewidth=3,
                legend=True,
                legend_kwds={
                    "label": "Number of frames",
                    "orientation": "horizontal",
                },
            )
        except Exception as e:
            self.log.error(f"Error plotting chloropleth on {class_id}: {str(e)}")
            return 4

        ax.set_axis_off()
        ax.margins(0)
        try:
            ax.set_title(
                f"Average Num. of {cm.coco_classes[str(class_id)]}s for {day_of_coverage}"
            )
            ax.title.set_size(50)
        except Exception as e:
            self.log.error(f"Error setting title for {class_id}: {str(e)}")
            return 4
        

        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/plots/{day_of_coverage}", exist_ok=True)
        try:
            plt.savefig(
                f"{INSTALL_DIR}/{PROJECT_NAME}/plots/{day_of_coverage}/{class_id}_density_{rID}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
        except Exception as e: 
            self.log.error(f"Error saving plot for {class_id}: {str(e)}")
            return 4
        plt.close()

    def init_day_of_coverage(self, doc, group=0):
        """
        Initializes the day of coverage by adding it to the graph, finding the nearest road, and adding detections.

        Args:
            doc: The day of coverage document to initialize.

        Returns:
            None
        """
        self.add_day_of_coverage(doc, group)
        self.coverage_to_nearest_road(doc, group)
        self.add_detections(doc, group)
        try:
            self.plot_coverage(doc, group)
        except Exception as e:
            self.log.error(f"Error plotting coverage for {doc}: {str(e)}")
        try:
            self.plot_detections(doc, 2, group)
        except Exception as e:
            self.log.error(f"Error plotting detections for {doc}: {str(e)}")
        self.log.info(f"Added {doc} to graph.")


    def join_days(self, DoCs, group=0):
        """
        Concatenates the metadata, nearest edges, and detections of multiple days of coverage into a single DayOfCoverage object.

        Args:
            DoCs (list): A list of DayOfCoverage objects to be concatenated.

        Returns:
            DayOfCoverage: A new DayOfCoverage object with the concatenated metadata, nearest edges, and detections.
        """

        # Check if all days are in graph
        for doc in DoCs:
            if self.get_day_of_coverage(doc, group) == 4:
                self.log.error(f"Day of coverage {doc} not in graph.")
                return 4

        # Get nearest edges for each day of coverage
        nearest_edges = []
        for doc in DoCs:
            nearest_edges.append(self.get_day_of_coverage(doc, group).nearest_edges)

        # Concatenate nearest edges
        nearest_edges = pd.concat(nearest_edges)

        # Get detections for each day of coverage
        detections = []
        for doc in DoCs:
            detections.append(self.get_day_of_coverage(doc, group).detections)

        # Concatenate detections
        detections = pd.concat(detections)

        # Get metadata for each day of coverage
        md = []
        for doc in DoCs:
            md.append(self.get_day_of_coverage(doc, group).frames_data)

        # Concatenate metadata
        md = pd.concat(md)
        md[TIME_COL] = pd.to_datetime(md[TIME_COL], unit="ms")
        md[TIME_COL] = md[TIME_COL].dt.tz_localize("UTC").dt.tz_convert(TZ)

        return DayOfCoverage("joined", md, nearest_edges, detections)

    def coco_agg_mappings(self, operation="mean", data=pd.DataFrame()):
        """
        Returns a dictionary with class IDs as keys and the specified operation as values.

        Parameters:
        operation (str): The operation to perform on the data. Default is 'mean'.
        data (pd.DataFrame): The data to perform the operation on. Default is an empty DataFrame.

        Returns:
        dict: A dictionary with class IDs as keys and the specified operation as values.
        """
        agg_dict = {}
        for class_id in cm.coco_classes.keys():
            if str(class_id) in data.columns.astype(str):
                agg_dict[str(class_id)] = operation
        return agg_dict

    def density_per_road_segment(
        self,
        DoCs,
        dtbounds=(
            datetime.datetime(1970, 1, 1, 0, 0, 0),
            datetime.datetime(2024, 1, 1, 0, 0, 0),
        ),
    ):
        """
        Calculates the density of detections per road segment within a given time range.

        Args:
        - DoCs: A list of DayOfCounts objects.
        - dtbounds: A tuple of two datetime objects representing the start and end of the time range. Default is from 1970 to 2024.

        Returns:
        - A pandas DataFrame containing the density of detections per road segment.
        """

        # Make sure dtbounds are localized in EST
        dtbounds = (
            dtbounds[0].astimezone(tz=pytz.timezone(TZ)),
            dtbounds[1].astimezone(tz=pytz.timezone(TZ)),
        )
        data = self.join_days(DoCs, group)

        md = data.frames_data

        md = md[(md[TIME_COL] >= dtbounds[0]) & (md[TIME_COL] <= dtbounds[1])]

        detections = data.detections
        # detections.set_index(detections.iloc[:,0], inplace=True)
        detections.fillna(0, inplace=True)

        density = md.merge(detections, left_on=IMG_ID, right_index=True)
        density = density.merge(
            data.nearest_edges, left_on=IMG_ID, right_on=IMG_ID
        )

        density = (
            density.groupby(["u", "v"])
            .agg(self.coco_agg_mappings(data=detections))
            .reset_index()
        )

        os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/df/density", exist_ok=True)
        density.describe().to_csv(
            f'{INSTALL_DIR}/{PROJECT_NAME}/df/density/density_describe.csv"
        )
        density.to_csv(f'{INSTALL_DIR}/{PROJECT_NAME}/df/density/density.csv")

        del data
        del detections
        del md

        return density

    def data2density(
        self,
        plot_data,
        plot_detections,
        plot_nearest_edges,
        class_id,
        car_offset=False,
    ):
        """
        Computes the density of a given class of objects in a set of frames.

        Args:
            data (Data): the data object containing the frames and detections.
            class_id (int): the ID of the class to compute the density for.
            dtbounds (tuple): a tuple containing the start and end timestamps of the frames to consider.
            car_offset (bool, optional): whether to add 1 to the number of detections for the given class. Defaults to False.

        Returns:
            pandas.DataFrame: a DataFrame containing the density of the given class of objects in each edge of the road network.
        """

        if car_offset:
            plot_detections.loc[:, str(class_id)] = (
                plot_detections.loc[:, str(class_id)] + 1
            )

        # plot_detections.set_index(plot_detections.iloc[:,0], inplace=True)
        plot_detections = plot_detections.fillna(0)

        try:
            density = plot_data.merge(
                plot_detections, left_on=IMG_ID, right_index=True, how="left"
            )
        except Exception as e:
            self.log.error(
                f"data2density: Problem merging plot_data with plot_detections: {e}"
            )
            return

        try:
            density = density.merge(
                plot_nearest_edges, left_on=IMG_ID, right_on=IMG_ID, how="left"
            )
        except Exception as e:
            self.log.error(
                f"data2density: Problem merging density with plot_nearest_edges: {e}"
            )
            return

        try:
            
            density = density.groupby(["u", "v"]).agg(
                self.coco_agg_mappings(data=plot_detections)
            )
            
        except Exception as e:
            self.log.error(
                f"data2density: Problem grouping density by u,v: {e}"
            )
            return

        del plot_data
        del plot_detections
        del plot_nearest_edges

        return density

    def plot_density_per_road_segment(
        self,
        output_dir,
        DoCs_0,
        DoCs_1,
        delta,
        b,
        density_0,
        density_1,
        class_id,
        ax_bounds=(0, 100),
        binned=True,
        car_offset=False,
        tod_flag=False,
    ):
        """
        Plots the density of a given class of objects per road segment, using a choropleth map.

        Parameters:
        -----------
        output_dir : str
            The directory where the plot will be saved.
        DoCs : list of str
            The dates of the data to be plotted.
        delta : int
            The time interval in minutes between each data point.
        b : datetime.datetime
            The time of the data point to be plotted.
        density : pandas.DataFrame
            The ID of the class of objects to be plotted.
        ax_bounds : tuple of float, optional
            The minimum and maximum values for the colorbar.
        binned : bool, optional
            Whether to plot the data using bins or not.
        car_offset : bool, optional
            Whether to offset the data to account for car speed.
        tod_flag : bool, optional
            Whether to plot the data as a function of time of day.

        Returns:
        --------
        None
        """
        try:
            density_0 = self.gdf_edges.merge(density_0, on=["u", "v"], how="left")
            density_0 = density_0.to_crs(PROJ_CRS)

            density_1 = self.gdf_edges.merge(density_1, on=["u", "v"], how="left")
            density_1 = density_1.to_crs(PROJ_CRS)


            fig, ax = plt.subplots(figsize=(40, 40), frameon=True)

            

            self.gdf_edges.plot(
                ax=ax, color="lightcoral", linewidth=2, alpha=0.2
            )

            
            
            

            density_0["binned"] = pd.cut(
                density_0.loc[:, str(class_id)], 100, duplicates="drop"
            )
            # convert bins into upper bound
            density_0["binned"] = density_0["binned"].apply(lambda x: x.right)

            density_1["binned"] = pd.cut(
                density_1.loc[:, str(class_id)], 100, duplicates="drop"
            )
            # convert bins into upper bound
            density_1["binned"] = density_1["binned"].apply(lambda x: x.right)

            os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{output_dir}", exist_ok=True)

            density_0.to_csv(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{output_dir}/density_0_{b[0].strftime('%H:%M')}.csv")
            density_1.to_csv(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{output_dir}/density_1_{b[0].strftime('%H:%M')}.csv")

            print("Densities saved successfully.")

            norm_0, cmap_0 = self.colorbar_norm_cmap(
                plt.cm.PuRd, ax_bounds
            )

            norm_1, cmap_1 = self.colorbar_norm_cmap(
                plt.cm.YlGn, ax_bounds
            )

            if norm_0 != norm_1:
                self.log.error("plot_density_per_road_segment: Norms do not match.")
                #return

            print("Norms and cmaps generated successfully.")

            density_0.plot(
                ax=ax,
                column="binned" if binned else str(class_id),
                cmap=cmap_0,
                linewidth=7,
                alpha=0.6,
                legend=False,
            )

            density_1.plot(
                ax=ax,
                column="binned" if binned else str(class_id),
                cmap=cmap_1,
                linewidth=7,
                alpha=0.6,
                legend=False,
            )

            # create a second axes for the colorbar, on left side
            ax2 = ax.figure.add_axes([0.05, 0.05, 0.03, 0.9])

            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_0, cmap=cmap_0), cax=ax2)
            cb.ax.tick_params(labelsize=50)
            cb.ax.yaxis.set_ticks_position("right")
            cb.ax.yaxis.set_label_position("right")
            cb.ax.invert_yaxis()
            cb.ax.set_yticklabels([f"{float(x):.2f}" for x in cb.get_ticks()])

            cb.ax.title.set_fontsize(50)
            cb.ax.title.set_horizontalalignment("center")
            cb.ax.title.set_verticalalignment("bottom")

            ax.set_axis_off()
            ax.margins(0)

            ctx.add_basemap(ax, crs=PROJ_CRS, source=ctx.providers.OpenStreetMap.Mapnik)

            if tod_flag:
                try:
                    ax.set_title(
                        f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs_0[0]}-{DoCs_0[-1]} \n vs. {DoCs_1[0]}-{DoCs_1[-1]} \n {b[0].strftime('%H:%M')}"
                    )
                except Exception as e:
                    self.log.error(
                        f"plot_density_per_road_segment: Problem setting title: {e}"
                    )
                    ax.set_title(
                        f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs_0[0].strftime('%Y-%m-%d')}-{DoCs_0[-1].strftime('%Y-%m-%d')} \n {b}"
                    )

            else:
                ax.set_title(
                    f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs_0[0]}-{DoCs_0[-1]} \n {b}"
                )
            ax.title.set_size(50)
            # Move title up
            ax.title.set_position([0.5, 1.05])

            rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/plots/{output_dir}", exist_ok=True)
            self.log.info("Output dir generated successfully.")
            plt.savefig(
                f"{INSTALL_DIR}/{PROJECT_NAME}/plots/{output_dir}/{class_id}_density_{rID}.png",
                bbox_inches="tight",
                pad_inches=0,
            )

            print("plot saved")
            plt.clf()
            plt.close()
        
        except Exception as e:
            print(e)
            self.log.error(f"plot_density_per_road_segment: {e}")
            return 4

        

    def compute_density_range(
        self,
        DoCs,
        group,
        class_id,
        dtbounds,
        delta,
        car_offset=False,
        time_of_day_merge=False,
    ):
        """
        Computes the minimum and maximum density of a given class of objects in a set of days of coverage, within a given time range.

        Args:
            DoCs (list): A list of documents containing object detections.
            class_id (int): The ID of the class of objects to compute density for.
            dtbounds (tuple): A tuple of two datetime objects representing the start and end of the time range to consider.
            delta (str): A string representing the time interval to group detections by (e.g. '5min', '1H', etc.).
            car_offset (bool, optional): Whether to apply a car offset to the detections. Defaults to False.
            time_of_day_merge (bool, optional): Whether to merge detections from different days based on their time of day. Defaults to False.

        Returns:
            list: A list containing the minimum and maximum density of the given class of objects within the specified time range.

        Raises:
            ValueError: If the minimum density computed is less than 0.
        """

        if time_of_day_merge:
            dtbounds = (
                dtbounds[0].replace(year=1970, month=1, day=1),
                dtbounds[1].replace(year=1970, month=1, day=1),
            )

            data = self.merge_days(DoCs, group)
        else:
            data = self.join_days(DoCs, group)

        dtbounds = (
            dtbounds[0].astimezone(tz=pytz.timezone(TZ)),
            dtbounds[1].astimezone(tz=pytz.timezone(TZ)),
        )

        md = data.frames_data[[TIME_COL, IMG_ID]]
        md = md[(md[TIME_COL] >= dtbounds[0]) & (md[TIME_COL] <= dtbounds[1])]

        detections = data.detections
        # detections.set_index(detections.iloc[:,0], inplace=True)
        detections.fillna(0, inplace=True)

        md = md.merge(data.detections, left_on=IMG_ID, right_index=True)

        md = md.merge(data.nearest_edges, how='left', left_on=IMG_ID, right_on=IMG_ID)

        md.dropna(subset=['osmid'],inplace=True)

        md[TIME_COL] = md[TIME_COL].dt.floor(delta)
        subsets = md.groupby([TIME_COL])

        bounds = [0, 0]
        # iterate through each subset
        for _, subset in subsets:
            # compute density of subset
            density = (
                subset.groupby(["u", "v"])
                .agg(self.coco_agg_mappings(data=detections))
                .reset_index()
            )
            # get min and max of density
            min = density[str(class_id)].min()
            max = density[str(class_id)].max()
            # update bounds
            if min < bounds[0]:
                bounds[0] = min
            if max > bounds[1]:
                bounds[1] = max

        # SANITY CHECK: min should not be < 0
        if bounds[0] < 0:
            raise ValueError(f"Min density is less than 0: {bounds[0]}")

        del md
        del detections
        del subsets

        return bounds

    def colorbar_norm_cmap(self, cmap_template, bounds):
        """
        Create a custom, continuous legend with a color map and normalized bins.

        Args:
            bounds (list): A list of b boundaries.

        Returns:
            tuple: A tuple containing the normalized bins and the custom color map.
        """
        try:
            # create custom, continuous legend
            # create a color map
            cmap = cmap_template
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # create the new map
            cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

            # define the bins and normalize
            
            norm = plt.Normalize(bounds[0], bounds[-1])

            return norm, cmap
        except Exception as e:
            self.log.error(f"Error in colorbar_norm_cmap: {e}")
            return 4

    
    def merge_days(self, DoCs, group=0):
        """
        Merges the data from multiple days of coverage into a single DayOfCoverage object.

        Args:
            DoCs (list): A list of DayOfCoverage objects to merge.

        Returns:
            DayOfCoverage: A new DayOfCoverage object containing the merged data.
        """

        # Check if all days are in graph
        for doc in DoCs:
            if self.get_day_of_coverage(doc, group) == 4:
                self.log.error(f"Day of coverage {doc} not in graph.")
                return 4

        # Get nearest edges for each day of coverage
        nearest_edges = []
        for doc in DoCs:
            nearest_edges.append(self.get_day_of_coverage(doc, group).nearest_edges)
            

        # Concatenate nearest edges
        nearest_edges = pd.concat(nearest_edges)

        # Get detections for each day of coverage
        detections = []
        for doc in DoCs:
            detections.append(self.get_day_of_coverage(doc, group).detections)

        # Concatenate detections

        detections = pd.concat(detections)

        # Get metadata for each day of coveragex
        md = []
        for doc in DoCs:
            md.append(self.get_day_of_coverage(doc, group).frames_data)

        # Concatenate metadata
        md = pd.concat(md)

        md[TIME_COL] = pd.to_datetime(md[TIME_COL], unit="ms")
        md[TIME_COL] = md[TIME_COL].dt.tz_localize("UTC").dt.tz_convert(TZ)

        # Merge datetimes by setting date to 1970-01-01
        try:
            md[TIME_COL] = md[TIME_COL].apply(
                lambda x: x.replace(year=1970, month=1, day=1)
            )
        except Exception as e:
            self.log.error(f"Error in merging days of coverage: {e}")

        self.log.info(f"Stripped date from {TIME_COL} column.")
        self.log.info(
            f"{len(md)} rows in merged metadata, with time spread from {md[TIME_COL].min()} to {md[TIME_COL].max()}"
        )

        # Sort by TIME_COL
        md = md.sort_values(by=[TIME_COL])

        return DayOfCoverage("merged", md, nearest_edges, detections)

    def parallel_args_generator(self, args):
        (
            idx,
            output_dir,
            DoCs,
            delta,
            tbounds,
            plot_data,
            plot_detections,
            plot_nearest_edges,
            class_id,
            b,
            car_offset,
            tod_flag,
        ) = args

        # exceptions are handled within data2density code
        density = self.data2density(
            plot_data,
            plot_detections,
            plot_nearest_edges,
            class_id,
            car_offset=car_offset,
        )

        copy_of_neighbors = self.precomputed_neighbors.copy()

        try:
            density = self.smoothing_optimized(density, idx, copy_of_neighbors)
        except Exception as e:
            self.log.error(f"Error in smoothing for {tbounds}: {e}")
            return

        del plot_data
        del copy_of_neighbors

        tod_flag = True
        self.log.info(f"Generated density for {idx}, returning.")

        return [
            output_dir,
            DoCs,
            delta,
            tbounds,
            density,
            class_id,
            b,
            car_offset,
            tod_flag,
        ]

    def density_over_time_of_day_gif(
        self, DoCs_0, DoCs_1, tbounds, class_id, delta="60min", car_offset=False
    ):
        """
        Generates a GIF animation showing the density of a given class of objects over time of day, for a given time range.

        Args:
            DoCs (list): A list of document IDs to consider.
            tbounds (tuple): A tuple of two datetime objects representing the start and end times of the time range to consider.
            class_id (int or str): The ID of the class of objects to consider.
            delta (str, optional): The time interval to use for the density computation. Defaults to "60min".
            car_offset (bool, optional): Whether to apply a car offset to the density computation. Defaults to False.

        Returns:
            None
        """
        class_id = str(class_id)
        data_0 = self.merge_days(DoCs_0, 0)
        self.log.info(
            f"Merged days of coverage {DoCs_0} into one hypothetical of coverage."
        )
        data_1 = self.merge_days(DoCs_1, 1)
        self.log.info(
            f"Merged days of coverage {DoCs_1} into one hypothetical of coverage."
        )
    

        

        # add jan 1 1970 to tbounds
        tbounds = (
            tbounds[0].replace(year=1970, month=1, day=1),
            tbounds[1].replace(year=1970, month=1, day=1),
        )

        # localize tbounds in est
        tbounds = (
            tbounds[0].astimezone(tz=pytz.timezone(TZ)),
            tbounds[1].astimezone(tz=pytz.timezone(TZ)),
        )

        # generate time bins
        bins = pd.date_range(tbounds[0], tbounds[1], freq=delta)
        # bins = bins.tz_localize(TZ)
        self.log.info(f"Generated {len(bins)} time bins.")

        output_dir = uuid.uuid4().hex[:8]
        self.log.info(f"Output directory: {output_dir}")

        # generate cb norm and cmap
        bounds_0 = self.compute_density_range(
            DoCs_0,
            0,
            class_id,
            tbounds,
            delta,
            car_offset=car_offset,
            time_of_day_merge=True,
        )

        bounds_1 = self.compute_density_range(
            DoCs_1,
            1,
            class_id,
            tbounds,
            delta,
            car_offset=car_offset,
            time_of_day_merge=True,
        )

        bounds = [min(bounds_0[0], bounds_1[0]), max(bounds_0[1], bounds_1[1])]
        self.log.info(
            f"Computed overall density range for {DoCs_0} and {DoCs_1}, lower bound: {bounds[0]}, upper bound: {bounds[1]}"
        )

        args_0 = []
        args_1 = []

        self.precomputed_neighbors = self.precompute_neighbors()

        first_it_args_0 = []
        first_it_args_1 = []

        for idx, b in tqdm(enumerate(bins), total=len(bins)):
            # Sliding window
            if idx < 6:
                tbounds = (bins[0], bins[idx])
            else:
                tbounds = (bins[idx - 6], bins[idx])
            # dtbounds = (b, b + pd.Timedelta(delta))

            try:
                plot_data_0 = data_0.frames_data[
                    (data_0.frames_data[TIME_COL] >= tbounds[0])
                    & (data_0.frames_data[TIME_COL] <= tbounds[1])
                ]

                # get detections rows that correspond to frames in plot_data

                plot_detections_0 = data_0.detections[
                    data_0.detections.index.isin(plot_data_0[IMG_ID])
                ]
                plot_nearest_edges_0 = data_0.nearest_edges[
                    data_0.nearest_edges.index.isin(plot_data_0[IMG_ID])
                ]

            

                if (len(plot_data_0) != len(plot_nearest_edges_0)) and not self.crop:
                    self.log.warning(
                        f"Lengths of plot data_0 ({len(plot_data_0)}) and plot nearest edges ({len(plot_nearest_edges_0)}) do not match for {tbounds}."
                    )
                

                first_it_args_0.append(
                    [
                        idx,
                        output_dir,
                        DoCs_0,
                        delta,
                        tbounds,
                        plot_data_0,
                        plot_detections_0,
                        plot_nearest_edges_0,
                        class_id,
                        bounds,
                        car_offset,
                        True,
                    ]
                )
            except Exception as e:
                self.log.error(
                    f"Error in first it args generation for {tbounds}: {e}"
                )
                continue

            try:
                plot_data_1 = data_1.frames_data[
                    (data_1.frames_data[TIME_COL] >= tbounds[0])
                    & (data_1.frames_data[TIME_COL] <= tbounds[1])
                ]

                # get detections rows that correspond to frames in plot_data

                plot_detections_1 = data_1.detections[
                    data_1.detections.index.isin(plot_data_1[IMG_ID])
                ]
                plot_nearest_edges_1 = data_1.nearest_edges[
                    data_1.nearest_edges.index.isin(plot_data_1[IMG_ID])
                ]

            

                if (len(plot_data_1) != len(plot_nearest_edges_1)) and not self.crop:
                    self.log.warning(
                        f"Lengths of plot data_1 ({len(plot_data_1)}) and plot nearest edges ({len(plot_nearest_edges_1)}) do not match for {tbounds}."
                    )

                first_it_args_1.append(
                    [
                        idx,
                        output_dir,
                        DoCs_1,
                        delta,
                        tbounds,
                        plot_data_1,
                        plot_detections_1,
                        plot_nearest_edges_1,
                        class_id,
                        bounds,
                        car_offset,
                        True,
                    ]
                )

            except Exception as e:
                self.log.error(
                    f"Error in first it args generation for {tbounds}: {e}"
                )
                continue

        args_0 = Parallel(n_jobs=3, timeout=999999)(
            delayed(self.parallel_args_generator)(arg)
            for arg in tqdm(
                first_it_args_0,
                desc="Generating arguments for density plotting (group 0).",
            )
        )

        args_1 = Parallel(n_jobs=3, timeout=999999)(
            delayed(self.parallel_args_generator)(arg)
            for arg in tqdm(
                first_it_args_1,
                desc="Generating arguments for density plotting (group 1).",
            )
        )

        self.log.info(
            f"Generated {len(args_0)} arguments for plotting density per road segment."
        )

        self.log.info(f"Plotting density per road segment for {DoCs_0}.")

        try:
            Parallel(n_jobs=NUM_CORES,  timeout=999999)(
                delayed(self.plot_density_per_road_segment_parallel)(arg0, arg1)
                for arg0, arg1 in
                    zip(args_0, args_1)  
                )
            
        except Exception as e:
            self.log.error(f"Error in plotting density per road segment: {e}")
            return 4
        self.log.info(f"Finished plotting density per road segment, frames in {output_dir}.")
        # generate gif
        self.generate_gif(output_dir, class_id, DoCs_0, delta)

    def plot_density_per_road_segment_parallel(self, args_0 , args_1):
        """
        Plots the density per road segment in parallel.

        Args:
            args (tuple): A tuple containing the following arguments:
                output_dir (str): The output directory for the plot.
                DoCs (list): A list of days of coverage to plot.
                delta (float): The delta value for the plot.
                b (str): The b for the plot.
                density (str): The density for the plot.
                class_id (str): The class ID for the plot.
                bounds (list): A list of bounds for the plot.
                car_offset (bool): Flag to offset car counts by 1. (Default: False)
                tod_flag (bool): Flag to merge on time of day.

        Returns:
            int: Returns 4 if there is an error, otherwise returns None.
        """
        try:
            (
                output_dir,
                DoCs_0,
                delta,
                b,
                density_0,
                class_id,
                bounds,
                car_offset,
                tod_flag,
            ) = args_0

            (
                output_dir,
                DoCs_1,
                delta,
                b,
                density_1,
                class_id,
                bounds,
                car_offset,
                tod_flag,
            ) = args_1
        except Exception as e: 
            self.log.error(f"Error unpacking args: {args_0} - {e}")
            return 4

        try:
            self.plot_density_per_road_segment(
                output_dir,
                DoCs_0,
                DoCs_1,
                delta,
                b,
                density_0,
                density_1,
                class_id,
                bounds,
                car_offset=car_offset,
                tod_flag=tod_flag,
            )

        except Exception as e:
            self.log.error(f"Error in plot density per road segment - {b}: {e}")
            return 4

    def generate_gif(
        self,
        frames_dir,
        class_id,
        days_of_coverage,
        delta,
        fps=24,
        duration=42,
        loop=0,
    ):
        """
        Generate a GIF from a directory of frames.

        Args:
            frames_dir (str): The directory containing the frames to be used in the GIF.
            class_id (str): The ID of the class being analyzed.
            days_of_coverage (int): The number of days of data being analyzed.
            delta (int): The time delta between each frame.
            fps (int, optional): The frames per second of the GIF. Defaults to 24.
            duration (int, optional): The duration of the GIF in seconds. Defaults to 42.
            loop (int, optional): The number of times the GIF should loop. Defaults to 0.
        """

        os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/gifs", exist_ok=True)

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if not frames_dir:
            self.log.error("No frames directory found.")
            return

        fp_out = f"{INSTALL_DIR}/{PROJECT_NAME}/gifs/{class_id}_{now}_{frames_dir}.gif"

        with contextlib.ExitStack() as stack:
            # lazily load images
            imgs = (
                stack.enter_context(Image.open(f))
                for f in sorted(glob(f"{INSTALL_DIR}/{PROJECT_NAME}/plots/{frames_dir}/*.png"))
            )

            # extract  first image from iterator
            img = next(imgs)

            img.save(
                fp=fp_out,
                format="GIF",
                append_images=imgs,
                save_all=True,
                duration=60,
                loop=0,
            )
