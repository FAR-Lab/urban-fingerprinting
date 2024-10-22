# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 10/02/2023

# This script is used to take the roads of a graph G, and a set of annotated dashcam frames F, and generate commodity densities for each road in G.

import uu
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

import user.params.io as urbanECG_io 
from importlib import reload
reload(urbanECG_io)

from user.params.io import INSTALL_DIR, PROJECT_NAME, OUTPUT_DIR, TOP_LEVEL_FRAMES_DIR

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
    def __init__(self, proj_path, graphml_input, crop=False, crop_id=None, write=False):
        self.DEBUG_MODE = False
        self.WRITE_MODE = write
        self.log = setup_logger("Graph")
        self.log.setLevel(logging.INFO)
        self.log.info(f"Loading graph at path {graphml_input}")
        self.PROJ_PATH = TOP_LEVEL_FRAMES_DIR
        self.days_of_coverage = []
        self.geo = ox.io.load_graphml(graphml_input)
        self.gdf_nodes = ox.utils_graph.graph_to_gdfs(self.geo, edges=False)
        self.gdf_edges = ox.utils_graph.graph_to_gdfs(self.geo, nodes=False)

        self.crop = crop
        self.crop_id = crop_id

        self.SMOOTH = False

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

    def _get_frames_worker(self, folder):
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

    def _get_md_worker(self, md_csv):
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

    def _get_data(self, day_of_coverage, num_workers=24):
        """
        Returns a GeoDataFrame object representing the metadata for the given day of coverage.
        """

        # Check if md file has already been written to output
        if os.path.exists(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/md.csv"):
            # read md file from output
            md = pd.read_csv(
                f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/md.csv", engine="pyarrow"
            )
            md = gpd.GeoDataFrame(
                md, geometry=wkt.loads(md["geometry"]), crs=PROJ_CRS
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
                os.path.join(self.PROJ_PATH, day_of_coverage, hex_dir, "metadata.csv")
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
        frames = Parallel(n_jobs=NUM_CORES,  timeout=99999)(
            delayed(self._get_frames_worker)(folder)
            for folder in tqdm(hex_dirs)
        )
        frames = list(chain.from_iterable(frames))
        self.log.info(f"All frames loaded. Number of frames: {len(frames)}")

        # Map get_md_counts_worker to all hex_dirs_md
        md = Parallel(n_jobs=NUM_CORES, timeout=99999)(
            delayed(self._get_md_worker)(md_csv) for md_csv in tqdm(hex_dirs_md)
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
            os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}", exist_ok=True)
            md.to_csv(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/md.csv")
            self.log.info(
                f"Wrote metadata to output for day of coverage {day_of_coverage}."
            )
        

        # return md for day of coverage
        return md

    def _add_day_of_coverage(self, day_of_coverage):
        """
        Adds a DayOfCoverage object representing the given day of coverage to the graph.
        """
        DoC = DayOfCoverage(day_of_coverage)
        DoC.frames_data = self._get_data(day_of_coverage)
        self.days_of_coverage.append(DoC)
        self.log.info(
            f"Added day of coverage {day_of_coverage} to graph, with {len(DoC.frames_data.index)} rows."
        )

    def get_day_of_coverage(self, day_of_coverage):
        """
        Returns a DayOfCoverage object representing the given day of coverage.
        """
        for DoC in self.days_of_coverage:
            if DoC.date == day_of_coverage:
                return DoC
        else:
            self.log.error(
                f"Day of coverage {day_of_coverage} not stored in graph."
            )
            return 4

    def _nearest_road_worker(self, subset):
        """
        Returns a list of the nearest edges for each row in the given subset of metadata.
        """
        # Get nearest edge for each row in md
        nearest_edges = ox.distance.nearest_edges(
            self.geo, subset.geometry.x, subset.geometry.y, return_dist=True
        )

        return nearest_edges

    def _coverage_to_nearest_road(self, day_of_coverage):
        """
        Calculates the nearest road for each frame in the given day of coverage.
        """
        # Check if day of coverage is in graph
        DoC = self.get_day_of_coverage(day_of_coverage)

        # Get day of coverage data
        md = self.get_day_of_coverage(day_of_coverage).frames_data
        md = md.to_crs(PROJ_CRS)

        # Check if nearest edges have already been written to output
        if os.path.exists(
            f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/nearest_edges.csv"
        ):
            # read nearest edges from output
            nearest_edges = pd.read_csv(
                f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/nearest_edges.csv",
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

        nearest = Parallel(n_jobs=NUM_CORES, timeout=99999)(
            delayed(self._nearest_road_worker)(subset)
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
            os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}", exist_ok=True)
            nearest_edges.to_csv(
                f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/nearest_edges.csv"
            )
            self.log.info(
                f"Wrote nearest edges to output for day of coverage {day_of_coverage}."
            )

        self.log.info(
            f"Added nearest edges to day of coverage {day_of_coverage}."
        )
        return 0

    def _add_detections(self, day_of_coverage):
        """
        Adds a GeoDataFrame object representing the detections for the given day of coverage to the graph.
        """
        detections = pd.read_csv(
            f"{OUTPUT_DIR}/{PROJECT_NAME}/df/{day_of_coverage}/detections.csv",
            engine="pyarrow",
            index_col=0,
        )
        DoC = self.get_day_of_coverage(day_of_coverage)
        DoC.detections = detections
        self.log.info(
            f"Added detections to day of coverage {day_of_coverage}."
        )
        return 0

    # Starting by optimizing the gaussian_kernel function
    def _gaussian_kernel_optimized(self, row, density, precomputed_neighbors):
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
    def smoothing(self, density, idx, precomputed_neighbors):
        if not density.empty:
            # Apply self.gaussian_kernel_optimized to each row of density
            tqdm.pandas(
                desc=f"Smoothing density for {idx}",
                position=idx,
                leave=True,
                total=len(density.index),
            )
            density.progress_apply(
                self._gaussian_kernel_optimized,
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

    def plot_edges(self):
        """
        Plots the edges of the graph.
        """
        _, ax = plt.subplots(figsize=(20, 20))
        self.gdf_edges.plot(ax=ax, color="black", linewidth=0.5)
        rID = uuid.uuid4().hex[:8]
        plt.savefig(f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/edges_{rID}.png")
        plt.close()

    def plot_coverage(self, day_of_coverage):
        """
        Plots the coverage of the given day of coverage.
        """
        DoC = self.get_day_of_coverage(day_of_coverage)
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

        # set colorbar font size
        cb = ax.get_figure().get_axes()[1]
        cb.tick_params(labelsize=35)

        # set colorbar label size
        cb.set_ylabel(cb.get_ylabel(), fontsize=40)

        # set colorbar title size
        cb.set_title(cb.get_title(), fontsize=40)

        ax.set_axis_off()
        ax.margins(0)
        ax.set_title(f"Coverage for {day_of_coverage}")
        ax.title.set_size(50)

        # add padding between title and plot
        plt.subplots_adjust(top=0.95)
        

        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(
            f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/coverage_{rID}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    def plot_detections(self, day_of_coverage, class_id):
        """
        Plots the detections of the given class for the given day of coverage.
        """

        DoC = self.get_day_of_coverage(day_of_coverage)
        _, ax = plt.subplots(figsize=(30, 30), frameon=True)

        
        
        
        coverage = DoC.detections.merge(DoC.nearest_edges, left_index=True, right_on=IMG_ID, how="left")
        
        #try:
        #    coverage = coverage.dropna(subset=['osmid'], axis=0)
        #except KeyError as e:
        #    self.log.error(f"KeyError in plotting detections for {day_of_coverage}: {str(e)}")
            


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
        os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/{day_of_coverage}", exist_ok=True)
        try:
            plt.savefig(
                f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/{day_of_coverage}/{class_id}_density_{rID}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
        except Exception as e: 
            self.log.error(f"Error saving plot for {class_id}: {str(e)}")
            return 4
        plt.close()

    def init_day_of_coverage(self, doc):
        """
        Initializes the day of coverage by adding it to the graph, finding the nearest road, and adding detections.

        Args:
            doc: The day of coverage document to initialize.

        Returns:
            None
        """
        self._add_day_of_coverage(doc)
        self._coverage_to_nearest_road(doc)
        self._add_detections(doc)
        try:
            self.plot_coverage(doc)
        except Exception as e:
            self.log.error(f"Error plotting coverage for {doc}: {str(e)}")
        try:
            self.plot_detections(doc, 2)
        except Exception as e:
            self.log.error(f"Error plotting detections for {doc}: {str(e)}")
        self.log.info(f"Initialized {doc} data in graph.")

    def generate_gif(self, DoCs):
        """
        Generates an animated chloropleth GIF using the given DoCs (days of coverage) data.

        Args:
        - self: the Graph object
        - DoCs: the days of coverage data

        Returns:
        - None
        """
        gif = am.AnimatedChloropleth(self, DoCs)
        gif.set_roads(self.geo)
        gif.generate_frames(TIME_COL, "2", "10min", "b", car_offset=True)
        gif.generate_gif()

    def join_days(self, DoCs):
        """
        Concatenates the metadata, nearest edges, and detections of multiple days of coverage into a single DayOfCoverage object.

        Args:
            DoCs (list): A list of DayOfCoverage objects to be concatenated.

        Returns:
            DayOfCoverage: A new DayOfCoverage object with the concatenated metadata, nearest edges, and detections.
        """

        # Check if all days are in graph
        for doc in DoCs:
            if self.get_day_of_coverage(doc) == 4:
                self.log.error(f"Day of coverage {doc} not in graph.")
                return 4

        # Get nearest edges for each day of coverage
        nearest_edges = []
        for doc in DoCs:
            try:
                nearest_edges.append(self.get_day_of_coverage(doc).nearest_edges)
            except Exception as e:
                self.log.error(
                    f"Error getting nearest edges for day of coverage {doc}: {str(e)}"
                )
                continue

        # Concatenate nearest edges
        
        try:
            nearest_edges = pd.concat(nearest_edges)
        except TypeError as e: 
            self.log.error(f"Error concatenating nearest edges: {str(e)}")

        # Get detections for each day of coverage
        detections = []
        for doc in DoCs:
            try:
                detections.append(self.get_day_of_coverage(doc).detections)
            except Exception as e:
                self.log.error(
                    f"Error getting detections for day of coverage {doc}: {str(e)}"
                )
                continue
            
        # Concatenate detections
        try:
            detections = pd.concat(detections)
        except TypeError as e:
            self.log.error(f"Error concatenating detections: {str(e)}")

        

        # Get metadata for each day of coverage
        md = []
        for doc in DoCs:
            try:
                md.append(self.get_day_of_coverage(doc).frames_data)
            except Exception as e:
                self.log.error(
                    f"Error getting metadata for day of coverage {doc}: {str(e)}"
                )
                continue
        
        
        try:
            # Concatenate metadata
            md = pd.concat(md)
            md[TIME_COL] = pd.to_datetime(md[TIME_COL], unit="ms")
            md[TIME_COL] = md[TIME_COL].dt.tz_localize("UTC").dt.tz_convert(TZ)
        except TypeError as e:
            self.log.error(f"Error concatenating metadata: {str(e)}")
        

        return DayOfCoverage("joined", md, nearest_edges, detections)

    def _coco_agg_mappings(self, operation="mean", data=pd.DataFrame()):
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

    def _density_per_road_segment(
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
        data = self.join_days(DoCs)

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
            .agg(self._coco_agg_mappings(data=detections))
            .reset_index()
        )

        os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density", exist_ok=True)
        density.describe().to_csv(
            f'{OUTPUT_DIR}/{PROJECT_NAME}/df/density/density_describe.csv'
        )
        density.to_csv(f'{OUTPUT_DIR}/{PROJECT_NAME}/df/density/density.csv')

        del data
        del detections
        del md

        return density

    def _data2density(
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
                self._coco_agg_mappings(data=plot_detections)
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

    def _plot_density_per_road_segment(
        self,
        output_dir,
        DoCs,
        delta,
        b,
        density,
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

        density = self.gdf_edges.merge(density, on=["u", "v"], how="left")
        density = density.to_crs(PROJ_CRS)

        # write density to csv 
        try: 
            os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/plot_density", exist_ok=True)
            os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/plot_density/{output_dir}", exist_ok=True)
            print(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/plot_density/{output_dir}/{class_id}_density_{b}.csv")
            density.to_csv(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/plot_density/{output_dir}/{class_id}_density_{b}.csv")
        except Exception as e: 
            self.log.error(f"Error writing density to csv: {str(e)}")


        fig, ax = plt.subplots(figsize=(40, 40), frameon=True)

        

        self.gdf_edges.plot(
            ax=ax, color="lightcoral", linewidth=2, alpha=0.2
        )

        
        
        

        density["binned"] = pd.cut(
            density.loc[:, str(class_id)], 100, duplicates="drop"
        )
        # convert bins into upper bound
        density["binned"] = density["binned"].apply(lambda x: x.right)

        norm, cmap = self._colorbar_norm_cmap(ax_bounds)

        density.plot(
            ax=ax,
            column="binned" if binned else str(class_id),
            cmap=cmap,
            linewidth=3,
            legend=False,
        )

        # create a second axes for the colorbar, on left side
        ax2 = ax.figure.add_axes([0.05, 0.05, 0.03, 0.9])

        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2)
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
                    f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs[0]}-{DoCs[-1]} \n {b[0].strftime('%H:%M')}"
                )
            except Exception as e:
                self.log.error(
                    f"plot_density_per_road_segment: Problem setting title: {e}"
                )
                ax.set_title(
                    f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs[0].strftime('%Y-%m-%d')}-{DoCs[-1].strftime('%Y-%m-%d')} \n {b}"
                )

        else:
            ax.set_title(
                f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs[0]}-{DoCs[-1]} \n {b}"
            )
        ax.title.set_size(50)
        # Move title up
        ax.title.set_position([0.5, 1.05])

        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/{output_dir}", exist_ok=True)
        plt.savefig(
            f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/{output_dir}/{class_id}_density_{rID}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
        plt.close()

        del density

    def _compute_density_range(
        self,
        DoCs,
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

            data = self.merge_days(DoCs)
        else:
            data = self.join_days(DoCs)

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

        try:
            md.dropna(subset=['osmid'],inplace=True)
        except KeyError as e:
            self.log.error(f"KeyError in computing density range: {str(e)}")
            

        md[TIME_COL] = md[TIME_COL].dt.floor(delta)
        subsets = md.groupby([TIME_COL])

        bounds = [0, 0]
        # iterate through each subset
        for _, subset in subsets:
            # compute density of subset
            density = (
                subset.groupby(["u", "v"])
                .agg(self._coco_agg_mappings(data=detections))
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

    def _colorbar_norm_cmap(self, bounds):
        """
        Create a custom, continuous legend with a color map and normalized bins.

        Args:
            bounds (list): A list of b boundaries.

        Returns:
            tuple: A tuple containing the normalized bins and the custom color map.
        """

        # create custom, continuous legend
        # create a color map
        cmap = plt.cm.RdPu
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list("Custom cmap", cmaplist, cmap.N)

        # define the bins and normalize
        bounds
        norm = plt.Normalize(bounds[0], bounds[-1])

        return norm, cmap

    def density_over_datetime_gif(
        self, DoCs, dtbounds, class_id, delta="60min", car_offset=False
    ):
        """
        Generates a density-over-datetime GIF animation for a given object class.

        Args:
            DoCs (list): A list of documents containing the data to be plotted.
            dtbounds (tuple): A tuple with two elements representing the datetime bounds of the plot.
            class_id (int or str): The ID of the class of road segments to be plotted.
            delta (str, optional): The time interval between each frame of the animation. Defaults to "60min".
            car_offset (bool, optional): Whether to apply a car offset to the data. Defaults to False.

        Returns:
            None
        """

        class_id = str(class_id)
        data = self.join_days(DoCs)

        # generate time bins
        bins = pd.date_range(dtbounds[0], dtbounds[1], freq=delta)
        bins = bins.tz_localize(TZ)

        output_dir = f"{DoCs[0]}_{DoCs[-1]}_{class_id}_{uuid.uuid4().hex[:8]}"

        # generate cb norm and cmap
        bounds = self._compute_density_range(
            DoCs, class_id, dtbounds, delta, car_offset=car_offset
        )

        args = []
        for idx, b in enumerate(bins):
            # Sliding window
            if idx < 6:
                dtbounds = (bins[0], bins[idx])
            else:
                dtbounds = (bins[idx - 6], bins[idx])
            # dtbounds = (b, b + pd.Timedelta(delta))
            plot_data = data.frames_data[
                (data.frames_data[TIME_COL] >= dtbounds[0])
                & (data.frames_data[TIME_COL] <= dtbounds[1])
            ].copy()
            plot_detections = data.detections[
                data.detections.index.isin(plot_data[IMG_ID])
            ].copy()
            plot_nearest_edges = data.nearest_edges[
                data.nearest_edges.index.isin(plot_data[IMG_ID])
            ].copy()
            density = self._data2density(
                plot_data,
                plot_detections,
                plot_nearest_edges,
                class_id,
                car_offset=car_offset,
            )

            # save density to csv 
            try:
                os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density/{output_dir}", exist_ok=True)
                density.to_csv(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density/{output_dir}/{idx}_{dtbounds[0]}_{dtbounds[1]}.csv")
                self.log.info(f"Saved density to csv for {idx}_{dtbounds[0]}_{dtbounds[1]}.")
            except Exception as e:
                self.log.error(f"Error saving density to csv: {str(e)}")

            del plot_data
            tod_flag = False
            args.append(
                (
                    output_dir,
                    DoCs,
                    delta,
                    b,
                    density,
                    class_id,
                    bounds,
                    car_offset,
                    tod_flag,
                )
            )

        Parallel(n_jobs=NUM_CORES, timeout=99999)(
            delayed(self._plot_density_per_road_segment_parallel)(arg)
            for arg in tqdm(
                args, desc="Generating density-over-datetime GIF frames."
            )
        )

        self.log.info("Generating GIF with id {output_dir}...")

        # generate gif
        self._compose_gif(output_dir, class_id, DoCs, delta)

    def merge_days(self, DoCs):
        """
        Merges the data from multiple days of coverage into a single DayOfCoverage object.

        Args:
            DoCs (list): A list of DayOfCoverage objects to merge.

        Returns:
            DayOfCoverage: A new DayOfCoverage object containing the merged data.
        """

        # Check if all days are in graph
        for doc in DoCs:
            if self.get_day_of_coverage(doc) == 4:
                self.log.error(f"Day of coverage {doc} not in graph.")
                return 4

        # Get nearest edges for each day of coverage
        nearest_edges = []
        for doc in DoCs:
            nearest_edges.append(self.get_day_of_coverage(doc).nearest_edges)
            

        # Concatenate nearest edges
        nearest_edges = pd.concat(nearest_edges)

        # Get detections for each day of coverage
        detections = []
        for doc in DoCs:
            detections.append(self.get_day_of_coverage(doc).detections)

        # Concatenate detections

        detections = pd.concat(detections)

        # Get metadata for each day of coveragex
        md = []
        for doc in DoCs:
            md.append(self.get_day_of_coverage(doc).frames_data)

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

    def _parallel_plot_args_generator(self, args):
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
        density = self._data2density(
            plot_data,
            plot_detections,
            plot_nearest_edges,
            class_id,
            car_offset=car_offset,
        )

        # write to csv 
        try:
            os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density/{output_dir}", exist_ok=True)
            density.to_csv(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density/{output_dir}/{idx}_{tbounds[0]}_{tbounds[1]}.csv")
            
            self.log.info(f"Saved density to csv for {idx}_{tbounds[0]}_{tbounds[1]}.")
        except Exception as e:
            print(f"Error saving density to csv: {str(e)}")
            self.log.error(f"Error saving density to csv: {str(e)}")

        copy_of_neighbors = self.precomputed_neighbors.copy()

        if self.SMOOTH:

            self.log.info(f"Smoothing density for {idx}...")

            try:
                density = self.smoothing(density, idx, copy_of_neighbors)
            except Exception as e:
                self.log.error(f"Error in smoothing for {tbounds}: {e}")
                return

            # write smoothed density to csv
            try:
                os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density/{output_dir}", exist_ok=True)
                density.to_csv(f"{OUTPUT_DIR}/{PROJECT_NAME}/df/density/{output_dir}/{idx}_{tbounds[0]}_{tbounds[1]}_smoothed.csv")
                self.log.info(f"Saved smoothed density to csv for {idx}_{tbounds[0]}_{tbounds[1]}.")
            except Exception as e:
                self.log.error(f"Error saving smoothed density to csv: {str(e)}")
        
        else: 
            self.log.info(f"Skipping smoothing for {idx}...")

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
        self, DoCs, tbounds, class_id, delta="60min", car_offset=False
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
        data = self.merge_days(DoCs)
        self.log.info(
            f"Merged days of coverage {DoCs} into one hypothetical of coverage."
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

        # output_dir = first DoC_last DoC_UUID
        output_dir = f"{DoCs[0]}_{DoCs[-1]}_{class_id}_{uuid.uuid4().hex[:8]}"

        # generate cb norm and cmap
        bounds = self._compute_density_range(
            DoCs,
            class_id,
            tbounds,
            delta,
            car_offset=car_offset,
            time_of_day_merge=True,
        )
        self.log.info(
            f"Computed overall density range for {DoCs}, lower bound: {bounds[0]}, upper bound: {bounds[1]}"
        )

        args = []

        self.precomputed_neighbors = self.precompute_neighbors()

        first_it_args = []
        for idx, b in tqdm(enumerate(bins), total=len(bins)):
            # Sliding window
            if idx < 2:
                tbounds = (bins[0], bins[idx])
            else:
                tbounds = (bins[idx - 2], bins[idx])
            # dtbounds = (b, b + pd.Timedelta(delta))
            try:
                plot_data = data.frames_data[
                    (data.frames_data[TIME_COL] >= tbounds[0])
                    & (data.frames_data[TIME_COL] <= tbounds[1])
                ]

                # get detections rows that correspond to frames in plot_data

                plot_detections = data.detections[
                    data.detections.index.isin(plot_data[IMG_ID])
                ]
                plot_nearest_edges = data.nearest_edges[
                    data.nearest_edges.index.isin(plot_data[IMG_ID])
                ]

            

                if (len(plot_data) != len(plot_nearest_edges)) and not self.crop:
                    self.log.warning(
                        f"Lengths of plot data ({len(plot_data)}) and plot nearest edges ({len(plot_nearest_edges)}) do not match for {tbounds}."
                    )

                first_it_args.append(
                    [
                        idx,
                        output_dir,
                        DoCs,
                        delta,
                        tbounds,
                        plot_data,
                        plot_detections,
                        plot_nearest_edges,
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

        args = Parallel(n_jobs=3, timeout=99999)(
            delayed(self._parallel_plot_args_generator)(arg)
            for arg in tqdm(
                first_it_args,
                desc="Generating arguments for density plotting.",
            )
        )

        self.log.info(
            f"Generated {len(args)} arguments for plotting density per road segment."
        )

        self.log.info(f"Plotting density per road segment for {DoCs}.")
        Parallel(n_jobs=NUM_CORES, timeout=99999)(
            delayed(self._plot_density_per_road_segment_parallel)(arg)
            for arg in tqdm(args, desc="Plotting density per road segment.")
        )

        # generate gif
        self._compose_gif(output_dir, class_id, DoCs, delta)

    def _plot_density_per_road_segment_parallel(self, args):
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
        (
            output_dir,
            DoCs,
            delta,
            b,
            density,
            class_id,
            bounds,
            car_offset,
            tod_flag,
        ) = args
        try:
            self._plot_density_per_road_segment(
                output_dir,
                DoCs,
                delta,
                b,
                density,
                class_id,
                bounds,
                car_offset=car_offset,
                tod_flag=tod_flag,
            )

        except Exception as e:
            self.log.error(f"Error in plot density per road segment - {b}: {e}")
            return 4

    def _compose_gif(
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

        os.makedirs(f"{OUTPUT_DIR}/{PROJECT_NAME}/gifs", exist_ok=True)

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if not frames_dir:
            self.log.error("No frames directory found.")
            return

        fp_out = f"{OUTPUT_DIR}/{PROJECT_NAME}/gifs/{class_id}_{frames_dir}_{now}.gif"

        with contextlib.ExitStack() as stack:
            # lazily load images
            imgs = (
                stack.enter_context(Image.open(f))
                for f in sorted(glob(f"{OUTPUT_DIR}/{PROJECT_NAME}/plots/{frames_dir}/*.png"))
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
