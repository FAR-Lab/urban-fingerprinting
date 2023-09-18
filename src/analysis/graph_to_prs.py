# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot 
# Last Edited: 09/15/2023 

# This script is used to take the roads of a graph G, and a set of annotated dashcam frames F, and generate commodity densities for each road in G.

import pandas as pd
import numpy as np
import geopandas as gpd 
import osmnx as ox 
import os
from glob import glob 
from concurrent.futures import ProcessPoolExecutor
from fire import Fire
from tqdm import tqdm

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)


# Constant for number of workers to use in parallel, should be equal to number of cores on machine
NUM_CORES = os.getenv("SLURM_CPUS_ON_NODE")
# Check if none
if NUM_CORES is None:
    # Set to 8
    NUM_CORES = 24
else: 
    NUM_CORES *= 2


class DayOfCoverage:
    """
    A class to represent a day of coverage from the Nexar dataset. 

    ...

    Attributes 
    ----------
    date : str
        The date of the day of coverage, in the format YYYY-MM-DD
    frames_data : pd.DataFrame
        A pandas dataframe containing the metadata for all frames in the day of coverage.
    nearest_edges : pd.DataFrame
        A pandas dataframe containing the nearest edges to each frame in frames_data.
    nearest_edges_dist : pd.DataFrame
        A pandas dataframe containing the distance to the nearest edge for each frame in frames_data.
    
    Methods
    -------
    None
    """
    def __init__(self, day_of_coverage):
        self.date = day_of_coverage 
        self.frames_data = [] 
        self.nearest_edges = [] 
        self.nearest_edges_dist = []


class G:
    """
    A class to represent a graph G, and a set of annotated dashcam frames F, and generate commodity densities for each road in G.

    ...

    Attributes
    ----------
    PROJ_PATH : str
        The path to the root of the Nexar dataset.
    days_of_coverage : list
        A list of DayOfCoverage objects, each representing a day of coverage from the Nexar dataset.
    geo : ox.graph
        The graph of the city, loaded from a graphml file.
    gdf_nodes : gpd.GeoDataFrame
        A GeoDataFrame containing the nodes of the graph.
    gdf_edges : gpd.GeoDataFrame
        A GeoDataFrame containing the edges of the graph.
    
    Methods
    -------
    get_frames_worker(folder)
        A worker function for get_data, which loads all frames in a given folder.
    get_md_worker(md_csv)
        A worker function for get_data, which loads the metadata for a given folder.
    get_data(day_of_coverage, num_workers=8)
        Loads the frames and metadata for a day of coverage from the Nexar dataset.
    load_day_of_coverage(day_of_coverage)
        Loads the metadata for a day of coverage from the Nexar dataset.
    add_day_of_coverage(day_of_coverage)
        Adds a day of coverage to the graph.
    get_day_of_coverage(day_of_coverage)
        Returns the DayOfCoverage object for a given day of coverage.
    nearest_road_worker(subset)
        A worker function for coverage_to_nearest_road, which finds the nearest road to each frame in a subset of the metadata.
    coverage_to_nearest_road(day_of_coverage)
        Finds the nearest road to each frame in a day of coverage.
    
    """ 
    def __init__(self, proj_path, graphml_input):
        self.log = logging.getLogger(__name__)
        self.log.info(f'Loading graph at path {graphml_input}')
        self.PROJ_PATH = proj_path 
        self.days_of_coverage = []
        self.geo = ox.io.load_graphml(graphml_input)
        self.gdf_nodes = ox.utils_graph.graph_to_gdfs(self.geo, edges=False)
        self.gdf_edges = ox.utils_graph.graph_to_gdfs(self.geo, nodes=False)
        
        self.log.info("Graph loaded.")
    
    def get_frames_worker(self, folder):

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
            
            # If md_csv is None, return 0
            if md_csv is None:
                return 0
    
            # Check if md_csv exists
            if not os.path.exists(md_csv):
                self.log.warning(f"Metadata CSV: {md_csv} does not exist.")
    
            # Read CSV
            df = pd.read_csv(md_csv, engine='pyarrow')
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["gps_info.longitude"], df["gps_info.latitude"], crs="EPSG:4326"))
            gdf = gdf.to_crs("EPSG:2263")
    
            return gdf

    def get_data(self, day_of_coverage, num_workers=8):
        # Glob all h3-6 hexagon directories within the given day of coverage 
        hex_dirs = glob(os.path.join(self.PROJ_PATH, day_of_coverage, "*"))
        # remove any non-directories 
        hex_dirs = [x for x in hex_dirs if os.path.isdir(x)]

        self.log.info(f"Number of hex_dirs: {len(hex_dirs)}")

        # Allocate a ProcessPoolExecutor with num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # create copy of hex_dirs that points to metadata CSVs
            hex_dirs_md = hex_dirs.copy()
            # glob the csv in each hex_dir
            for i, hex_dir in enumerate(hex_dirs_md):
                # get list of csvs in hex_dir
                csvs = glob(os.path.join(self.PROJ_PATH, day_of_coverage, hex_dir, "*.csv"))
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
            frames = executor.map(self.get_frames_worker, hex_dirs)
            frames = [item for sublist in frames for item in sublist]
            self.log.info(f"All frames loaded. Number of frames: {len(frames)}")

            # Map get_md_counts_worker to all hex_dirs_md
            md = executor.map(self.get_md_worker, hex_dirs_md)
            md = pd.concat(md)
            self.log.info(f"All metadata loaded. Number of rows in md: {len(md.index)}")


          
            # get frame ids from frames list 
            frame_ids = [x.split("/")[-1].split(".")[0] for x in frames]
            # filter md list to only include rows in frame_ids 
            md = md[md["frame_id"].isin(frame_ids)]
       

        # return md for day of coverage
        return md
        




    def load_day_of_coverage(self, day_of_coverage):
        return self.get_data(day_of_coverage)

        

    def add_day_of_coverage(self, day_of_coverage): 
        DoC = DayOfCoverage(day_of_coverage)
        DoC.frames_data = self.get_data(day_of_coverage)
        self.days_of_coverage.append(DoC)
        self.log.info(f"Added day of coverage {day_of_coverage} to graph, with {len(DoC.frames_data.index)} rows.")

    def get_day_of_coverage(self, day_of_coverage):
        for DoC in self.days_of_coverage:
            if DoC.date == day_of_coverage:
                return DoC
        else:
            self.log.error(f"Day of coverage {day_of_coverage} not stored in graph.")
            return 4

    
    def nearest_road_worker(self, subset): 
        # Get nearest edge for each row in md 
        nearest_edges = ox.distance.nearest_edges(self.geo, subset.geometry.x, subset.geometry.y, return_dist=True)

        return nearest_edges

    def coverage_to_nearest_road(self, day_of_coverage): 
        # Check if day of coverage is in graph
        DoC = self.get_day_of_coverage(day_of_coverage)

        # Get day of coverage data
        md = self.get_day_of_coverage(day_of_coverage).frames_data
        md = md.sample(frac=0.1)

        # Split md into 8 * 100 = 800 chunks 
        md_split = np.array_split(md, 80)
        # Allocate a ProcessPoolExecutor with NUM_CORES
        with tqdm(total=len(md_split)) as progress:   
            with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
                # Map nearest_road_worker to md_split
                nearest = [] 
                for ne in executor.map(self.nearest_road_worker, md_split):
                    # Append to nearest_edges
                    nearest.append(ne)
                    # Update progress bar
                    progress.update(1)

                
                nearest_edges, nearest_edges_dist = zip(*nearest)

                nearest_edges = pd.DataFrame(nearest_edges)
                nearest_edges_dist = pd.DataFrame(nearest_edges_dist)

                DoC.nearest_edges = nearest_edges
                DoC.nearest_edges_dist = nearest_edges_dist 
            
            self.log.info(f"Added nearest edges to day of coverage {day_of_coverage}.")
            return 0
        


if __name__ == '__main__':
    graph = G("/share/ju/nexar_data/nexar-scraper","/share/ju/urbanECG/data/geo/nyc.graphml")
    graph.add_day_of_coverage("2023-08-12")
    graph.coverage_to_nearest_road("2023-08-12")
