# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot 
# Last Edited: 09/15/2023 

# This script is used to take the roads of a graph G, and a set of annotated dashcam frames F, and generate commodity densities for each road in G.

import pandas as pd
import numpy as np
import geopandas as gpd 
import osmnx as ox 
import os
import sys 
from glob import glob 

from fire import Fire
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid 
from joblib import Parallel, delayed
from itertools import chain 
import datetime 

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

from shapely import wkt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import coco_mappings as cm
from visualization import AnimatedMap as am
from DayOfCoverage import DayOfCoverage


# Constant for number of workers to use in parallel, should be equal to number of cores on machine
NUM_CORES = os.getenv("SLURM_CPUS_ON_NODE")
# Check if none
if NUM_CORES is None:
    # Set to 8
    NUM_CORES = 8
else: 
    NUM_CORES *= 2



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

        self.DEBUG_MODE = False 
        self.WRITE_MODE = False 
        self.log = logging.getLogger(__name__)
        self.log.info(f'Loading graph at path {graphml_input}')
        self.PROJ_PATH = proj_path 
        self.days_of_coverage = []
        self.geo = ox.io.load_graphml(graphml_input)
        self.gdf_nodes = ox.utils_graph.graph_to_gdfs(self.geo, edges=False)
        self.gdf_edges = ox.utils_graph.graph_to_gdfs(self.geo, nodes=False)
        
        self.log.info("Graph loaded.")

    def toggle_debug(self):
        self.DEBUG_MODE = not self.DEBUG_MODE
        self.log.info(f"Debug mode set to {self.DEBUG_MODE}.")
    
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
    
            return df

    def get_data(self, day_of_coverage, num_workers=24):

        # Check if md file has already been written to output 
        if os.path.exists(f"../../output/df/{day_of_coverage}/md.csv"):
            # read md file from output 
            md = pd.read_csv(f"../../output/df/{day_of_coverage}/md.csv", engine='pyarrow')
            md = gpd.GeoDataFrame(md, geometry=wkt.loads(md['geometry']), crs="EPSG:4326")
            # return md
            self.log.info(f"Loading metadata from output for day of coverage {day_of_coverage}.")
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
        frames = Parallel(n_jobs=NUM_CORES)(delayed(self.get_frames_worker)(folder) for folder in tqdm(hex_dirs))
        frames = list(chain.from_iterable(frames))
        self.log.info(f"All frames loaded. Number of frames: {len(frames)}")

        # Map get_md_counts_worker to all hex_dirs_md
        md = Parallel(n_jobs=NUM_CORES)(delayed(self.get_md_worker)(md_csv) for md_csv in tqdm(hex_dirs_md))
        md = pd.concat(md)
        self.log.info(f"All metadata loaded. Number of rows in md: {len(md.index)}")


        
        # get frame ids from frames list 
        frame_ids = [x.split("/")[-1].split(".")[0] for x in frames]
        # filter md list to only include rows in frame_ids 
        md = md[md["frame_id"].isin(frame_ids)]

        # turn md into GeoDataFrame
        md = gpd.GeoDataFrame(md, geometry=gpd.points_from_xy(md["gps_info.longitude"], md["gps_info.latitude"], crs="EPSG:4326"))
        md = md.to_crs("EPSG:2263")
       
        if self.WRITE_MODE:
            os.makedirs(f"../../output/df/{day_of_coverage}", exist_ok=True)
            md.to_csv(f"../../output/df/{day_of_coverage}/md.csv")
            self.log.info(f"Wrote metadata to output for day of coverage {day_of_coverage}.")


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
        md = md.to_crs("EPSG:2263")

        # Check if nearest edges have already been written to output
        if os.path.exists(f"../../output/df/{day_of_coverage}/nearest_edges.csv"):
            # read nearest edges from output 
            nearest_edges = pd.read_csv(f"../../output/df/{day_of_coverage}/nearest_edges.csv", engine='pyarrow')
            # return nearest edges
            self.log.info(f"Loading nearest edges from output for day of coverage {day_of_coverage}.")
            DoC.nearest_edges = nearest_edges
            return 0

        md_split = np.array_split(md, 10 * NUM_CORES)


        nearest = Parallel(n_jobs=NUM_CORES)(delayed(self.nearest_road_worker)(subset) for subset in tqdm(md_split))
        nearest_edges, nearest_edges_dist = zip(*nearest)

        nearest_edges = [item for sublist in nearest_edges for item in sublist]
        nearest_edges_dist = [item for sublist in nearest_edges_dist for item in sublist]


        nearest_edges = pd.DataFrame(nearest_edges, columns=['u', 'v', 'key'])
        nearest_edges['dist'] = nearest_edges_dist
        nearest_edges['frame_id'] = md['frame_id'].tolist()
        nearest_edges = nearest_edges.set_index('frame_id')

        DoC.nearest_edges = nearest_edges

        if self.WRITE_MODE:
            os.makedirs(f"../../output/df/{day_of_coverage}", exist_ok=True)
            nearest_edges.to_csv(f"../../output/df/{day_of_coverage}/nearest_edges.csv")
            self.log.info(f"Wrote nearest edges to output for day of coverage {day_of_coverage}.")

    
        self.log.info(f"Added nearest edges to day of coverage {day_of_coverage}.")
        return 0
    
    def add_detections(self, day_of_coverage):
        detections = pd.read_csv(f"../../output/df/{day_of_coverage}/detections.csv", engine='pyarrow')
        DoC = self.get_day_of_coverage(day_of_coverage)
        DoC.detections = detections
        self.log.info(f"Added detections to day of coverage {day_of_coverage}.")
        return 0
    
    

    def plot_edges(self): 
        _, ax = plt.subplots(figsize=(20,20))
        self.gdf_edges.plot(ax=ax, color='black', linewidth=0.5)
        rID = uuid.uuid4().hex[:8]
        plt.savefig(f"../../output/plots/edges_{rID}.png")
        plt.close()
    
    def plot_coverage(self, day_of_coverage):
        DoC = self.get_day_of_coverage(day_of_coverage)
        _, ax = plt.subplots(figsize=(30,30))

        # group by nearest edge, plot chloropleth 
        print(DoC.nearest_edges.columns)
        print(DoC.nearest_edges.head())

        coverage = DoC.nearest_edges.groupby(['u', 'v']).size().reset_index(name='counts')



        coverage['binned'] = pd.qcut(coverage['counts'], 100, labels=False, duplicates='drop')
        

        self.gdf_edges.plot(ax=ax, color='lightcoral', linewidth=0.5)
        self.gdf_edges.merge(coverage, on=['u', 'v'], how='left').plot(ax=ax, column='binned', cmap='cividis', linewidth=0.5, legend=True,
                                                                                 legend_kwds={'label': "Number of frames", 'orientation': "horizontal"})
        
        ax.set_axis_off()
        ax.margins(0)
        ax.set_title(f"Coverage for {day_of_coverage}")
        ax.title.set_size(50)


        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(f"../../output/plots/coverage_{rID}.png", bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def plot_detections(self, day_of_coverage, class_id):
        DoC = self.get_day_of_coverage(day_of_coverage)
        _, ax = plt.subplots(figsize=(30,30), frameon=True)

        coverage = DoC.nearest_edges.merge(DoC.detections, left_index=True, right_index=True)

        coverage = coverage.groupby(['u', 'v']).agg({str(class_id): 'mean'}).reset_index()
        

        #coverage['binned'] = pd.qcut(coverage[str(class_id)], 100, labels=False, duplicates='drop')

        self.gdf_edges.plot(ax=ax, color='lightcoral', linewidth=0.5, alpha=0.2)
        self.gdf_edges.merge(coverage, on=['u', 'v'], how='left').plot(ax=ax, column=str(class_id), cmap='cividis', linewidth=0.5, legend=True,
                                                                                 legend_kwds={'label': "Number of frames", 'orientation': "horizontal"})
        
        ax.set_axis_off()
        ax.margins(0)
        ax.set_title(f"Average # of {cm.coco_classes[class_id]}s for {day_of_coverage}")
        ax.title.set_size(50)


        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.makedirs(f"../../output/plots/{day_of_coverage}", exist_ok=True)
        plt.savefig(f"../../output/plots/{day_of_coverage}/{class_id}_density_{rID}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    def init_day_of_coverage(self, doc):
        self.add_day_of_coverage(doc)
        self.coverage_to_nearest_road(doc)
        self.add_detections(doc)

        self.log.info(f"Added {doc} to graph.")
    
    def generate_gif(self, DoCs): 
        gif = am.AnimatedChloropleth(self, DoCs)
        gif.set_roads(self.geo)
        gif.generate_frames("captured_at", "2", "10min", "bin", car_offset=True)
        gif.generate_gif()








if __name__ == '__main__':
    #days_of_coverage = ["2023-08-10", "2023-08-11", "2023-08-12", "2023-08-13", "2023-08-14", "2023-08-17", "2023-08-18", "2023-08-20", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31"]
    days_of_coverage = ["2023-08-10", "2023-08-11", "2023-08-12"]
    graph = G("/share/ju/nexar_data/nexar-scraper","/share/ju/urbanECG/data/geo/nyc.graphml")
    for day in days_of_coverage:
        try:
            graph.init_day_of_coverage(day)
        except Exception as e:
            graph.log.error(f"Error in {day}: {e}")
    graph.generate_gif(days_of_coverage)
   
