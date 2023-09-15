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


# Set up logging
import logging
logging.basicConfig(level=logging.INFO)


# Constant for number of workers to use in parallel, should be equal to number of cores on machine
NUM_CORES = os.getenv("SLURM_CPUS_ON_NODE")
# Check if none
if NUM_CORES is None:
    # Set to 8
    NUM_CORES = 6

class G: 
    def __init__(self, proj_path, graphml_input):
        self.log = logging.getLogger(__name__)
        self.log.info(f'Loading graph at path {graphml_input}')
        self.PROJ_PATH = proj_path 
        self.days_of_coverage = {}
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
            df = pd.read_csv(md_csv)
    
            # Return length of CSV
            return df

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
        self.days_of_coverage[day_of_coverage] = self.load_day_of_coverage(day_of_coverage)
        self.log.info(f"Added day of coverage {day_of_coverage} to graph, with {len(self.days_of_coverage[day_of_coverage].index)} rows.")



if __name__ == '__main__':
    graph = G("/share/ju/nexar_data/nexar-scraper","/share/ju/urbankeg/")
