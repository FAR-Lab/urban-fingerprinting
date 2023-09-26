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
from PIL import Image

import contextlib

from fire import Fire
from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid 
from joblib import Parallel, delayed
from itertools import chain 
import datetime 
import pytz

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
    NUM_CORES = int(NUM_CORES)



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

    def toggle_latex_font(self):
        plt.rcParams['text.usetex'] = not plt.rcParams['text.usetex']
        self.log.info(f"Latex font set to {plt.rcParams['text.usetex']}.")

    
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
        ax.set_title(f"Average Num. of {cm.coco_classes[class_id]}s for {day_of_coverage}")
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

    def join_days(self, DoCs): 
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

        # Get metadata for each day of coverage
        md = []
        for doc in DoCs:
            md.append(self.get_day_of_coverage(doc).frames_data)
        
        # Concatenate metadata
        md = pd.concat(md)
        md['captured_at'] = pd.to_datetime(md['captured_at'], unit='ms')
        md['captured_at'] = md['captured_at'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

        return DayOfCoverage("joined", md, nearest_edges, detections)


    def coco_agg_mappings(self, operation='mean', data = pd.DataFrame()):
        agg_dict = {}
        for class_id in cm.coco_classes.keys():
            if str(class_id) in data.detections.columns.astype(str):
                agg_dict[str(class_id)] = operation
        return agg_dict


    def density_per_road_segment(self, DoCs, dtbounds=(datetime.datetime(1970,1,1,0,0,0), datetime.datetime(2024,1,1,0,0,0))): 
        # Make sure dtbounds are localized in EST 
        dtbounds = (dtbounds[0].astimezone(tz=pytz.timezone('America/New_York')), dtbounds[1].astimezone(tz=pytz.timezone('America/New_York')))
        data = self.join_days(DoCs)

        md = data.frames_data
        

        md = md[(md['captured_at'] >= dtbounds[0]) & (md['captured_at'] <= dtbounds[1])]

        
        detections = data.detections 
        detections.set_index(detections.iloc[:,0], inplace=True)
        detections.fillna(0, inplace=True)

       

        density = md.merge(detections, left_on='frame_id', right_index=True)
        density = density.merge(data.nearest_edges, left_on='frame_id', right_on='frame_id')
        



        density = density.groupby(['u', 'v']).agg(self.coco_agg_mappings(data=data)).reset_index()

        os.makedirs(f"../../output/df/density", exist_ok=True)
        density.describe().to_csv("../../output/df/density/density_describe.csv")
        density.to_csv("../../output/df/density/density.csv")

        del data
        del detections
        del md 

        return density 

    def data2density(self, data, class_id, dtbounds, car_offset=False):


        if car_offset:
            data.detections[str(class_id)] = data.detections[str(class_id)] + 1

        md = data.frames_data
        

        md = md[(md['captured_at'] >= dtbounds[0]) & (md['captured_at'] <= dtbounds[1])]

        
        detections = data.detections 
        detections.set_index(detections.iloc[:,0], inplace=True)
        detections.fillna(0, inplace=True)


        density = md.merge(detections, left_on='frame_id', right_index=True)
        density = density.merge(data.nearest_edges, left_on='frame_id', right_on='frame_id')
        



        density = density.groupby(['u', 'v']).agg(self.coco_agg_mappings(data=data)).reset_index()

        del md 
        del detections 
        del data 

        return density
        



    def plot_density_per_road_segment(self, output_dir, DoCs, bin, density, class_id, binned=True, car_offset=False): 

        density = self.gdf_edges.merge(density, on=['u', 'v'], how='left')
        density = density.to_crs("EPSG:2263")



        _, ax = plt.subplots(figsize=(40,40), frameon=True)

        self.gdf_edges.plot(ax=ax, color='lightcoral', linewidth=0.5, alpha=0.2)

        density['binned'] = pd.cut(density.loc[:, str(class_id)], 100, duplicates='drop')
        # convert bins into upper bound
        density['binned'] = density['binned'].apply(lambda x: x.right)




        density.plot(ax=ax, column='binned' if binned else str(class_id), cmap='BuPu', linewidth=1.5, legend=False)

        # create custom legend based on bins 
        bins = list(density['binned'].unique())
        bins.sort()
        
        # create custom, continuous legend 
        # create a color map
        cmap = plt.cm.BuPu
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
       
        # don't use bins, use a static min and max 
        bounds = np.linspace(0, 1, 60)
        norm = plt.Normalize(bounds.min(), bounds.max())   
        
        # create a second axes for the colorbar, on left side 
        ax2 = ax.figure.add_axes([0.05, 0.05, 0.03, 0.9])
        
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax2)
        cb.ax.tick_params(labelsize=50)
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')
        cb.ax.invert_yaxis()
        cb.ax.set_yticklabels([f"{float(x):.2f}" for x in cb.get_ticks()])
        
        
        cb.ax.title.set_fontsize(50)
        cb.ax.title.set_horizontalalignment('center')
        cb.ax.title.set_verticalalignment('bottom')

        ax.set_axis_off()
        ax.margins(0)
        ax.set_title(f"Average Num. of {cm.coco_classes[str(class_id)]}s per road segment \n {DoCs[0]}-{DoCs[-1]} \n {bin}")
        ax.title.set_size(50)
        # Move title up 
        ax.title.set_position([.5, 1.05])

        rID = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        os.makedirs(f"../../output/plots/{output_dir}", exist_ok=True)
        plt.savefig(f"../../output/plots/{output_dir}/{class_id}_density_{rID}.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close()

    

    def density_over_time_gif(self, docs, dtbounds, class_id, delta="60min", car_offset=True):
        class_id = str(class_id)
        data = self.join_days(docs)

        # generate time bins 
        bins = pd.date_range(dtbounds[0], dtbounds[1], freq=delta)
        bins = bins.tz_localize('America/New_York')

        output_dir = uuid.uuid4().hex[:8]

        args = [] 
        for idx, bin in enumerate(bins): 
            # Sliding window 
            if idx < 3: 
                dtbounds = (bins[0], bins[idx])
            else:
                dtbounds = (bins[idx-3], bins[idx])
            dtbounds = (bin, bin + pd.Timedelta(delta))
            plot_data = data.frames_data[(data.frames_data['captured_at'] >= dtbounds[0]) & (data.frames_data['captured_at'] <= dtbounds[1])]
            density = self.data2density(data, class_id, dtbounds, car_offset=car_offset)
            del plot_data 
            args.append((output_dir, docs, bin, density, class_id, car_offset))
        


        Parallel(n_jobs=NUM_CORES)(delayed(self.plot_density_per_road_segment_parallel)(arg) for arg in tqdm(args))

        # generate gif
        self.generate_gif(output_dir, class_id, docs, delta)
        
    def plot_density_per_road_segment_parallel(self, args): 
        output_dir, docs, bin, density, class_id, car_offset = args
        try:
            graph.plot_density_per_road_segment(output_dir, docs, bin, density, class_id, car_offset=car_offset)
            del args
            del density
        except Exception as e:
            graph.log.error(f"Error in {bin}: {e}")
            return 4

    def generate_gif(self, frames_dir, class_id, days_of_coverage, delta, fps=24, duration=42, loop=0):
        
        os.makedirs(f"../../output/gifs", exist_ok=True)

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if not frames_dir: 
            self.log.error("No frames directory found.")
            return

        fp_out = f"../../output/gifs/{class_id}_{frames_dir}_{now}.gif"

        with contextlib.ExitStack() as stack:

            # lazily load images
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob(f"../../output/plots/{frames_dir}/*.png")))

            # extract  first image from iterator
            img = next(imgs)

            img.save(fp=fp_out, format='GIF', append_images=imgs,
                    save_all=True, duration=42, loop=0)

        

        
        
        

            

    

                                            






    


            







if __name__ == '__main__':
    #days_of_coverage = ["2023-08-10", "2023-08-11", "2023-08-12", "2023-08-13", "2023-08-14", "2023-08-17", "2023-08-18", "2023-08-20", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31"]
    days_of_coverage = ["2023-08-10", "2023-08-11", "2023-08-12"]
    graph = G("/share/ju/nexar_data/nexar-scraper","/share/ju/urbanECG/data/geo/nyc.graphml")
    graph.toggle_latex_font()
    for day in days_of_coverage:
        try:
            graph.init_day_of_coverage(day)
        except Exception as e:
            graph.log.error(f"Error in {day}: {e}")
    #density = graph.density_per_road_segment(days_of_coverage, dtbounds=(datetime.datetime(2023,8,10,12,0,0), datetime.datetime(2023,8,10,14,0,0)))
    #graph.plot_density_per_road_segment(days_of_coverage, density, 2, car_offset=True)

    graph.density_over_time_gif(days_of_coverage, (datetime.datetime(2023,8,10,1,0,0), datetime.datetime(2023,8,11,0,0,0)), 2, delta="10min", car_offset=True)    

   

