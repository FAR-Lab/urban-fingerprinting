# 

import sys 
import os

import pandas as pd 
import numpy as np 
import geopandas as gpd 
import geoplot as gplt 
import geoplot.crs as gcrs

from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from user.params.data import LONGITUDE_COL, LATITUDE_COL, TIME_COL, COORD_CRS, PROJ_CRS, TZ, IMG_ID

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

# 


import mapclassify as mc 



log = setup_logger()

log.info("Start of notebook.")

# 
PROJ_CRS = 'EPSG:2263'

# 
from matplotlib import rc 
import matplotlib 




# 



# 


# 
class AnimatedChloropleth():
    def __init__(self, graph, days=None): 
        if days is None: 
            days = [doc.date for doc in graph.days_of_coverage]
        
        gdf = pd.concat([doc.frames_data for doc in graph.days_of_coverage if doc.date in days])
        
        # convert captured_at from epoch timestamp to EST datetime 
        gdf['captured_at'] = pd.to_datetime(gdf['captured_at'], unit='ms')
        gdf['captured_at'] = gdf['captured_at'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

        detections = pd.concat([doc.detections for doc in graph.days_of_coverage if doc.date in days])
        detections.index = detections.iloc[:,0]
        detections = detections.iloc[:,1:]

        gdf = gdf.merge(detections, left_on='frame_id', right_index=True, how='left')
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf['geometry'])
        gdf = gdf.set_crs(PROJ_CRS, allow_override=True)
        

        self.map_data = gdf
        if self.map_data.crs == None: 
            self.map_data.crs = 'EPSG:4326'
        self.background = None 
        self.crs = 'EPSG:4326'
        self.coco_classes = {
        "0": "person",
        "1": "bicycle",
        "2": "car",
        "3": "motorcycle",
        "4": "airplane",
        "5": "bus",
        "6": "train",
        "7": "truck",
        "8": "boat",
        "9": "traffic light",
        "10": "fire hydrant",
        "11": "stop sign",
        "12": "parking meter",
        "13": "bench",
        "14": "bird",
        "15": "cat",
        "16": "dog",
        "17": "horse",
        "18": "sheep",
        "19": "cow",
        "20": "elephant",
        "21": "bear",
        "22": "zebra",
        "23": "giraffe",
        "24": "backpack",
        "25": "umbrella",
        "26": "handbag",
        "27": "tie",
        "28": "suitcase",
        "29": "frisbee",
        "30": "skis",
        "31": "snowboard",
        "32": "sports ball",
        "33": "kite",
        "34": "baseball bat",
        "35": "baseball glove",
        "36": "skateboard",
        "37": "surfboard",
        "38": "tennis racket",
        "39": "bottle",
        "40": "wine glass",
        "41": "cup",
        "42": "fork",
        "43": "knife",
        "44": "spoon",
        "45": "bowl",
        "46": "banana",
        "47": "apple",
        "48": "sandwich",
        "49": "orange",
        "50": "broccoli",
        "51": "carrot",
        "52": "hot dog",
        "53": "pizza",
        "54": "donut",
        "55": "cake",
        "56": "chair",
        "57": "couch",
        "58": "potted plant",
        "59": "bed",
        "60": "dining table",
        "61": "toilet",
        "62": "tv",
        "63": "laptop",
        "64": "mouse",
        "65": "remote",
        "66": "keyboard",
        "67": "cell phone",
        "68": "microwave",
        "69": "oven",
        "70": "toaster",
        "71": "sink",
        "72": "refrigerator",
        "73": "book",
        "74": "clock",
        "75": "vase",
        "76": "scissors",
        "77": "teddy bear",
        "78": "hair drier",
        "79": "toothbrush"
        }


    def set_crs(self, crs): 
        self.crs = crs
    
    def set_background(self, shapefile): 
        if shapefile.crs != self.crs: 
            shapefile = shapefile.to_crs(self.crs)
        self.background = shapefile

    def set_roads(self, graphml): 
        shapefile = ox.graph_to_gdfs(graphml, nodes=False, edges=True)
        if shapefile.crs != self.crs: 
            shapefile = shapefile.to_crs(self.crs)
        self.roads = shapefile


    def plot_frame(self, args): 
        frame, time_column, coloring_column, grouping = args


        fig = plt.figure(figsize=(30, 30))
        ax = plt.subplot(frameon=False)
        
        #print(self.background.total_bounds)
        if grouping == 'bin': 
            print("generating plots with binning")
            scheme = mc.Quantiles(self.map_data[coloring_column], k=10)
            if self.background is not None: 
                self.background.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5, alpha=1)
                pass
            if self.roads is not None:
                print("plotting roads...")
                self.roads.plot(ax=ax, color='black', linewidth=0.3, alpha=0.5)
            gplt.pointplot(
                frame,
                hue=coloring_column,
                scale=coloring_column,
                ax=ax, 
                scheme=scheme,
                extent=self.roads.total_bounds,
                cmap='cividis',
                limits=(1,10),
                zorder=4
            )


        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.map_data[coloring_column].max())
        scalarmap = matplotlib.cm.ScalarMappable(norm=norm, cmap='cividis')
        n = self.map_data[coloring_column].max()
    
        cb = plt.colorbar(scalarmap, ticks=np.arange(0,n), ax=ax, orientation='horizontal', ticklocation='bottom', pad=0, anchor=(0.2,6), shrink=0.35)
        cb.ax.xaxis.set_label_position('top')
        cb.set_label(label=f'# of {self.curr_entity_name}s',size=40,weight='bold', labelpad=20)
        cb.ax.tick_params(labelsize=25)
        # set ticks to every 3 
        cb.set_ticks(np.arange(0,n,3))


        ax.set_title(f"Date: {frame[time_column].max()}")
        ax.title.set_size(40)
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        # set legend title using self.curr_entity_name
        #ax.get_legend().set_title(f"# of {self.curr_entity_name}s")
        # increase legend title font size to 35
        #ax.get_legend().get_title().set_fontsize('35')

     




        # increase legend size 
        #ax.get_legend().get_texts()[0].set_fontsize('25')

        plt.savefig(f"{self.frames_dir}/{frame[time_column].max().strftime('%Y-%m-%d-%H-%M-%S')}.png", bbox_inches="tight", pad_inches=0)
        fig.clf()
        plt.close(fig)

        del frame
        del fig
        del ax

    def generate_frames(self, time_column, coloring_column, time_delta, grouping='bin', car_offset=False): 

        matplotlib.pyplot.ioff()
        plt.tight_layout(pad=0.00)

        # Requires local LaTeX installation 
        try:
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
            matplotlib.rcParams['text.usetex'] = True
            rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage[T1]{fontenc}')

            log.info('LaTeX font rendering configured correctly.')

        except Exception as e: 
            log.warning(f'LaTeX font rendering not configured correctly, error: {e}.')

        try: 
            self.curr_entity_name = self.coco_classes[coloring_column]
        except KeyError as e:
            log.error(f"Could not find entity name for {coloring_column}.")
            return

         # add car_offset if applicable 
        if car_offset: 
            log.info('Car offset mode enabled (counting each dashcam image as proof of one car.)')
            self.map_data[coloring_column] = self.map_data[coloring_column] + 1
        
        # remove rows with values of 0 
        self.map_data = self.map_data[self.map_data[coloring_column] > 0]

        # make sure crs match 
        if self.map_data.crs != self.crs: 
            self.map_data = self.map_data.to_crs(self.crs)

        # get earliest and latest datetime
        earliest = self.map_data[time_column].min()
        latest = self.map_data[time_column].max()

        # generate range of datetimes with time_delta 
        intervals = pd.date_range(earliest, latest, freq=time_delta)


        # split map_data into intervals
        frames = []
        # get subset of map_data for each interval
        for i in tqdm(range(len(intervals)-1)):
            frames.append(self.map_data[(self.map_data[time_column] >= intervals[i]) & (self.map_data[time_column] < intervals[i+1])])


        

            
            
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # make directory with now variable 
        os.makedirs(now, exist_ok=True)

        self.frames_dir = now 

        args = []

        for i in range(len(frames)):
            # make sliding window of 3 frames 
            if i < 3:
                window = frames[:i+1]
            else:
                window = frames[i-2:i+1]
            # flatten list of frames
            window = pd.concat(window)

            args.append((window, time_column, coloring_column, grouping))


        frames = Parallel(n_jobs=8, timeout=10000)(delayed(self.plot_frame)(arg) for arg in tqdm(args))
        return frames
            
            
    def generate_gif(self):

        if not self.frames_dir: 
            log.error("No frames directory found.")
            return

        fp_out = f"{self.curr_entity_name}_{self.frames_dir}.gif"

        with contextlib.ExitStack() as stack:

            # lazily load images
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(f"{self.frames_dir}/*.png")))

            # extract  first image from iterator
            img = next(imgs)

            img.save(fp=fp_out, format='GIF', append_images=imgs,
                    save_all=True, duration=42, loop=0)



# 


# 



