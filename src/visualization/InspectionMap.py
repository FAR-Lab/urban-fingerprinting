# FARLAB - UrbanECG Project
# Dev: @mattwfranchi, with help from GitHub CoPilot 
# Last Modified: 10/26/2023

# This script houses a class that generates an interactive map of a set of Nexar frames. 

# Module Imports
import sys 
import os 

sys.path.append(os.path.abspath(os.path.join('../..', 'src')))
from src.processing.geometricUtils import Frame, Perspective
from user.params.data import *

import pandas as pd
import folium
import numpy as np
import imageio
import cv2
import base64


class InspectionMap():
    def __init__(self, center): 
        self.map = folium.Map(location=center, zoom_start=14, control_scale=True)

    def save(self, path):
        # make output directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # remove file if it already exists
        if os.path.exists(path):
            os.remove(path)
        self.map.save(path)

    def add_frame_marker(self, frame, path_to_image):
        # add jpg image to map
        try:
            with open(path_to_image, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            folium.Marker(
                location=[frame[LATITUDE_COL], frame[LONGITUDE_COL]],
                popup=folium.Popup(f'<img src="data:image/png;base64,{encoded}" width="1280" height="720">', max_width=1280),
                icon=folium.Icon(color='blue', icon='camera', prefix='fa')
            ).add_to(self.map)
        except Exception as e:
            print(f"Failed to add frame marker for frame {frame['img_id']}: {e}")

    def add_frames(self, frames, path_to_images):
        for frame in frames:
            self.add_frame_marker(frame, os.path.join(path_to_images, frame['img_id'] + ".jpg"))



