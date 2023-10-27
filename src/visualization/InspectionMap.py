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

from datetime import datetime

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
                img = f.read() 
                # compress image by 50% 
                img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                # now, encode the image as base64
                encoded = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

                popup_html = f'''
                <div style="width: 640px; height: 420px; overflow: hidden;">
                    <div id="frame-info">
                        <p><b>Frame ID:</b> {frame[IMG_ID]}, Capture Time: {datetime.fromtimestamp(int(frame[TIME_COL] / 1000))}</p>
                        <p>Location: {frame[LATITUDE_COL]}, {frame[LONGITUDE_COL]}, Heading: {frame[DIRECTION_COL]}</p>
                    </div>
                    <img src="data:image/png;base64,{encoded}" width="640" height="360">
                    
                </div>'''

                


                folium.Marker(
                    location=[frame[LATITUDE_COL], frame[LONGITUDE_COL]],
                    popup=folium.Popup(popup_html, max_width=640),
                    icon=folium.Icon(color='green', icon='camera', prefix='fa')
                ).add_to(self.map)
        except Exception as e:
            print(f"Failed to add frame marker for frame {frame[IMG_ID]}: {e}")

    def add_311_marker(self, complaint):
        popup_html = f'''
            <div style="width: 640px; height: 420px; overflow: hidden;">
                <div id="311_info">
                <table>
                    <tr>
                        <td><b>Complaint ID:</b></td>
                        <td>{complaint[ID_COL_311]}</td>
                        <td><b>Agency:</b></td>
                        <td>{complaint[AGENCY_NAME_311]}</td>
                    </tr>
                    <tr>
                        <td><b>Created:</b></td>
                        <td>{complaint[START_DATE_311]}</td>
                        <td><b>Closed:</b></td>
                        <td>{complaint[END_DATE_311]}</td>
                    </tr>
                    <tr>
                        <td><b>Location:</b></td>
                        <td>{complaint[LONGITUDE_COL_311]}, {complaint[LATITUDE_COL_311]}</td>
                        <td><b>Cross Streets:</b></td>
                        <td>{complaint[XSTREET_1_311]}, {complaint[XSTREET_2_311]}</td>
                    </tr>
                    <tr>
                        <td><b>Complaint Type:</b></td>
                        <td>{complaint[TYPE_COL_311]}</td>
                        <td><b>Descriptor:</b></td>
                        <td>{complaint[DESC_COL_311]}</td>
                    </tr>
                </table>
                </div>
            </div>'''

        icon_colors = { 
            'Sewer Backup (Use Comments) (SA)': 'red',
            'Catch Basin Clogged/Flooding (Use Comments) (SC)': 'purple',
            'Street Flooding (SJ)': 'blue',
            'Manhole Overflow (Use Comments) (SA1)': 'orange'
        }
        try:
            folium.Marker(
                location= [complaint[LATITUDE_COL_311], complaint[LONGITUDE_COL_311]],
                popup=folium.Popup(popup_html, max_width=640),
                icon=folium.Icon(color=icon_colors[complaint[DESC_COL_311]], icon='exclamation-triangle', prefix='fa')
            ).add_to(self.map)
        except ValueError as e:
            print(f"Failed to add 311 marker for complaint {complaint[ID_COL_311]}: {e}")


    def add_frames(self, frames, path_to_images):
        for frame in frames:
            self.add_frame_marker(frame, os.path.join(path_to_images, frame[IMG_ID] + ".jpg"))



