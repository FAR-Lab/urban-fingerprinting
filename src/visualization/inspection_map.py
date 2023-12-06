# FARLAB - UrbanECG Project
# Dev: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/26/2023

# This script houses a class that generates an interactive map of a set of Nexar frames.

# Module Imports
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../..", "src")))
from src.processing.geometric_utils import Frame, Perspective
from user.params.data import *
from src.utils.logger import setup_logger

import pandas as pd
import folium
import numpy as np
import imageio
import cv2
import base64
import logging

from datetime import datetime


class InspectionMap:
    def __init__(self, center):
        self.log = setup_logger("InspectionMap")
        self.log.setLevel(logging.DEBUG)
        self.map = folium.Map(
            location=center, zoom_start=14, control_scale=True
        )
        self.frames_group = folium.FeatureGroup(name="Nexar Frames")
        self.sensors_group = folium.FeatureGroup(name="FloodNet Sensors")
        self.complaints_groups = {}
        self.debug = False

    def init_complaint_group(self, name):
        try:
            self.complaints_groups[name] = folium.FeatureGroup(name=name)
        except Exception as e:
            print(f"Failed to initialize complaint group {name}: {e}")

    def save(self, path):
        # add frame group to map
        self.map.add_child(self.frames_group)
        # add sensor group to map
        self.map.add_child(self.sensors_group)

        # add complaints groups to map
        for group in self.complaints_groups.values():
            self.map.add_child(group)

        # add layer control
        folium.LayerControl().add_to(self.map)

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
                img = cv2.imdecode(
                    np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR
                )
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

                # now, encode the image as base64
                encoded = base64.b64encode(
                    cv2.imencode(".jpg", img)[1]
                ).decode("utf-8")

                debug_div = ""
                try:
                    if self.debug:
                        test = Frame(frame)
                        test_angle = test.angle((frame[LONGITUDE_COL_311], frame[LATITUDE_COL_311]))
                        test_angle_btwn = test.angle_btwn(test_angle)
                        test_distance = test.distance((frame[LONGITUDE_COL_311], frame[LATITUDE_COL_311]))

                        debug_div = f"""
                        <div id="debug-info">
                            <h3>Debug Info</h3>
                            <table>
                                <tr>
                                    <td><b>Relevant 311 ID</b>: {frame['Unique Key']}<td>
                                    <td><b> Angle between Points:</b> {test_angle}</td>
                                </tr>
                                <tr>
                                    <td><b> Angle Between Headings:</b> {test_angle_btwn}</td>
                                    <td><b> Distance to Complaint:</b> {test_distance}</td>
                                </tr>
                            </table>
                        </div>"""
                except Exception as e:
                    self.log.error(f"Failed to generate debug div: {e}")

                try:
                    popup_html = f"""
                    <div style="width: 640px; height: 720px; overflow: hidden;">
                        <div id="frame-info">
                            <h3>Frame Info</h3>
                            <p><b>Frame ID:</b> {frame[IMG_ID]}, Capture Time: {frame[TIME_COL]}</p>
                            <p>Location: {frame[LATITUDE_COL]}, {frame[LONGITUDE_COL]}, Heading: {frame[DIRECTION_COL]}, Angle: {frame[ORIENTATION_COL]}</p>
                            <p>Projected: {frame.geometry.x}, {frame.geometry.y}</p>
                        </div>
                        {debug_div}
                        <img src="data:image/png;base64,{encoded}" width="640" height="360">
                        
                    </div>"""
                except Exception as e:
                    self.log.error(
                        f"Failed to generate popup HTML for frame {frame[IMG_ID]}: {e}"
                    )
                    return

                

                try:
                    self.frames_group.add_child(
                        folium.Marker(
                            location=[frame[LATITUDE_COL], frame[LONGITUDE_COL]],
                            popup=folium.Popup(popup_html, max_width=640),
                            icon=folium.Icon(
                                color="green", icon="camera", prefix="fa"
                            ),
                        )
                    )
                except Exception as e:
                    self.log.error(
                        f"Failed to add frame marker for frame {frame[IMG_ID]}: {e}"
                    )
                    return

                self.log.success(f"Added frame {frame[IMG_ID]} to map")
        except Exception as e:
            self.log.error(
                f"Failed to add frame marker for frame {frame[IMG_ID]}: {e}"
            )
            return
        
    def add_floodnet_marker(self, sensor): 
        try:
            popup_html = f"""
                <div style="width: 640px; height: 420px; overflow: hidden;">
                    <div id="sensor_info"> 
                        <table> 
                        <tr>
                            <td><b>Sensor ID:</b></td>
                            <td>{sensor["deployment_id"]}</td>
                        </tr>
                        </table>
            """
        except Exception as e:
            self.log.error(
                f"Failed to generate popup HTML for sensor {sensor['deployment_id']}: {e}"
            )
            return
        
        try: 
            self.sensors_group.add_child(
                folium.Marker(
                    location=[sensor["lat"], sensor["lon"]],
                    popup=folium.Popup(popup_html, max_width=640),
                    icon=folium.Icon(
                        color="purple", icon="tint", prefix="fa"
                    ),
                )
            )

            self.log.success(f"Added sensor {sensor.deployment_id} to map")  
        except Exception as e:
            self.log.error(
                f"Failed to add sensor marker for sensor {sensor['deployment_id']}: {e}"
            )
            


    def add_311_marker(self, complaint):
        try:
            popup_html = f"""
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
                        <tr>
                            <td>Projected X:</td>
                            <td>{complaint.geometry.x}</td>
                            <td>Projected Y:</td>
                            <td>{complaint.geometry.y}</td>
                        </tr>
                    </table>
                    </div>
                </div>"""
        except Exception as e:
            self.log.error(
                f"Failed to generate popup HTML for complaint {complaint[ID_COL_311]}: {e}"
            )
            return

        icon_colors = {
            "Sewer Backup (Use Comments) (SA)": "red",
            "Catch Basin Clogged/Flooding (Use Comments) (SC)": "purple",
            "Street Flooding (SJ)": "blue",
            "Manhole Overflow (Use Comments) (SA1)": "orange",
        }
        try:
            self.complaints_groups[complaint[DESC_COL_311]].add_child(
                folium.Marker(
                    location=[
                        complaint[LATITUDE_COL_311],
                        complaint[LONGITUDE_COL_311],
                    ],
                    popup=folium.Popup(popup_html, max_width=640),
                    icon=folium.Icon(
                        color=icon_colors[complaint[DESC_COL_311]],
                        icon="exclamation-triangle",
                        prefix="fa",
                    ),
                )
            )
        except ValueError as e:
            print(
                f"Failed to add 311 marker for complaint {complaint[ID_COL_311]}: {e}"
            )

    def add_frames(self, frames, path_to_images):
        for frame in frames:
            self.add_frame_marker(
                frame, os.path.join(path_to_images, frame[IMG_ID] + ".jpg")
            )
