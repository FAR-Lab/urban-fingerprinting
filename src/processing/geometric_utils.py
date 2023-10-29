# FARLAB - UrbanECG
# Developers: @mattwfranchi, @DorinRu, with help from GitHub CoPilot
# Last Modified: 10/25/23

# This script houses a series of functions that compute the 'perspective' of a Nexar frame.

# Module Imports
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../..", "src")))

import math

from user.params.data import *

import numpy as np
import pandas as pd
import geopandas as gpd

from src.utils.logger import setup_logger


# Class Definitions
class Frame:
    def __init__(self, md_row):
        self.id = md_row[IMG_ID]
        self.lng = md_row[LONGITUDE_COL]
        self.lat = md_row[LATITUDE_COL]
        self.captured_at = md_row[TIME_COL]
        self.location = (self.lng, self.lat)
        self.heading = md_row[ORIENTATION_COL]
        self.direction = md_row[DIRECTION_COL]
        self.log = setup_logger(name=self.id)

    def distance(self, other):
        match other:
            case Frame():
                return np.sqrt(
                    (self.lng - other.lng) ** 2 + (self.lat - other.lat) ** 2
                )
            case (float(), float()):
                return np.sqrt((self.lng - other[0]) ** 2 + (self.lat - other[1]) ** 2)

    def angle(self, other):
        match other:
            case Frame():
                pass

            case (float(), float()):
                lng = other[0]
                lat = other[1]

                delta_lon = lng - self.lng

                x = math.atan2(
                    math.sin(delta_lon) * math.cos(lat),
                    math.cos(self.lat) * math.sin(lat)
                    - math.sin(self.lat) * math.cos(lat) * math.cos(delta_lon),
                )

                x = math.degrees(x)

                return x

    def angle_from_direction(self):
        match self.direction:
            case "NORTH":
                angle = 0
            case "NORTH_EAST":
                angle = 45
            case "EAST":
                angle = 90
            case "SOUTH_EAST":
                angle = 135
            case "SOUTH":
                angle = 180
            case "SOUTH_WEST":
                angle = 225
            case "WEST":
                angle = 270
            case "NORTH_WEST":
                angle = 315
            case _:
                angle = 0

        return angle

    def angle_btwn(self, other):
        match other:
            case Frame():
                return np.abs(self.heading - other.heading)
            case float():
                return np.abs(self.heading - other)

    def angle_btwn_direction(self, other):
        match other:
            case Frame():
                return np.abs(
                    self.angle_from_direction() - other.angle_from_direction()
                )
            case float():
                return np.abs(self.angle_from_direction() - other)

    def depicts_coordinates(self, coordinates: tuple):
        # compute angle between self and coordinates

        angle = self.angle(coordinates)

        # compute difference between angle and heading
        # if difference is within view cone, pass
        # else, fail

        if self.angle_btwn_direction(angle) > VIEW_CONE:
            self.log.info(
                f"Angle between frame and coordinates is {self.angle_btwn_direction(angle)}, > {VIEW_CONE}"
            )
            return False

        # now, compute distance between self and coordinates
        # if distance is within view distance, pass
        # else, fail
        if self.distance(coordinates) > VIEW_DISTANCE:
            self.log.info(
                f"Distance between frame and coordinates is {self.distance(coordinates)}, > {VIEW_DISTANCE}"
            )
            return False

        return True

    def __str__(self):
        return f"Frame {self.id} at {self.location}, captured at {self.captured_at}"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Perspective:
    # Rules of a Perspective:
    # 1. It is defined by a set of frames
    # 2. The set of frames must be on the same edge, or same intersection of the road graph
    # 3. The set of frames must be within a certain time window of each other (this is user-defined)
    # 4. We assume that frames in a Perspective may depict the same object, but from different angles.

    def __init__(self):
        self.frames = []
        self.u = None
        self.v = None
        self.timespan = None
        self.earliest = None
        self.latest = None
