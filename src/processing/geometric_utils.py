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

import pyproj
from pyproj import CRS
from shapely.geometry import box, Point

from src.utils.logger import setup_logger

import logging


# Class Definitions
class CRSTransformer:
    def __init__(self, coord_crs=COORD_CRS, proj_crs=PROJ_CRS):
        self.coord_crs = pyproj.CRS(coord_crs)
        self.proj_crs = pyproj.CRS(proj_crs)

        self.transformer = pyproj.Transformer.from_crs(
            self.coord_crs, self.proj_crs, always_xy=True
        )

    def within_crs_bounds(self, x, y):
        crs_bounds = self.transformer.transform_bounds(
            *self.proj_crs.area_of_use.bounds
        )
        crs_bounding_box = box(*crs_bounds)

        return Point(x, y).within(crs_bounding_box)


class Frame:
    # Class-level CRS transformer
    crs_transformer = CRSTransformer()

    def __init__(self, md_row):
        self.id = md_row[IMG_ID]
        self.log = setup_logger(name=self.id)
        self.log.setLevel(logging.DEBUG)
        
        self.lng = md_row[LONGITUDE_COL]
        self.lat = md_row[LATITUDE_COL]
        self.captured_at = md_row[TIME_COL]

        self.x, self.y = Frame.crs_transformer.transformer.transform(
            self.lng, self.lat
        )
        self.location = (self.x, self.y)
        self.log.debug(f"Coordinates: ({self.lng}, {self.lat})")
        self.log.debug(f"Location: {self.location}")


        self.heading = md_row[ORIENTATION_COL]
        self.direction = md_row[DIRECTION_COL]
        

    def distance(self, other):
        match other:
            case Frame():
                return np.sqrt(
                    (self.x - other.x) ** 2 + (self.y - other.y) ** 2
                )
            case (float(), float()):

                if not Frame.crs_transformer.within_crs_bounds(other[0], other[1]):
                    self.log.warning(
                        f"Frame {self.id} is outside of CRS bounds, projecting..."
                    )
                    x2, y2 = Frame.crs_transformer.transformer.transform(
                        other[0], other[1]
                    )
                    
                
                return np.sqrt(
                    (self.x - x2) ** 2 + (self.y - y2) ** 2
                )

    def angle(self, other):
        match other:
            case Frame():
                x1, y1 = self.x, self.y

                if not Frame.crs_transformer.within_crs_bounds(other.x, other.y):
                    self.log.warning(
                        f"Frame {self.id} is outside of CRS bounds, projecting..."
                    )
                    x2, y2 = Frame.crs_transformer.transformer.transform(
                        other.x, other.y
                    )
                else:
                    x2, y2 = other.x, other.y

            case (float(), float()):
                x1 = self.x
                y1 = self.y

                x2, y2 = other
                if not Frame.crs_transformer.within_crs_bounds(x2, y2):
                    self.log.warning(
                        f"Frame {self.id} is outside of CRS bounds, projecting..."
                    )
                    x2, y2 = Frame.crs_transformer.transformer.transform(
                        x2, y2
                    )
                else:
                    x2, y2 = other
            
            case _:
                self.log.error(f"Invalid type for other: {type(other)}")
                raise TypeError(f"Invalid type for other: {type(other)}")
                
        angle = math.degrees(
            math.atan2((y2 - y1), x2 - x1)
        )
        
        if angle > 90: 
            angle = 450 - angle
        else:
            angle = 90 - angle

        return angle

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
        if self.angle_btwn(angle) > VIEW_CONE:
            #print(self.angle_btwn_direction(angle))
            self.log.info(
                f"Angle between frame and coordinates is {self.angle_btwn(angle)}, > {VIEW_CONE}"
            )
            return False

        # now, compute distance between self and coordinates
        # if distance is within view distance, pass
        # else, fail
        if self.distance(coordinates) > VIEW_DISTANCE:
            print(self.distance(coordinates))
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
