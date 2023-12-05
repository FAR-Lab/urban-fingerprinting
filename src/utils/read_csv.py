# FARLAB - UrbanECG
# Developer: @mattwfranchi
# Last Edited: 12/05/2023

# This script extends the functionality of the pandas.read_csv() function
# with convenience manipulations for working with Nexar frames metadata. 

# Import libraries
import os
import sys
import inspect
import re

import pandas as pd
import numpy as np


# append ../ to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# append ../../ to sys path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# import all fields from params/data.py by name
from user.params.data import LONGITUDE_COL, LATITUDE_COL, TIME_COL, ORIENTATION_COL, \
DIRECTION_COL, COORD_CRS, PROJ_CRS, TZ, IMG_ID, QUALITY, LIGHT, STOCK_H3

# import logger
from src.utils.logger import setup_logger


# Define function with that takes generic args and kwargs of the pandas.read_csv() function
def read_csv(*args, **kwargs):

    # Setup logger, should share state with all read_csv() calls
    log = setup_logger('Nexar Metadata Reader')
    log.setLevel('INFO')

    # READING IN CSV

    # get function signature of stock pd.read_csv() function 
    pd_signature = str(inspect.signature(pd.read_csv))
    pd_signature = re.findall(r'(\w+):', pd_signature)

    # read in csv file with only kwargs that are valid for pd.read_csv()
    df = pd.read_csv(*args, **{k: v for k, v in kwargs.items() if k in pd_signature})

    log.info('Read in CSV file with shape: {}'.format(df.shape))

    # set h3 index column to numpy int64
    df[STOCK_H3] = df[STOCK_H3].astype(np.int64)

    # daytime_only and nighttime_only are mutually exclusive
    if kwargs.get('daytime_only', False) and kwargs.get('nighttime_only', False):
        raise ValueError('daytime_only and nighttime_only are mutually exclusive')

    # check for daytime_only kwarg
    if kwargs.get('daytime_only', False):
        len_before = len(df.index)
        # only keep images taken during the day
        df = df[df[LIGHT] == 'DAYLIGHT']
        log.info('Removed {} nighttime images'.format(len_before - len(df.index)))
        
    
    # check for nighttime_only kwarg
    if kwargs.get('nighttime_only', False):
        len_before = len(df.index)
        # only keep images taken at night
        df = df[df[LIGHT] == 'NIGHTTIME']
        log.info('Removed {} daytime images'.format(len_before - len(df.index)))

    
    # check for quality_thres kwarg
    if kwargs.get('quality_thres', False):
        len_before = len(df.index)
        # only keep rows with quality above quality_thres
        df = df[df[QUALITY] >= kwargs.get('quality_thres')]
        log.info('Removed {} images with quality below {}'.format(len_before - len(df.index), kwargs.get('quality_thres')))

    
    # check for directions kwarg
    if kwargs.get('directions', False):
        len_before = len(df.index)
        # make sure directions is a list
        if not isinstance(kwargs.get('directions'), list):
            raise ValueError('directions kwarg must be a list')
        try:
            # only keep rows with direction in directions list
            df = df[df[DIRECTION_COL].isin(kwargs.get('directions'))]
        except KeyError as e: 
            raise e
        
        log.info('Removed {} images with direction not in {}'.format(len_before - len(df.index), kwargs.get('directions')))

    log.success('Finished reading CSV file with final shape: {}'.format(df.shape))
        
    return df