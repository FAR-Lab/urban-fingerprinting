# FARLAB - UrbanECG Project
# Developer: @mattwfranchi, with help from GitHub Copilot
# Last Edited: 11/05/2023

# This file contains a function decorator to save the output of a function to a csv file.

# Import the necessary packages
import os
import sys
import pandas as pd 
import logging 
import time
import datetime


def to_csv(func):

    def wrap_func(*args, **kwargs):

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            result.to_csv(f'{now}_{func.__name__}.csv', index=False)
        else:
            logging.error(f'Function {func.__name__!r} did not return a pandas DataFrame.')
        return result

    return wrap_func