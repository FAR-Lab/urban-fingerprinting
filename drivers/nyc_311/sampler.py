# FARLAB - UrbanECG Project
# Developer: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/29/2023

# This script houses a driver for the NYC311Sampler class.


import os 
import sys 

sys.path.append(os.path.abspath(os.path.join("../")))
sys.path.append(os.path.abspath(os.path.join("../..")))
sys.path.append(os.path.abspath(os.path.join("../../../")))

from src.processing.nyc_311.sample import NYC311Sampler
from user.params.data import *

from fire import Fire

if __name__ == '__main__':
    sampler = NYC311Sampler(path_to_311_complaints_data=NYC_311_DATA_CSV_PATH ,write_mode=True)

    # use fire to sample from the 311 complaints data
    Fire(sampler.samples_for_all_col_values)
