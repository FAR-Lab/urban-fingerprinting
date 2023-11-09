# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/09/2023 

# This script houses a driver to geocode a dataset of addresses.

# Module Imports
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from src.processing.geocoding import Geocoder 


if __name__ == '__main__': 
    g = Geocoder("../../data/coords/parking_regulations.csv")
    g()