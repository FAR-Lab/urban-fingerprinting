# FARLAB - UrbanECG 
# Developer: @mattwfranchi 
# Last Edited: 11/09/2023 

# This script houses a class to geocode addresses given a borough code, street name, and two cross streets. 

# Module Imports
import os 
import sys 

import geocoder as gc

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from src.utils.logger import setup_logger

import pandas as pd
import numpy as np

class Geocoder:
    def __init__(self, dataset_path, state='New York'): 
        self.log = setup_logger("Geocoder")
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(self.dataset_path, engine='pyarrow')

        if not self.validate_dataset():
            self.log.error("Invalid dataset.")
            raise Exception("Invalid dataset.")

        self.geocoded = pd.DataFrame()
        self.state = state
        self.preprocess() 
    
    def validate_dataset(self): 
        # Must have following columns: 
        # boro,order_no,main_st,from_st,to_st,sos
        # boro: String 
        # order_no: String
        # main_st: String
        # from_st: String
        # to_st: String
        # sos: String
        # Return True if valid, False if not

        # Check for columns
        print(self.dataset.columns)
        for col in ['boro', 'order_no', 'main_st', 'from_st', 'to_st', 'sos']: 
            if col not in self.dataset.columns: 
                self.log.error(f"{col} not in dataset columns.")
                return False
        
        return True
    
    def preprocess(self): 
        
        self.expand_boro()

        self.dataset['address'] = self.dataset['main_st'].str.title() + ' at ' + self.dataset['from_st'].str.title() +', ' + self.dataset['boro'] + ', ' + self.state

        self.log.success("Preprocessing complete.")

        print(self.dataset.address.sample(n=5).tolist())


    def expand_boro(self): 
        mapping = { 
            'M': 'Manhattan',
            'B': 'The Bronx',
            'K': 'Brooklyn',
            'Q': 'Queens',
            'S': 'Staten Island'
        }
        
        self.dataset['boro'] = self.dataset['boro'].map(mapping)

        self.log.info("Expanded borough codes to full names.")

    def geocode(self, address): 
        g = gc.arcgis(address)
        return g.json
    
    def __call__(self):
        print(self.geocode(self.dataset.address.sample(n=1)))



    

    



