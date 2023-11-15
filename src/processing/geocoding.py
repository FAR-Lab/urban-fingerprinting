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

from user.params.io import INSTALL_DIR, PROJECT_NAME

from src.utils.logger import setup_logger

import pandas as pd
import numpy as np

from tqdm import tqdm

class Geocoder:
    def __init__(self, dataset_path, state='New York'): 
        self.log = setup_logger("Geocoder")
        self.log.setLevel("INFO")
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(self.dataset_path, engine='pyarrow')

        if not self.validate_dataset():
            self.log.error("Invalid dataset.")
            raise Exception("Invalid dataset.")

        self.intersections_1 = []
        self.intersections_2 = []
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

        self.dataset['intersection_1'] = self.dataset['main_st'].str.title() + ' at ' + self.dataset['from_st'].str.title() + ', ' + self.dataset['boro'] + ', ' + self.state
        self.dataset['intersection_2'] = self.dataset['main_st'].str.title() + ' at ' + self.dataset['to_st'].str.title() + ', ' + self.dataset['boro'] + ', ' + self.state

        self.log.success("Preprocessing complete.")


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
        for intersection in tqdm(self.dataset.intersection_1.tolist(), desc="Geocoding intersection 1"): 
            self.intersections_1.append(self.geocode(intersection))

        for intersection in tqdm(self.dataset.intersection_2.tolist(), desc="Geocoding intersection 2"):
            self.intersections_2.append(self.geocode(intersection))

        self.intersections_1 = pd.json_normalize(self.intersections_1)
        self.intersections_2 = pd.json_normalize(self.intersections_2)

        os.makedirs(f'{INSTALL_DIR}/{PROJECT_NAME}/geocoding', exist_ok=True)
        self.intersections_1.to_csv(f'{INSTALL_DIR}/{PROJECT_NAME}/geocoding/intersections_1.csv')
        self.intersections_2.to_csv(f'{INSTALL_DIR}/{PROJECT_NAME}/geocoding/intersections_2.csv')

        self.dataset = self.dataset.reset_index(drop=True)
        self.intersections_1 = self.intersections_1.reset_index(drop=True)
        self.intersections_2 = self.intersections_2.reset_index(drop=True)

        # add '_1' suffix to all columns in intersections_1
        self.intersections_1.columns = [str(col) + '_1' for col in self.intersections_1.columns]

        # add '_2' suffix to all columns in intersections_2
        self.intersections_2.columns = [str(col) + '_2' for col in self.intersections_2.columns]

        self.dataset = pd.concat([self.dataset, self.intersections_1, self.intersections_2], axis=1)

        # pretty-log distribution of sum(confidence_1, confidence_2)
        confidence = self.dataset['confidence_1'] * 0.5 + self.dataset['confidence_2'] * 0.5
        self.log.info(f"Confidence Distribution:\n{confidence.describe().to_string()}")

        score = self.dataset['score_1'] * 0.5 + self.dataset['score_2'] * 0.5
        self.log.info(f"Score Distribution:\n{score.describe().to_string()}")

        self.dataset.to_csv(f'{INSTALL_DIR}/{PROJECT_NAME}/geocoding/geocoded_dataset.csv', index=False)

        self.log.success("Geocoding complete.")
        



    

    



