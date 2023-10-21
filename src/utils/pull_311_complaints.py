# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot 
# Last Edited: 10/6/2023

# This script is used to pull an up-to-date version of OpenData's 311 Complaints Dataset, in GEOJSON format -- should update with time. 

# Imports
import requests
import logging
import os
from tqdm import tqdm

def pull_311_complaints():
    """
    Pull the 311 Complaints dataset from OpenData. 
    """
    # Get the 311 Complaints dataset

    # Make sure data/geo directory exists
    if not os.path.exists("../../data/geo"):
        os.makedirs("../../data/geo")

    logging.info("Pulling 311 Complaints dataset from OpenData.")
    filters = {
        'created_date': 
    }
    r = requests.get("https://data.cityofnewyork.us/resource/erm2-nwe9.geojson", stream=True)
    total_size_in_bytes= int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open("../../data/geo/311_complaints.geojson", "wb") as file:
        for data in r.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logging.error("ERROR, something went wrong")
    logging.info("311 Complaints dataset pulled from OpenData.")

    
    # Print message
    logging.info("311 Complaints dataset saved to disk.")

if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    # Pull 311 Complaints dataset
    pull_311_complaints()