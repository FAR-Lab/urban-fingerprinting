# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 12/02/2023

# This script houses a class to take a results dataframe in the below format and convert it into a format importable by Label Studio. 

# Dataframe Format:
# | image_path | img_id | custom_field_1 | custom_field_2 | ... | custom_field_n |

# Module Imports
import os 
import sys 
import pandas as pd 
import json

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)
sys.path.append(   
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
)

from src.utils.logger import setup_logger


class DF_2_LS:
    def __init__(self): 
        self.log = setup_logger("JSON to Label Studio")
        self.log.setLevel("INFO")

        self.needed_columns = [
            "img_path",
        ]

        self.df = pd.DataFrame()

    def load_df_from_csv(self, csv_path, header=None, random_sample_pct=1):
        """
        Loads a dataframe from a csv file.
        """
        

        self.df = pd.read_csv(csv_path, header=header)
        self.log.info(f"Columns in dataframe: {self.df.columns}")

        self.df = self.df.sample(frac=random_sample_pct)

        # shuffle dataframe
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.log.success(f"Loaded dataframe from {csv_path}.")


    def convert(self, output_path):
        """
        Converts a json file to a format importable by Label Studio.
        """
        # make sure df is a dataframe
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("df must be a pandas dataframe.")
        
        # make sure df has the correct columns
        if not all([c in self.df.columns for c in self.needed_columns]):
            raise ValueError(f"df must have the following columns: {self.needed_columns}")
        
        

        ls_data = []

        # map image_path to image with f'/data/local-files/?d={d["image"]}.jpg',
        self.df["image"] = self.df["img_path"].apply(lambda x: f"/data/local-files/?d={x}")

        for idx, row in self.df.iterrows():

            # turn df columns into a dictionary
            d = row.to_dict()

            

            # remove image_path from dictionary
            del d["img_path"]
            

            d["labels"] = ""

            d["id"] = idx

            ls_data.append(d)

        with open(output_path, "w") as f:
            json.dump(ls_data, f)

        self.log.info(f"Converted dataframe to label studio annotation files in {output_path}.")

    
if __name__ == "__main__":
    df_path = sys.argv[1]
    output_path = sys.argv[2]

    converter = DF_2_LS()
    converter.load_df_from_csv(df_path, header=0)
    converter.convert(output_path)