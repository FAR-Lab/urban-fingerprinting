# FARLAB - UrbanECG 
# Developer: @mattwfranchi, with help from GitHub CoPilot 
# Last Edited: 11/28/2023

# This script houses a class to parse the results of the OpenAI GPT4-Vision API.

# Module Imports
import os
import sys

#from sympy import N

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
)

from src.utils.logger import setup_logger

import json
from glob import glob

from tqdm import tqdm 

import pandas as pd 

class OAI_VP_Parser: 
    def __init__(self, results_json, outfile="parsed_results") -> None: 

        self.log = setup_logger("OAI Vision Parser")

        self.results_json = results_json
        self.outfile = outfile

        self.results = self._load_results()

        self.log.info("Loaded results json.")

        self.df = self._make_df(to_merge="/share/ju/urbanECG/output/street_flooding/train_set_gt_pt2_w_clipvitg_results_flooded_road.csv", merge_on="img_path")

        self.log.info("Created dataframe.")

        self._write_df()
        self._write_json()

        self.log.success("Wrote dataframe to csv.")

    # load results according to the json format specified above. want to grab the json inside 'content' 
    def _load_results(self): 
        with open(self.results_json, "r") as f: 
            results = f.readlines()
            #print(results)
            # wrap results in outer brackets to make it a valid json list
            results = "[" + ",".join(results) + "]"

            results = json.loads(results)
            
            # take out rows with 'null' 
            results = [result for result in results if result != None]


            # only keep the json inside 'choices' 
            results = [result["choices"][0]["message"]["content"] for result in results]
            # remove the '''json
            results = [result[7:-3] for result in results]
            # remove all \n and {} 
            results = [result.replace("\n", "").replace("{", "").replace("}", "") for result in results]
            # remove all whitespace outside of the key-value pairs
            results = [result.strip() for result in results]

            # turn dict into list of tuples
            results = [result.split(":") for result in results]
            
            # remove all double quotes 
            results = [[result[0].replace("\"", ""), result[1].replace("\"", "")] for result in results]

        return results 
            
    def _make_df(self, suffix=True, prefix=True, to_prepend="", to_merge=None, merge_on=None):
        df = pd.DataFrame(self.results, columns=["img_path", "gpt_classification"])

        if suffix: 
            # get the id from the image column
            df["id"] = df["img_path"].apply(lambda x: x.split("/")[-1].split(".")[0])

        if prefix: 
            # prepend the image column with the path to the image
            df["img_path"] = to_prepend + df["img_path"]

        if to_merge and merge_on: 
            # read in the csv to merge on
            to_merge = pd.read_csv(to_merge)
            # merge on the id column
            df = df.merge(to_merge, on=merge_on, how='inner')

        return df



    def _write_df(self):
        # write to json 
        self.df.to_csv(self.outfile+".csv", index=False)

    def _write_json(self):
        # write to json 
        self.df.to_json(self.outfile+".json", orient="records", indent=4)


if __name__ == '__main__':
    results_json = sys.argv[1]
    try:
        outfile = sys.argv[2]
    except IndexError:
        outfile = "parsed_results" 

    parser = OAI_VP_Parser(results_json, outfile)

