# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/28/2023 

# This script converts a json file to a format importable by Label Studio.

# Module Imports
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)
sys.path.append(   
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
)

from src.utils.logger import setup_logger


class JSON_2_LS:
    def __init__(self): 
        self.log = setup_logger("JSON to Label Studio")
        self.log.setLevel("INFO")

    def convert(self, json_path, output_path):
        """
        Converts a json file to a format importable by Label Studio.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        ls_data = []

        for i, d in enumerate(data):
            ls_data.append(
                {
                    "id": i,
                    "data": {
                        "image": f'/data/local-files/?d={d["image"]}.jpg',
                        "labels": "",
                        "clip_label": d["clip_label"],
                        "clip_score": d["clip_confidence"],
                        "gpt_label": d["gpt_classification"],
                        
                    },
                }
            )

        with open(output_path, "w") as f:
            json.dump(ls_data, f)

        self.log.info(f"Converted {json_path} to {output_path}.")

    
if __name__ == "__main__":
    json_path = sys.argv[1]
    output_path = sys.argv[2]

    converter = JSON_2_LS()
    converter.convert(json_path, output_path)