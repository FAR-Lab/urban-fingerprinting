# FARLAB - UrbanECG
# Developer: @mattwfranchi
# Last Modified: 10/30/2023

# This script houses tests to ensure proper functionality of functions within src/processing/geometric_utils.py


import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from src.processing.geometric_utils import *


test_frames_by_day = {
    "2023-09-29": {
        "1cec3cb99eac43db27e1b5fd0606e107": {
            "gps_info.longitude": -73.953137,
            "gps_info.latitude": 40.671425,
            "HEADING": "NORTH",
        }
    }
}

correct_results = { 
    "2023-09-29": {
        "1cec3cb99eac43db27e1b5fd0606e107": {
            "x": 997249.7770191849,
            "y": 183898.28552839367,
            "angle": 
        }
    }
}
test_frames_md = pd.DataFrame()
for day in test_frames_by_day:
    test_frames = test_frames_by_day[day]

    day_md = pd.read_csv(
        "../../output/df/{}/md.csv".format(day), engine="pyarrow", index_col=0
    )
    day_md = day_md[day_md[IMG_ID].isin(test_frames)]
    test_frames_md = pd.concat([test_frames_md, day_md])

print(test_frames_md)

test_frames = [Frame(row) for _, row in test_frames_md.iterrows()]

