# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 10/02/2023

# This file is used to test/sandbox the functionality of the Graph class.

# Import the Graph class from Graph.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)
from src.analysis.graph_comparative import G

import datetime

# all days of coverage pulled in august with coverage > 0.95
augDoCs = [
    "2023-08-10",
    "2023-08-11",
    "2023-08-12",
    "2023-08-13",
    "2023-08-14",
    "2023-08-17",
    "2023-08-18",
    "2023-08-20",
    "2023-08-21",
    "2023-08-22",
    "2023-08-23",
    "2023-08-24",
    "2023-08-28",
    "2023-08-29",
    "2023-08-30",
    "2023-08-31",
]
# tunable subset of days of coverage for quicker testing
augDoCs_subset = augDoCs[0:1]

flood = ["2023-09-29"]

oct_DoCs = [
    "2023-10-22",
    "2023-10-23",
    "2023-10-24",
    "2023-10-25",
    "2023-10-26",
    "2023-10-27",
]

# parent directory with 'YYYY-MM-DD' subdirectories of frames
FRAMES_DIR = "/share/ju/nexar_data/2023"
# path to graphml file with graph of nyc
GRAPHML_DIR = "/share/ju/urbanECG/data/geo/nyc.graphml"

# flag to use subset of days of coverage
SUBSET_FLAG = False

if __name__ == "__main__":
    DoCs_1 = ["2023-09-29"]
    DoCs_0 = ["2023-10-21", "2023-10-22", "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27"]
    #DoCs_0 = DoCs_0[:2]
    #DoCs = augDoCs 
    graph = G(FRAMES_DIR, GRAPHML_DIR, crop=True, crop_id='862a10747ffffff')
    #graph = G(FRAMES_DIR, GRAPHML_DIR)
    graph.toggle_latex_font()
    
    for day in DoCs_0:
        graph.init_day_of_coverage(day, 0)
    for day in DoCs_1:
        graph.init_day_of_coverage(day, 1)

    graph.density_over_time_of_day_gif(
        DoCs_0, DoCs_1,
        (
            datetime.datetime(2023, 10, 29, 0, 0, 0),
            datetime.datetime(2023, 10, 29, 23, 59, 59),
        ),
        2,
        delta="15min",
        car_offset=True,
    )
