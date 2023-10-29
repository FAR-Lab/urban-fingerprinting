# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 10/02/2023

# This file is used to test/sandbox the functionality of the Graph class.

# Import the Graph class from Graph.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.graph import G

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

# parent directory with 'YYYY-MM-DD' subdirectories of frames
FRAMES_DIR = "/share/ju/nexar_data/nexar-scraper"
# path to graphml file with graph of nyc
GRAPHML_DIR = "/share/ju/urbanECG/data/geo/nyc.graphml"

# flag to use subset of days of coverage
SUBSET_FLAG = True

if __name__ == "__main__":
    if SUBSET_FLAG:
        DoCs = augDoCs_subset
    else:
        DoCs = augDoCs

    graph = G(FRAMES_DIR, GRAPHML_DIR)
    graph.toggle_latex_font()
    for day in DoCs:
        try:
            graph.init_day_of_coverage(day)
        except Exception as e:
            graph.log.error(f"Error in {day}: {e}")

    graph.density_over_time_of_day_gif(
        DoCs,
        (
            datetime.datetime(2023, 8, 10, 0, 0, 0),
            datetime.datetime(2023, 8, 10, 23, 59, 59),
        ),
        2,
        delta="5min",
        car_offset=True,
    )
