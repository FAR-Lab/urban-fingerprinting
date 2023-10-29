import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)

from src.analysis.day_of_coverage import DayOfCoverage
from src.visualization.annotation_viewer import AnnotationViewer

import datetime

if __name__ == "__main__":
    # create CoverageCard object
    viz = AnnotationViewer("2023-08-18")

    viz()
