import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.analysis.DayOfCoverage import DayOfCoverage
from src.visualization.Annotation import AnnotationViewer

import datetime

if __name__ == '__main__':

    # create CoverageCard object
    viz = AnnotationViewer("2023-08-18")

    viz()