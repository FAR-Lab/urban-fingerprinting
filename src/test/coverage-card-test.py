import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.analysis.DayOfCoverage import DayOfCoverage
from src.visualization.CoverageCard import CoverageCard

import datetime

if __name__ == '__main__':
    # create list of DayOfCoverage objects
    days_of_coverage = [
        DayOfCoverage('2019-01-01'),
        DayOfCoverage('2019-01-02'),
        DayOfCoverage('2019-01-03'),
        DayOfCoverage('2019-01-04'),
        DayOfCoverage('2019-01-05'),
    ]


    # create CoverageCard object
    card = CoverageCard(days_of_coverage)
    card.latex(True)
    # plot
    card.plot()