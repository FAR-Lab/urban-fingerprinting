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
from src.visualization.coverage_card import CoverageCard

import datetime

if __name__ == "__main__":
    # create list of DayOfCoverage objects
    days_of_coverage = [
        DayOfCoverage("2019-01-01"),
        DayOfCoverage("2019-01-02"),
        DayOfCoverage("2019-01-03"),
        DayOfCoverage("2019-01-04"),
        DayOfCoverage("2019-01-05"),
    ]

    days_of_coverage_month = [
        "2023-09-01",
        "2023-09-02",
        "2023-09-11",
        "2023-09-12",
        "2023-09-13",
        "2023-09-19",
        "2023-09-20",
    ]

    days_of_coverage_year = [
        "2023-07-16",
        "2023-09-01",
        "2023-09-02",
        "2023-09-11",
        "2023-09-12",
        "2023-09-13",
        "2023-09-19",
        "2023-09-20",
        "2023-08-02",
        "2023-10-03",
        "2023-10-01",
        "2023-10-05",
    ]

    # create CoverageCard object
    card = CoverageCard(
        days_of_coverage_year, attached_plot="flooding-test.gif"
    )
    card.latex(True)
    # plot
    card.plot()
