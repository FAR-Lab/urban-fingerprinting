# Import the Graph class from Graph.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import matplotlib.pyplot as plt
from analysis.graph import G


def coverage_growth_over_time(G):
    """
    Calculate the coverage growth over time, proxied by number of roads with > 1 image.
    """
    G.log.info("Calculating coverage growth over time")

    roads_covered = []
    roads_covered_growth = {}

    for DoC in G.days_of_coverage:
        if len(roads_covered) == 0:
            roads_covered = (
                DoC.nearest_edges.groupby(["u", "v"])
                .size()
                .reset_index(name=f"count_{DoC.date}")
            )
        else:
            roads_covered = roads_covered.merge(
                DoC.nearest_edges.groupby(["u", "v"])
                .size()
                .reset_index(name=f"count_{DoC.date}"),
                on=["u", "v"],
                how="outer",
            )

        G.log.info(
            f"After adding {DoC.date}, {len(roads_covered.index)} roads covered"
        )
        roads_covered_growth[DoC.date] = len(roads_covered.index)

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("../../output/coverage_growth", exist_ok=True)

    df = pd.DataFrame()
    df["roads_covered"] = roads_covered_growth

    G.log.info(
        f"Saving coverage growth to output/coverage_growth/{len(G.days_of_coverage)}_days_of_coverage_{timestamp}.csv..."
    )
    df.to_csv(
        f"../../output/coverage_growth/{len(G.days_of_coverage)}_days_of_coverage_{timestamp}.csv"
    )

    return df


def plot_coverage_growth_over_time(data_frame, graph):
    """
    Plot the coverage growth over time.

    Args:
        data_frame (pandas.DataFrame): The data frame containing the coverage growth data.
        graph (analysis.graph.G): The graph object.
    """
    # Create a figure and axis object.
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot roads covered over time, index as date.
    ax.plot(
        data_frame.index,
        data_frame.roads_covered,
        color="b",
        label="Roads Covered",
    )

    # Add dashed horizontal line at 100% coverage.
    ax.axhline(
        y=len(graph.gdf_edges),
        color="r",
        linestyle="--",
        label="100% Coverage",
    )

    # Set the axis labels and title.
    ax.set_xlabel("After Adding Day of Coverage")
    ax.set_ylabel("Roads Covered")
    ax.set_title("Roads Covered With Successive Days of Coverage")

    # Create the output directories.
    os.makedirs("../../output/coverage_growth", exist_ok=True)
    os.makedirs("../../output/coverage_growth/figures", exist_ok=True)

    # Get the current timestamp.
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Tight layout and small margins.
    plt.tight_layout()
    plt.margins(0.05)

    # Add the legend and save the figure.
    plt.legend()
    plt.savefig(
        f"../../output/coverage_growth/figures/{len(graph.days_of_coverage)}_days_of_coverage_{timestamp}.png"
    )


def map_coverage_growth_over_time():
    """
    Map the coverage growth over time.
    """
    pass


if __name__ == "__main__":
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
    augDoCs_subset = augDoCs[0:3]

    # parent directory with 'YYYY-MM-DD' subdirectories of frames
    FRAMES_DIR = "/share/ju/nexar_data/nexar-scraper"
    # path to graphml file with graph of nyc
    GRAPHML_DIR = "/share/ju/urbanECG/data/geo/nyc.graphml"

    SUBSET_FLAG = False

    if SUBSET_FLAG:
        DoCs = augDoCs_subset
    else:
        DoCs = augDoCs

    graph = G(FRAMES_DIR, GRAPHML_DIR)
    for day in DoCs:
        try:
            graph.init_day_of_coverage(day)
        except Exception as e:
            graph.log.error(f"Error in {day}: {e}")

    df = coverage_growth_over_time(graph)
    plot_coverage_growth_over_time(df, graph)
