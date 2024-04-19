# Import the Graph class from Graph.py
import sys
import os
import geopandas as gpd 
import h3pandas 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import user.params.io as IO 

import pandas as pd
import matplotlib.pyplot as plt
from analysis.graph import G

import h3.api.basic_int as h3

from utils.toggle_project import set_project, reset_project 
set_project("street_flooding") 

from importlib import reload 
reload(IO) 

from user.params.io import OUTPUT_DIR, PROJECT_NAME

# enable latex plotting 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



def coverage_growth_over_time(G, h3_res=10):
    """
    Calculate the coverage growth over time, proxied by number of roads with > 1 image.
    """
    G.log.info("Calculating coverage growth over time")

    cells_covered = []
    cells_covered_growth = {}

    for DoC in G.days_of_coverage:

        DoC.frames_data["h3"] = DoC.frames_data.apply(
            lambda row: h3.h3_to_parent(row["h3_index_res12"], h3_res), axis=1)

        if len(cells_covered) == 0:
            cells_covered = (
                DoC.frames_data.groupby(["h3"])
                .size()
                .reset_index(name=f"count_{DoC.date}")
            )  
            cells_covered['first_covered'] = DoC.date
        else:
            cells_covered = cells_covered.merge(
                DoC.frames_data.groupby(["h3"])
                .size()
                .reset_index(name=f"count_{DoC.date}"),
                on=["h3"],
                how="outer",
            )
            cells_covered['first_covered'] = cells_covered['first_covered'].fillna(DoC.date)

        G.log.info(
            f"After adding {DoC.date}, {len(cells_covered.index)} cells covered"
        )
        cells_covered_growth[DoC.date] = len(cells_covered.index)

        G.log.info(f"Unique cells covered on {DoC.date}: {len(cells_covered)}")
       

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(f'{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth', exist_ok=True)

    df = pd.DataFrame()
    df["cells_covered"] = cells_covered_growth

    df.to_csv(
        f"{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth/{len(G.days_of_coverage)}_days_of_coverage_{h3_res}_{timestamp}.csv"
    )

    return df, cells_covered


def plot_coverage_growth_over_time(data_frame, graph, h3_res=10):
    """
    Plot the coverage growth over time.

    Args:
        data_frame (pandas.DataFrame): The data frame containing the coverage growth data.
        graph (analysis.graph.G): The graph object.
    """
    # Create a figure and axis object.
    fig, ax = plt.subplots(figsize=(20, 10))

    # get bounds of graph 
    total_bounds = graph.gdf_edges.to_crs('EPSG:4326').total_bounds

    # make polygon of bounds 
    total_bounds = [
        [total_bounds[1], total_bounds[0]],
        [total_bounds[3], total_bounds[0]],
        [total_bounds[3], total_bounds[2]],
        [total_bounds[1], total_bounds[2]],
        [total_bounds[1], total_bounds[0]]
    ]

    # make geojson 
    total_bounds = {
        "type": "Polygon",
        "coordinates": [total_bounds]
    }




    h3_in_G = h3.polyfill(total_bounds, h3_res)

    graph.log.info(f"Number of pre-filtered cells in bounds of graph: {len(h3_in_G)}")

    # add lat/lon to num_h3_in_G
    h3_in_G = pd.DataFrame(h3_in_G, columns=["h3"])
    lat, lon = zip(*h3_in_G.h3.apply(lambda h3_index: h3.h3_to_geo(h3_index)))
    h3_in_G["lat"] = lat
    h3_in_G["lon"] = lon

    graph.log.info("Converted h3 to lat/lon")

    # project to epsg 2263 
    h3_in_G = gpd.GeoDataFrame(h3_in_G, geometry=gpd.points_from_xy(h3_in_G.lon, h3_in_G.lat), crs="EPSG:4326")
    h3_in_G = h3_in_G.to_crs("EPSG:2263")

    graph.log.info("Projected to EPSG:2263")



    # now, remove cells that are > 50 ft from an edge in the graph 
    h3_in_G = gpd.sjoin_nearest(h3_in_G, graph.gdf_edges.to_crs('EPSG:2263'), distance_col="distance", how="left")

    THRES = h3.edge_length(h3_res, unit="m") / 2
    # convert to feet 
    THRES = THRES * 3.28084
    h3_in_G = h3_in_G[h3_in_G['distance'] <= THRES]

    num_h3_in_G = len(h3_in_G)

    graph.log.info(f"Number of cells in bounds of graph within {THRES} ft of an edge: {num_h3_in_G}")

    # Plot cells covered over time, index as date.
    ax.plot(
        data_frame.index,
        data_frame.cells_covered,
        color="b",
        label="cells covered",
    )

    # Add dashed horizontal line at 100% coverage.
    ax.axhline(
        y=num_h3_in_G,
        color="r",
        linestyle="--",
        label="100% Coverage",
    )

    # Set the axis labels and title.
    ax.set_xlabel("After Adding Day of Coverage")
    ax.set_ylabel("cells covered")
    ax.set_title("cells covered With Successive Days of Coverage")

    # Create the output directories.
    os.makedirs(f'{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth/figures', exist_ok=True)

    # Get the current timestamp.
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Tight layout and small margins.
    plt.tight_layout()
    plt.margins(0.05)

    # Add the legend and save the figure.
    plt.legend()
    plt.savefig(
        f"{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth/figures/{len(graph.days_of_coverage)}_{h3_res}_days_of_coverage_{timestamp}.png"
    )





def map_cells_covered(cells_covered, h3_res=10): 
    """
    Map the cells covered over time. 
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    # plot the graph 
    graph.gdf_edges.plot(ax=ax, color="black", linewidth=0.25, alpha=0.5)

    print(cells_covered.head())
    
    # convert h3 column from numpy int to string
    cells_covered['h3'] = cells_covered['h3'].apply(lambda x: h3.h3_to_string(x))

    # plot the cells covered 
    
    # make a geodataframe of the cells covered, using h3 as geometry 
    cells_covered = cells_covered.set_index("h3")
    cells_covered = cells_covered.h3.h3_to_geo_boundary()

    cells_covered = cells_covered.to_crs("EPSG:2263")

    cells_covered.to_csv(f'{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth/{len(graph.days_of_coverage)}_days_of_coverage_{h3_res}_cells_covered.csv')


    cells_covered.plot(ax=ax, color="blue", alpha=0.75)

    os.makedirs(f'{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth/figures', exist_ok=True)
    # save the figure 
    plt.savefig(
        f"{OUTPUT_DIR}/{PROJECT_NAME}/coverage_growth/figures/{len(graph.days_of_coverage)}_{h3_res}_days_of_coverage_map.png"
    )



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

    all_dates = ["2023-08-10", "2023-08-11", "2023-08-12", "2023-08-13", "2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17", "2023-08-18", "2023-08-19", "2023-08-20", "2023-08-21", "2023-08-22", "2023-08-23", "2023-08-24", "2023-08-28", "2023-08-29", "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-03", "2023-09-07", "2023-09-28", "2023-09-29", "2023-10-01", "2023-10-20", "2023-10-21", "2023-10-22", "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-27", "2023-10-28", "2023-10-29", "2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-04", "2023-11-05", "2023-11-06", "2023-11-07", "2023-11-08", "2023-11-09", "2023-11-10", "2023-11-11", "2023-11-12", "2023-11-13", "2023-11-14", "2023-11-15", "2023-11-16", "2023-11-17", "2023-11-18", "2023-11-19", "2023-11-20", "2023-11-21", "2023-11-22", "2023-11-23", "2023-11-24", "2023-11-25", "2023-11-26", "2023-11-27", "2023-11-28", "2023-11-29", "2023-11-30", "2023-12-01", "2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05", "2023-12-17", "2023-12-18", "2023-12-19", "2024-01-10"]
    # tunable subset of days of coverage for quicker testing
    augDoCs_subset = augDoCs[0:4]

    # parent directory with 'YYYY-MM-DD' subdirectories of frames
    FRAMES_DIR = "/share/ju/nexar_data/nexar-scraper"
    # path to graphml file with graph of nyc
    GRAPHML_DIR = "/share/ju/urbanECG/data/geo/nyc.graphml"

    SUBSET_FLAG = False

    if SUBSET_FLAG:
        DoCs = augDoCs_subset
    else:
        DoCs = all_dates

    graph = G(FRAMES_DIR, GRAPHML_DIR, write=True)
    for day in DoCs:
        try:
            graph.init_day_of_coverage(day)
        except Exception as e:
            graph.log.error(f"Error in {day}: {e}")

    for h3_res in [6,7,8,9, 10, 11, 12]:
        df, cells_covered = coverage_growth_over_time(graph, h3_res=h3_res)
        #plot_coverage_growth_over_time(df, graph, h3_res=h3_res)
        map_cells_covered(cells_covered, h3_res=h3_res)