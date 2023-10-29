# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 09/14/2023

# This script is used to pull the graph of New York City from OpenStreetMap -- should update with time.

# Imports
import osmnx as ox
import logging
import os


def pull_nyc_graph():
    """
    Pull the graph of New York City from OpenStreetMap.
    """
    # Get the graph of New York City
    logging.info("Pulling graph of NYC from OSM.")
    G = ox.graph_from_place(
        "New York City, New York, USA", network_type="drive"
    )
    logging.info("Graph of NYC pulled from OSM.")

    # Project graph to EPSG:2263
    logging.info("Projecting graph of NYC to EPSG:2263.")
    G = ox.projection.project_graph(G, to_crs="EPSG:2263")
    logging.info("Graph of NYC projected to EPSG:2263.")

    # Make sure data/geo directory exists
    if not os.path.exists("../../data/geo"):
        os.makedirs("../../data/geo")
    # Save graph to disk
    ox.save_graphml(G, filepath="../../data/geo/nyc.graphml")
    # Print message
    logging.info("Updated graph of NYC saved to disk.")


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(level=logging.INFO)
    # Pull NYC graph
    pull_nyc_graph()
