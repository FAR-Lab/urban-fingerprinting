import osmnx as ox 
import logging 
import os

def pull_nyc_graph(): 
    # Get the graph of New York City
    logging.info("Pulling graph of NYC from OSM.")
    G = ox.graph_from_place("New York City, New York, USA", network_type="drive")
    logging.info("Graph of NYC pulled from OSM.")
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