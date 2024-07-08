# FARLAB - UrbanECG 
# Developer: @mattwfranchi

# This script visualizes geospatial data generated from the coverage-growth-over-time-h3 script.

# Import libraries
from numpy import pad
import pandas as pd
import geopandas as gpd 
from shapely import wkt 

import matplotlib.pyplot as plt 

# enable latex plotting 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


import osmnx as ox


from fire import Fire 

def load_dfs(paths): 
    """Load the dataframes from the paths"""
    dfs = [pd.read_csv(p, engine='pyarrow') for p in paths]

    # convert to geodataframes
    for i, df in enumerate(dfs): 
        dfs[i] = gpd.GeoDataFrame(df, geometry=df['geometry'].apply(wkt.loads))

    return dfs

def map_dfs(dfs, roads_graphml_path, num_doc, labels=None):
    if labels is None:
        labels = [f'DF {i+1}' for i in range(len(dfs))]
    
    # Adjusted retrieval of the colormap, compliant with new Matplotlib versions
    cmap = plt.get_cmap('Blues', len(dfs)+3)

    fig, ax = plt.subplots(figsize=(40, 40))
    for i, df in enumerate(dfs):
        # Select a unique color from the colormap for each DataFrame
        color = cmap(i+3 / (len(dfs)+2))  # Normalized color selection
        # Ensure you're plotting a type that supports legend handles directly
        df.plot(ax=ax, color=color, alpha=0.25)

    # custom legend 
    custom_lines = [plt.Line2D([0], [0], color=cmap(i+3 / (len(dfs)+2)), lw=4) for i in range(len(dfs))]
    ax.legend(custom_lines, labels, fontsize=40, loc='upper right', title='H3 Granularity', title_fontsize=45, shadow=True, fancybox=True, facecolor='white', edgecolor='black')

    ax.set_axis_off()
    #ax.set_title('Coverage Growth Over {} Days of Coverage'.format(num_doc))

    # read roads graphml 
    G = ox.load_graphml(roads_graphml_path)
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    edges.plot(ax=ax, linewidth=0.25, color='gray', alpha=0.5)





    plt.savefig('test.png', dpi=300, bbox_inches='tight', pad_inches=0.01)


if __name__ == "__main__": 
    

    paths = [
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_6_cells_covered.csv",
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_7_cells_covered.csv",
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_8_cells_covered.csv",
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_9_cells_covered.csv",
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_10_cells_covered.csv",
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_11_cells_covered.csv",
        "/share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_12_cells_covered.csv"

    ]

    # get number of DoC from first path 
    num_doc = int(paths[0].split('/')[-1].split('_')[0])

    # assert that all paths have the same number of DoC
    assert all([int(p.split('/')[-1].split('_')[0]) == num_doc for p in paths])

    # extract labels 
    # example: /share/ju/urbanECG/output/street_flooding/coverage_growth/4_days_of_coverage_6_cells_covered.csv -> h3_6
    labels = [f"h3-{int(p.split('/')[-1].split('_')[4])}" for p in paths]


    dfs = load_dfs(paths)

    map_dfs(dfs, "/share/ju/urbanECG/data/geo/nyc.graphml", num_doc, labels)




