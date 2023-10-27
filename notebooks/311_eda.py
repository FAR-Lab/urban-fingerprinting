# %%
import pandas as pd 
import geopandas as gpd 
import numpy as np 

import matplotlib.pyplot as plt
import osmnx as ox

from glob import glob
from tqdm import tqdm

# %%
# turn on latex plotting 
plt.rc('text', usetex=False)

# %%
import sys 
import os
# add parent directory to path
sys.path.insert(1, os.path.join(sys.path[0], '..'))


# %%
from user.params.data import *

from src.processing.geometricUtils import Frame, Perspective

# %%
nyc_roads = ox.io.load_graphml('../data/geo/nyc.graphml')

# %%
nyc_roads_gdf = ox.graph_to_gdfs(nyc_roads, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
nyc_roads_gdf = nyc_roads_gdf.to_crs(PROJ_CRS)

# %%
nyc_ct = gpd.read_file('../data/geo/nyc_ct/nyct2020.shp')
nyc_ct = nyc_ct.to_crs(PROJ_CRS)

# %%
nyc_311 = pd.read_csv('../data/geo/311_Service_Requests_from_2010_to_Present.csv', engine='pyarrow')

# %%
nyc_311.info()

# %%
# make output/eda directory if it doesn't exist
if not os.path.exists('../output/eda'):
    os.makedirs('../output/eda')
nyc_311['Complaint Type'].value_counts().sort_values(ascending=False).to_csv('../output/eda/311_complaint_types.csv')

# %%
# make output/eda directory if it doesn't exist
if not os.path.exists('../output/eda'):
    os.makedirs('../output/eda')
nyc_311['Descriptor'].value_counts().sort_values(ascending=False).to_csv('../output/eda/311_descriptor_counts.csv')

# %%
list(nyc_311['Complaint Type'].unique())

# %%
nyc_311['Created Date'] = pd.to_datetime(nyc_311['Created Date'])

# %%
nyc_311_gdf = gpd.GeoDataFrame(nyc_311, geometry=gpd.points_from_xy(nyc_311.Longitude, nyc_311.Latitude), crs='EPSG:4326')
nyc_311_gdf = nyc_311_gdf.to_crs(PROJ_CRS)

# %%
fig, ax = plt.subplots(figsize=(20,20))
nyc_ct.plot(ax=ax, color='grey', linewidth=0.5)
nyc_311_gdf.plot(ax=ax, column='Agency', legend=True, markersize=5)

# %%
# filter on rows created on september 29th 2023
sep29 = nyc_311_gdf[nyc_311_gdf['Created Date'].dt.date == pd.to_datetime('2023-09-29').date()]

# %%
sep29['Descriptor'].value_counts()

# %%
flooding_descs = ['Catch Basin Clogged/Flooding (Use Comments) (SC)', 'Sewer Backup (Use Comments) (SA)', 'Street Flooding (SJ)', 'Highway Flooding (SH)', 'SLOW LEAK', 'Leak (Use Comments) (WA2)', 'Hydrant Leaking (WC1)', 'Sewage Leak', 'HEAVY FLOW', 'Overflowing', 'Manhole Overflow (Use Comments) (SA1)', 'WATER SUPPLY', 'Dirty Water (WE)', 'Failure To Retain Water/Improper Drainage- (LL103/89)', 'Excessive Water In Basement (WEFB)', 'Puddle on Sidewalk', 'Puddle on Driveway'] 

# %%
flooding = sep29[sep29['Descriptor'].isin(flooding_descs)]

# %%
#fig, ax = plt.subplots(figsize=(15,15))
#nyc_ct.plot(ax=ax, edgecolor='grey', color='w', linewidth=0.5)
#flooding.plot(ax=ax, column='Descriptor', markersize=5, legend=True, legend_kwds={'loc': 'upper left', 'title': 'Type of 311 Complaint', 'borderpad':2, 'labelspacing': 1, 'fontsize': 14})

# increase legend font size 
#leg = ax.get_legend() 

# increase legend title size 
#leg.get_title().set_fontsize('20')




#plt.axis('off')
# tight layout
#plt.tight_layout()

#plt.savefig('../output/plots/311_flooding.png', dpi=400, bbox_inches='tight')


# %%
flooding.to_csv('../data/coords/sep29_flooding.csv', index=False)

# %%
#nyc_ct_w_flooding_reports = nyc_ct.merge(flooding.sjoin_nearest(nyc_ct, how='left').groupby('BoroCT2020').size().to_frame(), on='BoroCT2020', how='left').fillna(0)

# %%
#nyc_ct.merge(flooding.sjoin_nearest(nyc_ct, how='left').groupby('BoroCT2020').size().to_frame(), on='BoroCT2020', how='left').fillna(0).describe()

# %%
# rename '0' column to 'flood_reports'
#nyc_ct_w_flooding_reports.rename(columns={0: 'flood_reports'}, inplace=True)

# %%
#value_counts = nyc_ct_w_flooding_reports.flood_reports.value_counts().sort_index()

# %%
#fig, ax = plt.subplots(figsize=(10,10))
# scatter plot with trendline
#ax.scatter(value_counts.index, value_counts.values, s=50)
#ax.set_xlabel('Number of Flooding Reports', fontsize=20)
#ax.set_ylabel('Number of Census Tracts', fontsize=20)
#ax.set_title('Distribution of Flooding Reports by Census Tract, Sep 29 2023', fontsize=25)

# add box with percent of census tracts with 0 flooding reports
#ax.text(0.09, 0.95, 'Percent of Census Tracts with 0 Flooding Reports: {:.2f}%'.format(value_counts[0]/len(nyc_ct_w_flooding_reports)*100), transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# add box with percent of census tracts with > 5 flooding reports
#ax.text(0.175, 0.125, 'Percent of Census Tracts with g.e.t 5 Flooding Reports: {:.2f}%'.format(value_counts[value_counts.index >= 5].sum()/len(nyc_ct_w_flooding_reports)*100), transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
# red dashed vertical line at x=5 
#ax.axvline(x=5, linestyle='--', color='red')


#plt.savefig('../output/plots/311_flooding_dist.png', dpi=400, bbox_inches='tight')


# %%
#nyc_ct_w_flooding_reports.flood_reports.describe()

# %%
samples = glob('./output/*/')
samples = [x for x in samples if 'depicts' in x]

print(samples)

# %%
sep29_md = pd.concat([pd.read_csv(x, engine='pyarrow') for x in tqdm(glob("../../nexar_data/2023/2023-09-29/*/metadata.csv"))])

# %%


# %%
for sample in samples: 
    
    imgs = glob(f"{sample}/*.jpg")

    frame_ids = [img.split('/')[-1].split('.')[0] for img in imgs]

    frames = sep29_md[sep29_md['frame_id'].isin(frame_ids)]

    frames = gpd.GeoDataFrame(frames, geometry=gpd.points_from_xy(frames['gps_info.longitude'], frames['gps_info.latitude']), crs='EPSG:4326')
    frames = frames.to_crs(PROJ_CRS)

    #print(frames)
    



    fig, ax = plt.subplots(figsize=(30,30))

    nyc_roads_gdf.plot(ax=ax, color='grey', linewidth=0.5, alpha=0.25, zorder=0)
    
    nyc_ct.plot(ax=ax, edgecolor='black', color='w', linewidth=0.5, alpha=0.2, zorder=1)
    print(sample.rsplit('_',6)[2])
    flooding_sample = flooding[flooding['Descriptor'] == sample.rsplit('_',6)[2]]
    flooding_sample = gpd.GeoDataFrame(flooding_sample, geometry=gpd.points_from_xy(flooding_sample.Longitude, flooding_sample.Latitude), crs='EPSG:4326')
    flooding_sample = flooding_sample.to_crs(PROJ_CRS)

    flooding_sample.plot(ax=ax, color='g', markersize=7.5,legend=False, alpha=0.4, zorder=2)

    # plot each frame in frame as a cone, with direction facing the camera heading of the frame 
    for idx, frame in tqdm(frames.iterrows(), desc='plotting cones...'):
        # get the camera heading of the frame 
        heading = Frame(frame).angle_from_direction()

        # add 90 degrees to heading to account for plt arrow direction
        heading += 90

        # get the geometry of the frame 
        geom = frame['geometry']
        # get the x and y coordinates of the frame 
        x, y = geom.x, geom.y
        
        radius = 500
        # project x, y in same direction as heading, with true north as 0 degrees
        x1 = x + radius * np.cos(np.radians(heading))
        y1 = y + radius * np.sin(np.radians(heading))

        # get the difference between the projected x, y and the original x, y
        dx = x1 - x
        dy = y1 - y

        

        ax.arrow(x, y, dx, dy, head_width=1, head_length=0.5, fc='k', ec='k', linewidth=0.6, alpha=0.5, zorder=3)
        # add text with camera heading 
        #ax.text(x1, y1, f'{heading:.2f}', fontsize=10, zorder=3)
    



    frames.plot(ax=ax, color='b', markersize=1, legend=False, zorder=4)

    ax.legend(['NYC Roads', 'Flooding Reports', 'Nexar Images'], loc='upper left', fontsize=15)

    ax.set_title(f'Sample v. Reports: {sample.rsplit("_",6)[2]}', fontsize=25)

    os.makedirs('../output/plots/flooding_sample_verification', exist_ok=True)

    plt.axis('off')


    plt.savefig(f'../output/plots/flooding_sample_verification/{sample.rsplit("_",6)[2].replace("/","")}_w_depicts.png', dpi=400, bbox_inches='tight')

    plt.clf()
    
    

# %%



