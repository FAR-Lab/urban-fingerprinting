# %%
import os 
import sys 
import glob 
import pandas as pd 
import geopandas as gpd
from tqdm import tqdm

# %%
# add parent directory to path 
sys.path.append('/share/ju/urbanECG')


# %%
from src.utils.data_pull.random_sample_DoC import ImagePull

from user.params.data import * 

# %%
flooding = pd.read_csv('../data/coords/sep29_flooding.csv', engine='pyarrow')

# %%
flooding = gpd.GeoDataFrame(flooding, geometry=gpd.points_from_xy(flooding.Longitude, flooding.Latitude), crs=COORD_CRS)

# %%
flooding = flooding.to_crs(PROJ_CRS)

# %%
flooding['Descriptor'].value_counts()

# %%
nexar_pull = ImagePull("/share/ju/nexar_data/2023", "2023-09-29")

# %%
descriptors = list(flooding['Descriptor'].unique())
descriptors



# %%
descriptors = ['Manhole Overflow (Use Comments) (SA1)', 'Catch Basin Clogged/Flooding (Use Comments) (SC)', 'Sewer Backup (Use Comments) (SA)', 'Street Flooding (SJ)']

# %%
for descriptor in descriptors:
    nexar_pull.pull_images(1000, f"{descriptor}_w_depicts", coords=flooding[flooding['Descriptor'] == descriptor].copy(), proximity=100, time_delta=30)
    

# %%


# %%



