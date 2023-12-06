import os 
import sys 

sys.path.append(os.path.abspath(os.path.join('../..')))

from user.params.io import INSTALL_DIR, PROJECT_NAME

from glob import glob
import pandas as pd 
import geopandas as gpd 

from src.visualization.inspection_map import InspectionMap
from user.params.data import *

if __name__ == "__main__":
    # Create map object
    im = InspectionMap(center=[40.7128, -74.0060])
    # Save map to output directory

    #frames = glob('../../notebooks/output/*depicts*/*.jpg')

    metadata = pd.read_csv('../../output/street_flooding/subsets/frames_near_floodnet.csv')

    #metadata = pd.concat([pd.read_csv(path, engine='pyarrow') for path in glob('../../../nexar_data/2023/2023-09-29/*/metadata.csv')])

    #metadata = metadata[metadata['frame_id'].isin([os.path.splitext(os.path.basename(frame))[0] for frame in frames])]

    metadata = gpd.GeoDataFrame(metadata, geometry=gpd.points_from_xy(metadata[LONGITUDE_COL], metadata[LATITUDE_COL]), crs=COORD_CRS)
    metadata = metadata.to_crs(PROJ_CRS)

    #subset_descs = ['Manhole Overflow (Use Comments) (SA1)', 'Catch Basin Clogged/Flooding (Use Comments) (SC)', 'Sewer Backup (Use Comments) (SA)', 'Street Flooding (SJ)']

    #flooding = pd.read_csv("../../data/coords/sep29_flooding.csv", engine='pyarrow')
    #flooding = flooding[flooding['Descriptor'].isin(subset_descs)]

    #flooding = gpd.GeoDataFrame(flooding, geometry=gpd.points_from_xy(flooding[LONGITUDE_COL_311], flooding[LATITUDE_COL_311]), crs=COORD_CRS)
    #flooding = flooding.to_crs(PROJ_CRS)

    floodnet = pd.read_csv("../../data/nyc_flooding/floodnet_sensor_coordinates.csv")



    #for desc in subset_descs:
    #    im.init_complaint_group(desc)



    print(len(metadata.index))
    for index, row in metadata.iterrows():
        #print(glob(f"../../notebooks/*depicts*/{row['frame_id']}.jpg"))
        try:
            im.add_frame_marker(row, row['img_path'])
        except Exception as e:
            print(f"Failed to add frame marker for frame {row['frame_id']}: {e}")

    #for index, row in flooding.iterrows():
    #    im.add_311_marker(row)

    for index, row in floodnet.iterrows():
        im.add_floodnet_marker(row)

    os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/inspection_maps", exist_ok=True)

    im.save(f"{INSTALL_DIR}/{PROJECT_NAME}/inspection_maps/floodnet_inspection_map.html")