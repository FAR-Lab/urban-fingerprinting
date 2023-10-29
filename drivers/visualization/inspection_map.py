import os 
import sys 

sys.path.append(os.path.abspath(os.path.join('../..')))

from glob import glob
import pandas as pd 

from src.visualization.inspection_map import InspectionMap

if __name__ == "__main__":
    # Create map object
    im = InspectionMap(center=[40.7128, -74.0060])
    # Save map to output directory

    frames = glob('../../notebooks/output/*depicts*/*.jpg')

    metadata = pd.concat([pd.read_csv(path, engine='pyarrow') for path in glob('../../../nexar_data/2023/2023-09-29/*/metadata.csv')])

    metadata = metadata[metadata['frame_id'].isin([os.path.splitext(os.path.basename(frame))[0] for frame in frames])]

    subset_descs = ['Manhole Overflow (Use Comments) (SA1)', 'Catch Basin Clogged/Flooding (Use Comments) (SC)', 'Sewer Backup (Use Comments) (SA)', 'Street Flooding (SJ)']

    flooding = pd.read_csv("../../data/coords/sep29_flooding.csv", engine='pyarrow')
    flooding = flooding[flooding['Descriptor'].isin(subset_descs)]

    for desc in subset_descs:
        im.init_complaint_group(desc)




    for index, row in metadata.iterrows():
        im.add_frame_marker(row, glob(f"../../notebooks/output/*depicts*/{row['frame_id']}.jpg")[0])

    for index, row in flooding.iterrows():
        im.add_311_marker(row)

    im.save("../../output/inspection_maps/inspection_map.html")