import os 
import sys 

sys.path.append(os.path.abspath(os.path.join('../..')))

from glob import glob
import pandas as pd 

from src.visualization.InspectionMap import InspectionMap

if __name__ == "__main__":
    # Create map object
    im = InspectionMap(center=[40.7128, -74.0060])
    # Save map to output directory

    frames = glob('../../notebooks/output/flooding_929_Sewer Backup (Use Comments) (SA)_w_depicts_10000_2023-09-29/*.jpg')

    metadata = pd.concat([pd.read_csv(path, engine='pyarrow') for path in glob('../../../nexar_data/2023/2023-09-29/*/metadata.csv')])

    metadata = metadata[metadata['frame_id'].isin([os.path.splitext(os.path.basename(frame))[0] for frame in frames])]

    for index, row in metadata.iterrows():
        im.add_frame_marker(row, os.path.join('../../notebooks/output/flooding_929_Sewer Backup (Use Comments) (SA)_w_depicts_10000_2023-09-29', row['frame_id'] + '.jpg'))

    im.save("../../output/inspection_maps/inspection_map.html")