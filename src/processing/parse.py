# %%
import pandas as pd 
import matplotlib.pyplot as plt 

import logging 
from termcolor import colored

from glob import glob 
from concurrent.futures import ProcessPoolExecutor
import os 

from tqdm import tqdm

# %%
class ColorfulFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    def format(self, record):
        log_level = record.levelname
        msg = super().format(record)
        return colored(msg, self.COLORS.get(log_level, 'green'))

def setup_logger():
    logger = logging.getLogger('urban_fingerprinting')
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = ColorfulFormatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)
    return logger

log = setup_logger()


log.info("Start of notebook.")

# %%
PROJ_PATH = '/share/ju/nexar_data/nexar-scraper'
SAMPLE_TO_ANL = '2023-08-13'

# %%
preds_regex = f"{PROJ_PATH}/{SAMPLE_TO_ANL}/*/*/uf_detections/exp/labels/*.txt"
imgs_regex = f"{PROJ_PATH}/{SAMPLE_TO_ANL}/*/*/*.jpg"

# %%
preds_list = glob(preds_regex)
log.info(f"Number of detected predictions: {len(preds_list)}")

# %%
# Function to extract yhat from the given file path
def extract_w_conf(file_path):
    d = pd.read_csv(file_path, 
                    sep=' ', 
                    names=['class_type', 'dummy1', 'dummy2', 'dummy3', 'dummy4', 'conf'])
    
    d = d[['class_type', 'conf']]
    d = d.groupby('class_type').agg('size')

    # set all columns names to string versions of the class id 
    d = d.to_frame().T
    d.columns = [str(x) for x in d.columns]

    d['frame_id'] = file_path.split('/')[-1].split('.')[0]

    d = d.set_index('frame_id')

    return d 
    



# %%
pd.concat([extract_w_conf(preds_list[0]), extract_w_conf(preds_list[1])])


# %%

all_preds = pd.DataFrame(preds_list)
all_preds.columns = ['path']
all_preds['frame_id'] = all_preds['path'].str.split('/').str[-1].str.split('.').str[0]

all_preds = all_preds.set_index('frame_id')

# Use a parallel process to apply the function to each file path
def parallel_yhat_extraction(paths, num_processes=64):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(extract_w_conf, paths), total=len(paths), desc="Processing files"))
        results = pd.concat(results)
    return results

len_before_drop = len(all_preds)
existing_paths = all_preds['path'].apply(os.path.exists)
all_preds = all_preds[existing_paths]
len_after_drop = len(all_preds)
if len_before_drop != len_after_drop:
    log.warning(f'Dropped {len_before_drop - len_after_drop} rows due to missing paths')

all_preds_vals = parallel_yhat_extraction(all_preds['path'].tolist())

all_preds = all_preds.merge(all_preds_vals, left_index=True, right_index=True)


# %%
all_preds.to_csv('/share/ju/nexar_data/nexar-scraper/fingerprinting/08-13-2023-detections.csv')


