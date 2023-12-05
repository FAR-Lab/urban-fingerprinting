# FARLAB - UrbanECG
# Developer: @mattwfranchi
# Last Edited: 11/29/2023

# This script houses a driver to run inference on a batch of images using OpenCLIP

# Module Imports
import os
from random import shuffle
import sys
from glob import glob

from tqdm import tqdm

import numpy as np

import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.cv.open_clip.inferencer import OpenCLIPInferencer

if __name__ == "__main__":  

    async def main():
    
        config = {
            "choices": ["road with potholes", "road with damage", "maintained road", "no road"],
            "output_dir": "/share/ju/urbanECG/output/open_clip/potholes/2023-08-18/",
            "prefix": "a photo showing"
        }

        inferencer = OpenCLIPInferencer(config)

        DoC = "/share/ju/nexar_data/2023/2023-08-18/*/frames"
        img_paths = glob(os.path.join(DoC, "*.jpg"))

        shuffle(img_paths)

        BATCH_SIZE = 8

        # split img_paths into lists of size BATCH_SIZE
        batches = np.array_split(img_paths, len(img_paths) // BATCH_SIZE)

        # submit batches to the inferencer
        for batch in tqdm(batches): 
            await inferencer.infer(batch)
    

    asyncio.run(main())





