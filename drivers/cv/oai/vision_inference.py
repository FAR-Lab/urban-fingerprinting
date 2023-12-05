# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/08/2023

# This script houses a driver for OpenAI's GPT4-Vision API.

# Module Imports
import os 
import sys 

import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
)

from glob import glob
from random import shuffle 
from src.cv.oai.vision_inference import OAI_Session
import json
from tqdm import tqdm

import asyncio


if __name__ == '__main__': 

    #img_path = "/share/ju/nexar_data/training_datasets/flooding_CLIP_336/87ca38a717a92881d0919956b54fb697_flooded road_0.7010676860809326.jpg"
    img_dir = "/share/ju/nexar_data/training_datasets/flooding_CLIP_336/"
    imgs = glob(img_dir + "*.jpg")

    imgs = pd.read_csv("/share/ju/urbanECG/output/street_flooding/train_set_gt_pt2_w_clipvitg_results_flooded_road.csv")["img_path"].tolist()

    results = pd.read_csv("./flooding_gpt4v.csv")

    # filter out images that have already been inferred
    imgs = [img for img in imgs if img not in results["img_path"].tolist()]

    print(len(imgs))

    NUM_IMGS_TO_INFER = 1

    shuffle(imgs)
    #imgs = imgs[:NUM_IMGS_TO_INFER]

    async def main(): 
        async with OAI_Session() as session: 
            pass
            tasks = [await session.infer_image(img) for img in tqdm(imgs, desc="Running inference on images")]
            #tasks = [await session.infer_image(img_path)]
            return tasks 

    results = asyncio.run(main())