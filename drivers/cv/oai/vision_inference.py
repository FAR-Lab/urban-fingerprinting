# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/08/2023

# This script houses a driver for OpenAI's GPT4-Vision API.

# Module Imports
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
)

from src.cv.oai.vision_inference import OAI_Session

import asyncio

if __name__ == '__main__': 

    img_path = "/share/ju/nexar_data/training_datasets/flooding_CLIP_336/87ca38a717a92881d0919956b54fb697_flooded road_0.7010676860809326.jpg"

    async def main(): 
        async with OAI_Session() as session: 
            print(await session.infer_image(img_path))

    asyncio.run(main())