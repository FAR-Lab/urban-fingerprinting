# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 12/23/2023 

# This script houses a simple class, relying on smartcrop.js, to coordinate the cropping of a batch of images

# Module Imports
import smartcrop
import argparse 

import sys 
import PIL 
from PIL import Image

import os 
from glob import glob 

from tqdm import tqdm 

import asyncio 
import aiofiles
from io import BytesIO


sys.path.append("../../../")
from src.utils.logger import setup_logger

from src.utils.timer import timer 

class SmartCropHandler:
    def __init__(self, width, height): 
        self.logger = setup_logger("SmartCropHandler")
        self.logger.setLevel("INFO")
        self.logger.info("SmartCropHandler initialized")

        self.sc = smartcrop.SmartCrop()

        self.width = 512 
        self.height = 256 

        self.images_to_process = []

    def add_images_from_dir(self, dir_path: str):
        # add all images from a directory to self.images_to_process
        for filename in os.listdir(dir_path): 
            if filename.endswith(".jpg"): 
                self.images_to_process.append(os.path.join(dir_path, filename))
                self.logger.debug(f"Added {filename} to images_to_process")
            else: 
                self.logger.warning(f"Skipping {filename} because it is not a .jpg file")

    def _smartcrop(self, image: Image):
        # crop the image using smartcrop.js
        # returns a PIL.Image object
        result = self.sc.crop(image, self.width, self.height)

        # convert the result to a PIL.Image object
        cropped_image = image.crop((result["top_crop"]["x"], result["top_crop"]["y"], result["top_crop"]["x"] + result["top_crop"]["width"], result["top_crop"]["y"] + result["top_crop"]["height"]))
        return cropped_image

    @timer
    def process_images(self, output_dir):

        async def _process_image(image_path, output_dir):
            # process a single image
            async with aiofiles.open(image_path, mode="rb") as f: 
                # read the image asynchronosly
                image = await f.read()
                # convert to BytesIO object
                image = BytesIO(image)
                image = Image.open(image)


                # crop the image, but send task to thread pool
                cropped_image = await asyncio.get_running_loop().run_in_executor(None, self._smartcrop, image)
               

                buf = BytesIO()
                cropped_image.save(buf, format="JPEG")

                # save the cropped image asynchronosly

                async with aiofiles.open(os.path.join(output_dir, os.path.basename(image_path)), mode="wb") as f: 
                    # convert cropped image to bytes
                    await f.write(buf.getbuffer())
               

                self.logger.success(f"Successfully cropped {image_path}")
        
        async def _process_images(output_dir):
            # make task list
            tasks = []
            for idx, image_path in enumerate(self.images_to_process):
                task = _process_image(image_path, output_dir)
                tasks.append(asyncio.create_task(task))  # Create tasks for asyncio

                if idx == 100:
                    break

            # run tasks
            await asyncio.gather(*tasks)

        # run the async function
        asyncio.run(_process_images(output_dir))

    

# implement a cli for this class 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Crop a batch of images using smartcrop.js")
    parser.add_argument("--input-dir", type=str, help="Path to directory containing images to crop")
    parser.add_argument("--output-dir", type=str, help="Path to directory to save cropped images", default="./")
    parser.add_argument("--width", type=int, help="Width of cropped images", default=512)
    parser.add_argument("--height", type=int, help="Height of cropped images", default=256)

    args = parser.parse_args()

    # initialize the SmartCropHandler
    sch = SmartCropHandler(args.width, args.height)

    # add images from the input directory
    sch.add_images_from_dir(args.input_dir)

    # process the images
    sch.process_images(args.output_dir)


    





    

