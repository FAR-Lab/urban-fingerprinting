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

sys.path.append("../../../")
from src.utils.logger import setup_logger

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

    def process_images(self, output_dir):
        # process all images in self.images_to_process
        for idx, path in tqdm(enumerate(self.images_to_process), desc="SmartCropping Images", total=len(self.images_to_process)): 
            # load the image
            image = Image.open(path)

            # crop the image
            cropped_image = self._smartcrop(image)

            # save the image
            cropped_image.save(os.path.join(output_dir, os.path.basename(path)))
            self.logger.success(f"Saved {path} to {os.path.join(output_dir, os.path.basename(path))}")

            if idx == 10: 
                break
    

# implement a cli for this class 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Crop a batch of images using smartcrop.js")
    parser.add_argument("--input_dir", type=str, help="Path to directory containing images to crop")
    parser.add_argument("--output_dir", type=str, help="Path to directory to save cropped images", default="./")
    parser.add_argument("--width", type=int, help="Width of cropped images", default=512)
    parser.add_argument("--height", type=int, help="Height of cropped images", default=256)

    args = parser.parse_args()

    # initialize the SmartCropHandler
    sch = SmartCropHandler(args.width, args.height)

    # add images from the input directory
    sch.add_images_from_dir(args.input_dir)

    # process the images
    sch.process_images(args.output_dir)


    





    

