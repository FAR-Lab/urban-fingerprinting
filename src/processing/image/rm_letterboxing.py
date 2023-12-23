# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 12/23/2023 

# This script houses a simple class to coordinate the removal of letterboxing from a batch of images

# Module Imports
import cv2
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append("../../../")

from src.utils.logger import setup_logger
from src.utils.read_csv import read_csv 

class LetterboxingHandler:
    def __init__(self): 
        self.logger = setup_logger("LetterboxingHandler")
        self.logger.setLevel("INFO")
        self.logger.info("LetterboxingHandler initialized")

        self.images_to_process = []
        

    def add_images_from_dir(self, dir_path: str):
        # add all images from a directory to self.images_to_process
        for filename in os.listdir(dir_path): 
            if filename.endswith(".jpg"): 
                self.images_to_process.append({"filename": filename, "path": os.path.join(dir_path, filename)})
                self.logger.debug(f"Added {filename} to images_to_process")
            else: 
                self.logger.warning(f"Skipping {filename} because it is not a .jpg file")

    


    def _remove_letterboxing(self, image: np.ndarray):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale image
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initial crop coordinates
        top_y = 0
        bottom_y = image.shape[0]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if the contour is at the top or bottom of the image
            if y < image.shape[0] // 2 or y + h > image.shape[0] // 2:
                # Check if the contour is approximately rectangular
                contour_area = cv2.contourArea(contour)
                bbox_area = w * h
                area_ratio = contour_area / bbox_area

                # Adjust the threshold as needed
                if 0.9 < area_ratio < 1.1:
                    # Update crop coordinates
                    if y < image.shape[0] // 2:
                        top_y = max(top_y, y + h)
                    else:
                        bottom_y = min(bottom_y, y)

        # Crop the image to remove top and bottom black bars
        cropped_image = image[top_y:bottom_y, 0:image.shape[1]]

        # return the image
        return cropped_image


    
    def remove_letterboxing_batch(self, output_dir="./", prefix=""): 
        # remove letterboxing from each image in self.images_to_process
        for idx, row in tqdm(enumerate(self.images_to_process), total=len(self.images_to_process), desc="Removing letterboxing from images..."): 
            image = cv2.imread(row["path"])
            image = self._remove_letterboxing(image)
            cv2.imwrite(os.path.join(output_dir, f"{prefix}{row['filename']}"), image)
            self.logger.debug(f"Removed letterboxing from {row['filename']}")

    
# implement a CLI for this class
if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser(description="Remove letterboxing from a batch of images")
    
    parser.add_argument("--input_dir", type=str, help="path to directory containing images", required=True)
    parser.add_argument("--output_dir", type=str, help="path to output directory", default="./")
    parser.add_argument("--prefix", type=str, help="prefix to add to each filename", default="")
    args = parser.parse_args()

    letterboxing_handler = LetterboxingHandler()
    letterboxing_handler.add_images_from_dir(args.input_dir)
    letterboxing_handler.remove_letterboxing_batch(args.output_dir, args.prefix)

    
        
