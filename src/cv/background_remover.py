# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/23/2023

# This script houses a class to remove background from a set in images in a directory in parallel, asynchronously.

# Module Imports
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import torch

from utils.logger import setup_logger

from src.cv.FastSAM.fastsam import FastSAM, FastSAMPrompt
import numpy as np
import cv2

import glob


class BackgroundRemover:
    def __init__(self):
        self.log = setup_logger()
        self.log.info("Initializing BackgroundRemover...")
        self.log.info("BackgroundRemover initialized.")
        self.model = FastSAM("./fastsam_weights/FastSAM-x.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def segment_background(self, image_path):
        img = cv2.imread(image_path)
        # get width and height of image
        height, width = img.shape[:2]
        results = self.model(
            image_path,
            device=self.device,
            retina_masks=True,
            imgsz=max(height, width),
            conf=0.5,
            iou=0.75,
        )
        prompt_process = FastSAMPrompt(image_path, results, device=self.device)

        # get the background mask
        ann = prompt_process.text_prompt(text="a person")
        # convert ann to a binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for a in ann:
            mask[a == True] = 1
        ann = mask.astype(bool)

        # make ann compatible with cv2 bitwise_and
        # False -> (0, 0, 0)
        # True -> (255, 255, 255)
        ann = ann.astype(np.uint8)
        ann = cv2.cvtColor(ann, cv2.COLOR_GRAY2RGB)
        ann *= 255

        # turn everything outside of ann to black
        # turn false pixels to black
        img = cv2.bitwise_and(img, ann)
        img[ann == 0] = 0

        # save the image
        cv2.imwrite(f"{self.output_dir}/{os.path.basename(image_path)}", img)

    def batch(self, dir_to_scan, output_dir):
        # scan the directory for images
        self.log.info("Scanning directory for images...")
        images = glob.glob(dir_to_scan + "/*.png")
        self.log.info("Found " + str(len(images)) + " images.")
        self.log.info("Removing background from images...")
        self.output_dir = output_dir
        for image in images:
            self.segment_background(image)
        self.log.info("Background removal complete.")
        self.log.info("Images saved to " + output_dir + ".")
