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

#from src.cv.FastSAM.fastsam import FastSAM, FastSAMPrompt

from segment_anything import SamPredictor, sam_model_registry


import numpy as np
import cv2

import glob


class BackgroundRemover:
    def __init__(self):
        self.log = setup_logger()
        self.log.setLevel("INFO")

        self.log.info("Initializing BackgroundRemover...")
        self.log.info("BackgroundRemover initialized.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = sam_model_registry["vit_h"](checkpoint='../../static/models/samhq/sam_hq_vit_h.pth')
        self.model.to(self.device)

        self.predictor = SamPredictor(self.model)

        self.hq_token_only = True # set to false when images typically have multiple objects

        self.output_dir = './'



    def segment_background(self, image_path):
        img = cv2.imread(image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get width and height of image
        height, width = img.shape[:2]

        # point_coords (np.ndarray or None): A Nx2 array of point prompts to the
        #    model. Each point is in (X,Y) in pixels.
        # point_labels (np.ndarray or None): A length N array of labels for the
        #   point prompts. 1 indicates a foreground point and 0 indicates a
        #    background point.

        pcs = np.array([[0, 0], [width, 0], [0, height], [width, height], [0, height/2], [width, height/2], [width / 2, height / 2], [width / 2, height*0.1], [width / 2, height*0.9]])
        pls = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

        self.predictor.set_image(img)
        masks, _, _ = self.predictor.predict(point_coords=pcs, point_labels=pls, hq_token_only=self.hq_token_only, multimask_output=False)

        ann = masks


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
