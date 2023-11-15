# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub Copilot
# Last Edited: 11/06/2023

# This script houses a class to run CLIP-powered zero-shot image classification on a dataset of images.


# Module Imports
import os
import sys
from glob import glob
from random import shuffle

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)

from src.utils.logger import setup_logger
from src.utils.timer import timer

from user.params.io import TRAINING_DATA_DIR

import pandas as pd
from PIL import Image

import torch
from lavis.models import load_model_and_preprocess


class CLIPZeroShot:
    def __init__(self, frames_dir, output_prefix, choices={}):
        self.log = setup_logger("CLIP Zero Shot")
        self.log.setLevel("INFO")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.log.info(f"Using device: {self.device}")

        self.frames_dir = frames_dir
        self.frames = glob(
            os.path.join(self.frames_dir, "*", "frames", "*.jpg")
        )
        self.log.info(f"Found {len(self.frames)} images in dataset")

        self.output_prefix = output_prefix
        
        # make sure output directory exists
        os.makedirs(f"{TRAINING_DATA_DIR}/{self.output_prefix}", exist_ok=True)
        

        (
            self.zeroshot_model,
            self.zeroshot_vis_processors,
            self.zeroshot_txt_processors,
        ) = load_model_and_preprocess(
            "clip_feature_extractor",
            model_type="ViT-L-14-336",
            is_eval=True,
            device=self.device,
        )
        self.log.info(f"Loaded CLIP model: {self.zeroshot_model}")

        self.choices = choices

    def update_choices(self, choices):
        self.choices = choices

    def zeroshot(self, image_path, choices):
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.log.error(f"Error opening image: {e}")
            return None

        choices = [
            self.zeroshot_txt_processors["eval"](choice)
            for choice in self.choices
        ]
        self.log.debug(f"choices: {choices}")

        sample = {
            "image": self.zeroshot_vis_processors["eval"](raw_image)
            .unsqueeze(0)
            .to(self.device),
            "text_input": choices,
        }

        clip_features = self.zeroshot_model.extract_features(sample)

        image_features = clip_features.image_embeds_proj
        text_features = clip_features.text_embeds_proj

        sims = (image_features @ text_features.t())[0] / 0.01
        probs = torch.nn.Softmax(dim=0)(sims).tolist()

        for cls_nm, prob in zip(choices, probs):
            self.log.debug(f"{cls_nm}: {prob}")

        most_likely = max(zip(choices, probs), key=lambda x: x[1])
        if most_likely[1] < 0.5:
            self.log.info(
                f"None of the choices are likely, most likely: {most_likely}"
            )
            return most_likely
        
        if most_likely[0] == choices[0]:
            os.symlink(
                image_path,
                f"{TRAINING_DATA_DIR}/{self.output_prefix}/{os.path.splitext(os.path.basename(image_path))[0]}_{most_likely[0]}_{most_likely[1]}.jpg",
            )
            self.log.success(
                f"{most_likely[0]} detected with probability {most_likely[1]}"
            )

        return most_likely

    @timer
    def zeroshot_all(self):
        self.log.info("Starting zeroshot classification task...")
        shuffle(self.frames)
        self.log.info(f"Shuffled dataset")

        if len(self.choices) == 0:
            self.log.error("No choices provided")
            return None

        if len(self.choices) < 2:
            self.log.error("Not enough choices provided")
            return None

        for img in tqdm(self.frames, desc="Zeroshotting dataset"):
            self.zeroshot(img, self.choices)

        self.log.success("Finished zeroshotting dataset.")
