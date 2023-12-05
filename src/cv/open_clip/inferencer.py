# FARLAB - UrbanECG 
# Developer: @mattwfranchi 
# Last Edited: 11/29/2023

# This script houses a class to run inference on a batch of images using OpenCLIP

# Module Imports
from cgitb import text
import torch
from PIL import Image
import open_clip

import cv2

import os 
import sys 

import numpy as np

import asyncio 
import aiofiles

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.logger import setup_logger

class OpenCLIPInferencer:
    def __init__(self, config: dict): 
        # make sure config has a choices key that has a list of strings as the value 
        if "choices" not in config.keys():
            raise KeyError("config must have a 'choices' key")
        if not isinstance(config["choices"], list): 
            raise TypeError("config['choices'] must be a list of strings")
        for choice in config["choices"]: 
            if not isinstance(choice, str): 
                raise TypeError("elemtents of config['choices'] must be strings")
        
        self.choices = config["choices"]

        # make sure "output_dir" exists and is a string
        if "output_dir" not in config.keys():
            raise KeyError("config must have a 'output_dir' key")
        if not isinstance(config["output_dir"], str): 
            raise TypeError("config['output_dir'] must be a string")
        

        # "prefix" is optional
        if "prefix" in config.keys(): 
            if not isinstance(config["prefix"], str): 
                raise TypeError("config['prefix'] must be a string")
            self.prefix = config["prefix"]
        else: 
            self.prefix = ""
        
        self.output_dir = config["output_dir"]

        # make output directory if it doesn't exist
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        # maie sure output_dir/frames exists
        if not os.path.exists(os.path.join(self.output_dir, "frames")): 
            os.makedirs(os.path.join(self.output_dir, "frames"))
        


        self.logger = setup_logger("OpenCLIPInferencer")
        self.logger.setLevel("INFO")
        self.logger.info("Initializing OpenCLIP Inferencer")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=self.device)
        #self.model.eval()
        #self.model = self.model.cuda()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

    
    async def infer(self, img_paths): 
        original_images = []
        images = [] 

        self.logger.debug("Reading in images...")

        for img_path in img_paths: 
            async with aiofiles.open(img_path, "rb") as f:
                # read in image from f as bytes with aiofiles and cv2
                image = await f.read()
                image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

                original_images.append(image)
                image = self.preprocess(image)
                images.append(image)
                
            
        self.logger.debug("Running inference...")
       
        text_tokens = self.tokenizer([self.prefix + " " + c for c in self.choices]).to(self.device)
        image_input = torch.tensor(np.stack(images)).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # turn text_probs from tensor to list of floats
            text_probs = text_probs.tolist()
        
        #print(f"Label probs for {os.path.basename(img_path)}:", text_probs)

        self.logger.debug("Writing results...")

        for img_path, text_prob in zip(img_paths, text_probs):
            self.logger.success(f"{os.path.basename(img_path)}: {text_prob}")

            # generate symlink to image in output directory, only if it doesn't already exist
            if not os.path.exists(os.path.join(self.output_dir, "frames", os.path.basename(img_path))): 
                os.symlink(img_path, os.path.join(self.output_dir, "frames", os.path.basename(img_path)))

            async with aiofiles.open(os.path.join(self.output_dir, "results.txt"), "a") as f: 
                await f.write(f"{img_path}, {os.path.splitext(os.path.basename(img_path))[0]}, {', '.join(map(str,text_prob))}\n")
        

            

        return text_probs

