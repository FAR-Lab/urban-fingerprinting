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

from utils.logger import setup_logger

from rembg import remove 
import aiofiles
import asyncio 

class BackgroundRemover(): 
    def __init__(self)
        self.log = setup_logger()
        self.log.info("Initializing BackgroundRemover...")
        self.log.info("BackgroundRemover initialized.")
    
    async def remove_background(self, img_path, output_path):
        """Remove background from an image and save it to output_path. 

        Args: 
            img_path (str): path to image to remove background from 
            output_path (str): path to save image with background removed to 
        """
        
        with open(img_path, 'rb') as f: 
            img = f.read()
        
        result = remove.bg(img)
        async with aiofiles.open(output_path, 'wb') as f: 
            await f.write(result)
        self.log.debug(f"Background removed from {img_path} and saved to {output_path}.")

    async def remove_backgrounds(self, img_paths, output_path):
        """Remove background from a set of images and save them to output_path. 

        Args: 
            img_paths (list): list of paths to images to remove background from 
            output_path (str): path to save images with background removed to 
        """
        tasks = []
        for img_path in img_paths: 
            tasks.append(self.remove_background(img_path, output_path))
        await asyncio.gather(*tasks)
        self.log.debug(f"Background removed from {len(img_paths)} images and saved to {output_path}.")
    
    def __call__(self, img_paths, output_path):
        """Remove background from a set of images and save them to output_path. 

        Args: 
            img_paths (list): list of paths to images to remove background from 
            output_path (str): path to save images with background removed to 
        """
        asyncio.run(self.remove_backgrounds(img_paths, output_path))
        self.log.info(f"Background removed from {len(img_paths)} images and saved to {output_path}.")
    





