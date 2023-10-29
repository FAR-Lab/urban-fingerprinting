# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/23/2023

# This script houses a class to crop annotated regions from images in batch, using asyncio.

# Import packages
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import setup_logger

from glob import glob

from PIL import Image

from io import BytesIO

import asyncio
import aiofiles


class Cropper:
    def __init__(self, DoC, image_dir):
        self.log = setup_logger()
        self.image_dir = image_dir
        self.image_list = glob(f"{image_dir}/*.jpg")
        self.log.info(f"Found {len(self.image_list)} images in {image_dir}")
        self.DoC = DoC
        self.SIZE_THRESHOLD = 3600
        self.ASPECT_RATIO = 9 / 16
        self.class_ids = [0]
        self.class_ids = list(map(str, self.class_ids))

        os.makedirs(f"../../output/cropper", exist_ok=True)

    async def crop(self, image_path):
        """Crop image to bounding box.

        Args:
            image_path (str): path to image to crop
            output_dir (str): path to output directory
        """
        self.log.info(f"Cropping annotated regions of {image_path}...")
        # get image id
        image_id = image_path.split("/")[-1].split(".")[0]

        # open image asynchronously
        async with aiofiles.open(image_path, "rb") as f:
            img = Image.open(image_path)

        # get annotation path
        try:
            annotation_path = glob(
                f"../../output/yolo/{self.DoC}/*/exp/labels/{image_id}.txt"
            )[0]
        except IndexError:
            self.log.error(f"No annotation found for {image_id}.txt")
            return 0
        # read annotation
        async with aiofiles.open(annotation_path, "r") as f:
            annotations = await f.readlines()

            for idx, a in enumerate(annotations):
                annotation = a.split()

                # get class
                class_id, cx, cy, w, h, conf = annotation

                if class_id not in self.class_ids:
                    continue

                # convert x,y,w,h from normalized to pixel coordinates
                cx = float(cx) * img.width
                cy = float(cy) * img.height
                w = float(w) * img.width
                h = float(h) * img.height

                # skip if area is < SIZE_THRESHOLD
                if w * h < self.SIZE_THRESHOLD:
                    self.log.info(
                        f"Area of {image_id}_{idx}.png is {w*h}, less than {self.SIZE_THRESHOLD}."
                    )
                    continue

                # skip if aspect ratio is not ASPECT_RATIO += 0.25
                if (w / h < (self.ASPECT_RATIO - 0.25)) or (
                    w / h > (self.ASPECT_RATIO + 0.25)
                ):
                    self.log.info(
                        f"Aspect ratio of {image_id}_{idx}.png is {w/h}, out of range."
                    )
                    continue

                # top
                top = cy - h / 2
                # bottom
                bottom = cy + h / 2
                # left
                left = cx - w / 2
                # right
                right = cx + w / 2

                output_dir = f"../../output/cropper/{class_id}"
                # make sure output directory exists
                os.makedirs(output_dir, exist_ok=True)

                output_path = f"{output_dir}/{image_id}_{idx}.png"

                buf = BytesIO()
                img.crop((left, top, right, bottom)).save(buf, format="PNG")

                async with aiofiles.open(output_path, "wb") as f:
                    self.log.info(f"Saving {image_id}_{idx}.png to {output_dir}")
                    await f.write(buf.getbuffer())

                    self.log.info(f"Saved {image_id}_{idx}.png to {output_dir}")

        return 1

    async def crop_batch(self):
        """Crop annotated regions of images in batch.

        Args:
            image_paths (list): list of paths to images to crop
            output_dir (str): path to output directory
        """
        self.log.info(f"Cropping annotated regions of {len(self.image_list)} images...")

        # crop each image
        tasks = []

        for image_path in self.image_list:
            tasks.append(asyncio.create_task(self.crop(image_path)))

        return await asyncio.gather(*tasks)

    def __call__(self):
        results = asyncio.run(self.crop_batch())

        self.log.success(f"Cropped annotations of {sum(results)} images successfully.")
