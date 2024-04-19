# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/23/2023

# This script houses a class to crop annotated regions from images in batch, using asyncio.

# Import packages
from codecs import ascii_encode
import sys
import os
from importlib import reload 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import setup_logger

from src.utils.toggle_project import set_project, reset_project 
set_project("informal_transportation")

import user.params.io as urbanECG_io 
reload(urbanECG_io)

from user.params.io import INSTALL_DIR, PROJECT_NAME, OUTPUT_DIR


from glob import glob

from PIL import Image

from io import BytesIO

import asyncio
import aiofiles




class Cropper:
    def __init__(self, DoC, image_dir):
        self.log = setup_logger("croppper")
        self.log.setLevel("INFO")
        self.image_dir = image_dir
        self.image_list = glob(f"{image_dir}/*/frames/*.jpg")
        self.log.info(f"Found {len(self.image_list)} images in {image_dir}")
        self.DoC = DoC
        self.SIZE_THRESHOLD = 2304
        self.ASPECT_RATIO = 16 / 9
        self.class_ids = [5,6,7]
        self.classes_to_filter = []
        self.class_ids = list(map(str, self.class_ids))
        self.classes_to_filter = list(map(str, self.classes_to_filter))




    async def crop(self, image_path, s):
        """Crop image to bounding box.

        Args:
            image_path (str): path to image to crop
            output_dir (str): path to output directory
        """

        async with s:

            self.log.info(f"Cropping annotated regions of {image_path}...")
            # get image id
            image_id = image_path.split("/")[-1].split(".")[0]
            
            try: 

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

                    # sort annotations by class, classes_to_filter first, then class_ids
                    # format: {class_id} {cx} {cy} {w} {h} {conf}
                    # put classes to filter first, then class_ids, then any others 
                    annotations = sorted(
                        annotations,
                        key=lambda x: (
                            x.split()[0] in self.classes_to_filter,
                            x.split()[0] in self.class_ids,
                            ((x.split()[0] not in self.classes_to_filter) and (x.split()[0] not in self.class_ids)),
                        ),
                        reverse=True
                    )

                
                    self.log.debug(f"Annotations: {annotations}")
                    

                    crops_to_filter = [] 
                    intersecting_crop = None

                    for idx, a in enumerate(annotations):
                        annotation = a.split()
                        fused = False

                        # get class
                        class_id, cx, cy, w, h, conf = annotation

                        if class_id not in self.class_ids + self.classes_to_filter:
                            continue


                        # convert x,y,w,h from normalized to pixel coordinates
                        cx = float(cx) * img.width
                        cy = float(cy) * img.height
                        w = float(w) * img.width
                        h = float(h) * img.height

                        if class_id in self.classes_to_filter:
                            # append {class_id, idx, bounding box} to crops_to_filter
                            crops_to_filter.append([class_id, idx, [cx, cy, w, h]])
                            continue 

                        # skip if area is < SIZE_THRESHOLD
                        if w * h < self.SIZE_THRESHOLD:
                            self.log.info(
                                f"Area of {image_id}_{idx}.png is {w*h}, less than {self.SIZE_THRESHOLD}."
                            )
                            continue

                        # skip if aspect ratio is not ASPECT_RATIO += 0.25
                        # April1: +.5 / -.5 should null the aspect ratio check
                        if (w / h < (self.ASPECT_RATIO - 0.5)) or (
                            w / h > (self.ASPECT_RATIO + 0.5)
                        ):
                            self.log.info(
                                f"Aspect ratio of {image_id}_{idx}.png is {w/h}, out of range."
                            )
                            continue

                        # skip if bounding box has IOU > 0.5 with any bounding box in crops_to_filter
                        try:
                            if len(crops_to_filter) > 0:
                                for crop in crops_to_filter:
                                    iou = self.iou([cx, cy, w, h], crop[2])
                                    if iou > 0.025:
                                        self.log.info(
                                            f"IOU of {image_id}_{idx}.png with {image_id}_{crop[0]}.png is {iou}, greater than 0.5."
                                        )
                                        
                                        # save crop to fused output directory
                                        fused = True 
                                        intersecting_crop = crop
                                        break 
                        except Exception as e:
                            self.log.error(f"iou check: {e}")
                            return e

                        # top
                        top = cy - h / 2
                        # bottom
                        bottom = cy + h / 2
                        # left
                        left = cx - w / 2
                        # right
                        right = cx + w / 2
                        try: 
                            if fused:
                                output_dir = f"{OUTPUT_DIR}/{PROJECT_NAME}/cropper/{class_id}_{intersecting_crop[0]}"
                            else:
                                output_dir = f"{OUTPUT_DIR}/{PROJECT_NAME}/cropper/{class_id}"
                        
                        except Exception as e:
                            self.log.error(f"output_dir: {e}")
                            return e
                        # make sure output directory exists
                        os.makedirs(output_dir, exist_ok=True)

                        output_path = f"{output_dir}/{image_id}_{idx}_{self.DoC}.png"

                        buf = BytesIO()
                        img.crop((left, top, right, bottom)).save(buf, format="PNG")

                        async with aiofiles.open(output_path, "wb") as f:
                            self.log.info(f"Saving {image_id}_{idx}_{self.DoC}.png to {output_dir}")
                            await f.write(buf.getbuffer())

                            self.log.success(f"Saved {image_id}_{idx}_{self.DoC}.png to {output_dir}")

                            await f.close()

                await f.close()
            except Exception as e:
                self.log.error(f"crop: {e}")
                return e
        
        
        return 1

    
    def iou(self, box1, box2):
        """Calculate intersection over union of two bounding boxes.

        Args:
            box1 (list): bounding box 1 [cx, cy, w, h]
            box2 (list): bounding box 2 [ cx, cy, w, h]

        Returns:
            float: intersection over union of box1 and box2
        """
        try:
            # get bounding box coordinates
            top1, bottom1, left1, right1 = self.get_box_coords(box1)
            top2, bottom2, left2, right2 = self.get_box_coords(box2)

            # get intersection coordinates
            top_int = max(top1, top2)
            bottom_int = min(bottom1, bottom2)
            left_int = max(left1, left2)
            right_int = min(right1, right2)

            # calculate intersection area
            intersection_area = max(0, bottom_int - top_int) * max(
                0, right_int - left_int
            )

            # calculate union area
            union_area = (
                (bottom1 - top1) * (right1 - left1)
                + (bottom2 - top2) * (right2 - left2)
                - intersection_area
            )

            # calculate intersection over union
            iou = intersection_area / union_area

            return iou
        except Exception as e: 
            self.log.error(f"iou: {e}")
            return e
                
    def get_box_coords(self, box):
        """Get bounding box coordinates.

        Args:
            box (list): bounding box [cx, cy, w, h]

        Returns:
            tuple: bounding box coordinates (top, bottom, left, right)
        """
        try: 
            cx, cy, w, h = box

            # top
            top = cy - h / 2
            # bottom
            bottom = cy + h / 2
            # left
            left = cx - w / 2
            # right
            right = cx + w / 2

            return top, bottom, left, right
        except Exception as e:
            self.log.error(f"bounding box: {e}")
            return e

    async def crop_batch(self):
        """Crop annotated regions of images in batch.

        Args:
            image_paths (list): list of paths to images to crop
            output_dir (str): path to output directory
        """

        s = asyncio.Semaphore(100)


        self.log.info(f"Cropping annotated regions of {len(self.image_list)} images...")

        # crop each image
        tasks = []

        for image_path in self.image_list:
            tasks.append(asyncio.create_task(self.crop(image_path, s)))

        return await asyncio.gather(*tasks)

    def __call__(self):



        results = asyncio.run(self.crop_batch())

        self.log.success(f"Cropped annotations of {sum(results)} images successfully.")

        reset_project()
