# FARLAB - UrbanECG
# Developer: Matt Franchi, with help from GitHub CoPilot
# Last Modified: 10/18/2023


import sys
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patheffects as path_effects

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.data_pull.random_sample_DoC import ImagePull
from src.utils.logger import setup_logger


class AnnotationViewer:
    def __init__(self, DoC):
        self.log = setup_logger()
        self.DoC = DoC
        self.image_pull = ImagePull("/share/ju/nexar_data/2023", DoC)
        self.pulled_imgs_dir = self.image_pull.pull_images(
            1, "annotation_viewer"
        )

    def get_annotations(self, frame_id, yolo_detections_path=""):
        if yolo_detections_path == "":
            yolo_detections_path = f"../../output/yolo/{self.DoC}"

        # frame_id must be in self.image_pull.image_list
        if frame_id not in self.image_pull.image_list["frame_id"].values:
            raise ValueError(
                f"frame_id {frame_id} not in self.image_pull.image_list"
            )

        self.log.info(f"Getting annotation for frame_id {frame_id}...")
        # get path of annotation for frame_id
        frame_id = str(frame_id)

        try:
            frame_id = glob(
                f"/share/ju/urbanECG/output/yolo/2023-08-18/*/exp/labels/{frame_id}.txt"
            )[0]
        except IndexError:
            self.log.error(
                f"frame_id {frame_id} not found in {yolo_detections_path}"
            )
            raise ValueError(
                f"frame_id {frame_id} not found in {yolo_detections_path}"
            )

        # read annotation
        with open(frame_id, "r") as f:
            annotations = f.readlines()

        # turn annotation into dataframe
        annotations = pd.DataFrame([x.split() for x in annotations])
        annotations.columns = ["class", "x", "y", "w", "h", "conf"]

        print(annotations)
        return annotations

    def visualize_annotations(self, frame_id, annotations, class_id=-1):
        # get path of image for frame_id
        frame_id = str(frame_id)
        frame_path = glob(f"{self.pulled_imgs_dir}/{frame_id}.jpg")[0]
        frame = Image.open(frame_path)
        frame = np.array(frame)
        # frame = np.flip(frame, axis=2)
        # frame = np.flip(frame, axis=1)
        # frame = np.flip(frame, axis=0)
        print(frame.shape)

        fig, ax = plt.subplots(figsize=(16, 9))

        ax.imshow(frame)

        # set axis limits
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)

        plt.axis("off")
        plt.tight_layout()

        if class_id != -1:
            annotations = annotations[annotations["class"] == class_id]

        for _, annotation in annotations.iterrows():
            class_id, x, y, w, h, conf = annotation

            # convert from normalized to pixel coordinates
            x = float(x) * frame.shape[1]
            y = float(y) * frame.shape[0]
            w = float(w) * frame.shape[1]
            h = float(h) * frame.shape[0]

            bottom_left = (x - w / 2, y - h / 2)
            top_center = (x, y - h / 2)

            # generate random color for class_id
            np.random.seed(int(class_id))
            color = np.random.randint(0, 255, size=3)
            # normalize color
            color = color / 255

            # plot bounding box
            rect = plt.Rectangle(
                bottom_left, w, h, fill=False, edgecolor=color, linewidth=2
            )
            ax.add_patch(rect)

            # plot rectangle behind text
            text_bbox = ax.text(
                top_center[0],
                top_center[1],
                f"{class_id} [{conf}]",
                color="black",
            )
            text_bbox.set_bbox(
                dict(facecolor=color, alpha=0.5, edgecolor=color)
            )

            # antialiasing
            text_bbox.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )

            # make sure output dire exists if not create it
            if not os.path.exists(f"../../output/annotation_viewer"):
                os.makedirs(f"../../output/annotation_viewer")
            plt.savefig(
                f"../../output/annotation_viewer/{frame_id}.png",
                bbox_inches="tight",
                pad_inches=0,
            )

    def save_crops(self, frame_id, annotations, class_id=-1):
        # get path of image for frame_id
        frame_id = str(frame_id)
        frame_path = glob(f"{self.pulled_imgs_dir}/{frame_id}.jpg")[0]
        frame = Image.open(frame_path)
        frame = np.array(frame)
        # frame = np.flip(frame, axis=2)
        # frame = np.flip(frame, axis=1)
        # frame = np.flip(frame, axis=0)
        print(frame.shape)

        if class_id != -1:
            annotations = annotations[annotations["class"] == class_id]

        for idx, annotation in annotations.iterrows():
            class_id, x, y, w, h, conf = annotation

            # convert from normalized to pixel coordinates
            x = float(x) * frame.shape[1]
            y = float(y) * frame.shape[0]
            w = float(w) * frame.shape[1]
            h = float(h) * frame.shape[0]

            bottom_left = (x - w / 2, y - h / 2)
            top_center = (x, y - h / 2)

            # crop image
            crop = frame[
                int(y - h / 2) : int(y + h / 2),
                int(x - w / 2) : int(x + w / 2),
            ]

            # make sure output dire exists if not create it
            if not os.path.exists(
                f"../../output/annotation_viewer/{class_id}"
            ):
                os.makedirs(f"../../output/annotation_viewer/{class_id}")

            # save crop
            plt.imsave(
                f"../../output/annotation_viewer/{class_id}/{frame_id}_{idx}.png",
                crop,
            )

    def __call__(self):
        first_frame = glob(f"{self.pulled_imgs_dir}/*.jpg")[0]
        first_frame_id = os.path.basename(first_frame).split(".")[0]
        annotations = self.get_annotations(first_frame_id)
        self.visualize_annotations(first_frame_id, annotations)
        self.save_crops(first_frame_id, annotations, class_id=0)
