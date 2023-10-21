# FARLAB - UrbanECG 
# Developer: Matt Franchi, with help from GitHub CoPilot 
# Last Modified: 10/18/2023


import sys 
import os 
from glob import glob
import pandas as pd 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.data_pull.random_sample_DoC import ImagePull
from src.utils.logger import setup_logger
class AnnotationViewer: 
    
    def __init__(self, DoC): 
        self.log = setup_logger()
        self.DoC = DoC
        self.image_pull = ImagePull("/share/ju/nexar_data/2023",DoC)
        self.pulled_imgs_dir = self.image_pull.pull_images(1, "annotation_viewer")

    def get_annotation(self, frame_id, yolo_detections_path=""):
        
        if yolo_detections_path == "":
            yolo_detections_path = f"../../output/yolo/{self.DoC}"

        # frame_id must be in self.image_pull.image_list 
        if frame_id not in self.image_pull.image_list['frame_id'].values: 
            raise ValueError(f"frame_id {frame_id} not in self.image_pull.image_list")
        
        self.log.info(f"Getting annotation for frame_id {frame_id}...")
        # get path of annotation for frame_id 
        frame_id = str(frame_id)
        print(f"{yolo_detections_path}/*/{frame_id}.txt")
        try: 
            frame_id = glob(f"{yolo_detections_path}/*/{frame_id}.txt")[0]
        except IndexError:
            self.log.error(f"frame_id {frame_id} not found in {yolo_detections_path}")
            raise ValueError(f"frame_id {frame_id} not found in {yolo_detections_path}")

        # read annotation
        with open(frame_id, 'r') as f: 
            annotation = f.readlines()
        
        # turn annotation into dataframe 
        annotation = pd.DataFrame([x.split() for x in annotation])
        annotation.columns = ['class', 'x', 'y', 'w', 'h']

        print(annotation)
        return annotation
    

    def __call__(self): 
        first_frame = glob(f"{self.pulled_imgs_dir}/*.jpg")[0]
        first_frame_id = os.path.basename(first_frame).split(".")[0]
        self.get_annotation(first_frame_id)
        
        
        

    

    


