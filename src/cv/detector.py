import pandas as pd 
import numpy as np 
import subprocess 
import shlex 
import glob 
import os 
import asyncio 
import fire

class YOLO_Detector:
    def __init__(self, TOP_LEVEL_DIR, YOLO_DIR='./yolov7', YOLO_WEIGHTS='./yolov7_weights/yolov7-e6e.pt'):
        self.TOP_LEVEL_DIR = TOP_LEVEL_DIR
        self.YOLO_DIR = YOLO_DIR
        self.YOLO_WEIGHTS = YOLO_WEIGHTS
        # A Semaphore that allows 16 concurrent tasks.
        self.semaphore = asyncio.Semaphore(16)



    async def detect(self, frames_dir, worker_id):
        async with self.semaphore:
            detect_cmd = f'python {self.YOLO_DIR}/detect.py --weights {self.YOLO_WEIGHTS} --source  {frames_dir} --save-txt --save-conf --project {frames_dir}uf_detections  --device {worker_id} --img-size 1280 --nosave --conf-thres 0.5 --augment'
            detect_cmd = shlex.split(detect_cmd)
            print(detect_cmd)
            proc = await asyncio.create_subprocess_exec(*detect_cmd)
            returncode = await proc.wait()
            print(f'Worker {worker_id} finished processing {frames_dir} with return code {returncode}')
    
    def queue_detect_tasks(self, n_workers=4): 
        loop = asyncio.get_event_loop()
        tasks = [] 
        # can fit 16 detection tasks on a single GPU 
        frames_dirs = glob.glob(f'{self.TOP_LEVEL_DIR}/*/2023*/')
        for idx, frames_dir in enumerate(frames_dirs): 
            worker_id = idx % n_workers
            tasks.append(asyncio.ensure_future(self.detect(frames_dir, worker_id)))
        
        loop.run_until_complete(asyncio.gather(*tasks))



def ui_wrapper(TOP_LEVEL_DIR): 
    detector = YOLO_Detector(TOP_LEVEL_DIR=TOP_LEVEL_DIR)
    detector.queue_detect_tasks(n_workers=3)
    

if __name__ == '__main__':
    
    fire.Fire(ui_wrapper)

