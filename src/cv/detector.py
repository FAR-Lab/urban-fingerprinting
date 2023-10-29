# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot
# Last Edited: 09/22/2023

# Imports
import shlex
import glob
import os
import logging
import subprocess
import asyncio
import fire
import pandas as pd
import numpy as np
from tqdm import tqdm


# Insights into choices for detection pipeline:
##### As sampling density is not uniform across space, I opt to chunk the frames into equal chunks
##### I've found that with the E6E weights, I can fit 16 detection tasks on a single GPU
##### The yolov7 repo has a detect.py script that can be used to run detection on a single GPU
#####  No parallelization is built-in, so we parallelize by running multiple detect.py instances on a single GPU
##### Further parallelization is achieved by running 16 * n instances, where n is the number of GPUs available
##### This is achieved by creating a list of frames lists, where each frames list contains 16 frames
##### However, detection is much slower when feeding in a txt.file containing a list of frames, not sure why
########## (seems to be because each file is loaded with cv2, in a different way then when just specifying a dir that contains .jpg files)
##### So, we create symlinks to each frame in the frames list, and feed in the directory containing the symlinks
##### This is done in the create_batch_symlinks function
##### This script processes about 600 frames per second, after startup overhead


class YOLO_Detector:
    """
    A class to run YOLOv7 object detection on a set of frames.

    ...

    Attributes
    ----------
    TOP_LEVEL_DIR : str
        The top level directory containing the frames to be processed
    DAY_OF_COVERAGE : str
        The day of coverage to be processed (assumes a TOP_LEVEL_DIR/DAY_OF_COVERAGE directory structure)
    YOLO_DIR : str
        The directory containing the yolov7 repo
    YOLO_WEIGHTS : str
        The path to the yolov7 weights file
    OUTPUT_DIR : str
        The directory to write the output to
    NUM_GPUS : int
        The number of GPUs available for parallelization
    semaphores : list
        A list of semaphores, one for each GPU

    Methods
    -------
    write_n_frames_list(frames_dirs, n, frames_list_path)
        Writes a list of frames to a file, where each file is a equal-sized chunk denoted by len(all frames) / n
    create_batch_symlinks(frames_path_lists, frames_list_path)
        Creates symlinks to each frame in the frames list
    detect(frames_list_path, worker_id)
        Runs detection on a single GPU
    queue_detect_tasks(n_workers=4, write_frames_lists=True)
        Queues up detection tasks for each GPU, also handles writing of frames lists and creation of symlinks if write_frames_lists=True
    """

    def __init__(
        self,
        TOP_LEVEL_DIR,
        DAY_OF_COVERAGE,
        YOLO_DIR="./yolov7",
        YOLO_WEIGHTS="./yolov7_weights/yolov7-e6e.pt",
        OUTPUT_DIR="../../output/yolo/",
    ):
        self.log = logging.getLogger(__name__)
        self.TOP_LEVEL_DIR = TOP_LEVEL_DIR
        self.YOLO_DIR = YOLO_DIR
        self.DAY_OF_COVERAGE = DAY_OF_COVERAGE
        self.OUTPUT_DIR = OUTPUT_DIR
        self.YOLO_WEIGHTS = YOLO_WEIGHTS
        self.NUM_GPUS = 4
        self.MAX_TASKS_PER_GPU = 16
        # A Semaphore that allows 16 concurrent tasks.
        # Create list of semaphores for each GPU
        self.semaphores = [
            asyncio.Semaphore(self.MAX_TASKS_PER_GPU) for _ in range(self.NUM_GPUS)
        ]

    def write_n_frames_list(self, frames_dirs, n, frames_list_path):
        all_frames = []
        for frames_dir in frames_dirs:
            frames = glob.glob(f"{frames_dir}/*.jpg")
            all_frames.extend(frames)
        all_frames = sorted(all_frames)

        # Split into n chunks
        chunks = np.array_split(all_frames, n)

        # Write each chunk to a file
        files_list = []

        # Create directory if it doesn't exist
        os.makedirs(frames_list_path, exist_ok=True)

        for idx, chunk in enumerate(chunks):
            chunk = "\n".join(chunk)
            with open(f"{frames_list_path}/chunk_{idx}.txt", "w") as f:
                f.write(chunk)
            files_list.append(f"{frames_list_path}/chunk_{idx}.txt")

        del chunks

        return files_list

    def create_batch_symlinks(self, frames_path_lists, frames_list_path):
        batch_paths = []
        for frames_list in tqdm(
            frames_path_lists, desc="Creating symlinks for all frames lists"
        ):
            batch_paths.append(f"{frames_list_path}/{os.path.basename(frames_list)}")
            # Create directory if it doesn't exist
            os.makedirs(
                f"{frames_list_path}/{os.path.splitext(os.path.basename(frames_list))[0]}",
                exist_ok=True,
            )

            with open(frames_list, "r") as f:
                frames = f.readlines()
                # Create symlink to each frame in the batch
                for frame in tqdm(frames, desc=f"Creating symlinks for {frames_list}"):
                    frame = frame.strip()
                    frame_name = frame.split("/")[-1]
                    os.symlink(
                        frame,
                        f"{frames_list_path}/{os.path.splitext(os.path.basename(frames_list))[0]}/{frame_name}",
                    )

            f.close()

        return batch_paths

    async def detect(self, frames_list_path, worker_id, gpu_id):
        # make sure project dir exists
        os.makedirs(
            f"{self.OUTPUT_DIR}/{self.DAY_OF_COVERAGE}/{worker_id}", exist_ok=True
        )
        async with self.semaphores[gpu_id]:
            try:
                detect_cmd = f"python {self.YOLO_DIR}/detect.py --weights {self.YOLO_WEIGHTS} --source  {frames_list_path} --save-txt --save-conf --project {self.OUTPUT_DIR}/{self.DAY_OF_COVERAGE}/{worker_id}  --device {gpu_id} --img-size 1280 --nosave --conf-thres 0.5 --augment"
                detect_cmd = shlex.split(detect_cmd)
                print(detect_cmd)
                proc = await asyncio.create_subprocess_exec(*detect_cmd)
                returncode = await proc.wait()
                print(
                    f"Worker {worker_id} finished processing {frames_list_path} with return code {returncode}"
                )
            finally:
                self.semaphores[gpu_id].release()

    def run_detect_scripts(self):
        # make sure project dir exists
        os.makedirs(f"{self.OUTPUT_DIR}/{self.DAY_OF_COVERAGE}", exist_ok=True)

        # run slurm script
        slurm_cmds = []
        for i in range(self.NUM_GPUS):
            slurm_cmd = f"sbatch ../../jobs/detector_w1.sub {self.DAY_OF_COVERAGE} {i}"
            slurm_cmds.append(slurm_cmd)

        for slurm_cmd in slurm_cmds:
            slurm_cmd = shlex.split(slurm_cmd)
            # subprocess.run(slurm_cmd)

    def queue_detect_tasks(self, n_workers=4, write_frames_lists=True):
        # can fit 16 detection tasks on a single GPU
        if write_frames_lists:
            frames_dirs = glob.glob(
                f"{self.TOP_LEVEL_DIR}/{self.DAY_OF_COVERAGE}/*/2023*/"
            )
            frames_paths_list = self.write_n_frames_list(
                frames_dirs,
                n_workers * self.MAX_TASKS_PER_GPU,
                f"{self.OUTPUT_DIR}/{self.DAY_OF_COVERAGE}/frames_lists",
            )
            batch_paths = self.create_batch_symlinks(
                frames_paths_list,
                f"{self.OUTPUT_DIR}/{self.DAY_OF_COVERAGE}/frames_lists",
            )
        else:
            batch_paths = [
                f"{self.OUTPUT_DIR}/{self.DAY_OF_COVERAGE}/frames_lists/chunk_{idx}"
                for idx in range(n_workers * self.MAX_TASKS_PER_GPU)
            ]

        self.run_detect_scripts()


def ui_wrapper(TOP_LEVEL_DIR, DAY_OF_COVERAGE):
    detector = YOLO_Detector(
        TOP_LEVEL_DIR=TOP_LEVEL_DIR, DAY_OF_COVERAGE=DAY_OF_COVERAGE
    )
    detector.queue_detect_tasks(n_workers=detector.NUM_GPUS)


if __name__ == "__main__":
    fire.Fire(ui_wrapper)
