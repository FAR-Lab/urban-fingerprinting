import pandas as pd
import matplotlib.pyplot as plt

import logging
from termcolor import colored

from glob import glob
from concurrent.futures import ProcessPoolExecutor
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from user.params.io import INSTALL_DIR, PROJECT_NAME

from user.params.data import IMG_ID

from tqdm import tqdm

import fire

import numpy as np
from collections import Counter
from functools import reduce
import operator
from itertools import chain

from joblib import Parallel, delayed


class ColorfulFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }

    def format(self, record):
        log_level = record.levelname
        msg = super().format(record)
        return colored(msg, self.COLORS.get(log_level, "green"))


def setup_logger():
    logger = logging.getLogger("urban_fingerprinting")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = ColorfulFormatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)
    return logger


class Parser:
    def __init__(self, DAY_OF_COVERAGE):
        self.log = setup_logger()

        self.NUM_CORES = int(os.getenv("SLURM_CPUS_ON_NODE") or 8)

        self.DAY_OF_COVERAGE = DAY_OF_COVERAGE

        self.PREDICTIONS_REGEX = (
            f"{INSTALL_DIR}/{PROJECT_NAME}/yolo/{DAY_OF_COVERAGE}/*/exp*/labels/*.txt"
        )
        self.IMAGES_REGEX = f"{INSTALL_DIR}/{PROJECT_NAME}/yolo/{DAY_OF_COVERAGE}/frames_lists/*/*.jpg"

        self.IMAGES_LIST = glob(self.IMAGES_REGEX)
        self.log.info(
            f"Number of images for {self.DAY_OF_COVERAGE}: {len(self.IMAGES_LIST)}"
        )
        self.PREDICTIONS_LIST = glob(self.PREDICTIONS_REGEX)
        self.log.info(
            f"Number of detected predictions for {self.DAY_OF_COVERAGE}: {len(self.PREDICTIONS_LIST)}"
        )

        self.ALL_PREDICTIONS = pd.DataFrame(self.PREDICTIONS_LIST)
        self.ALL_PREDICTIONS.columns = ["path"]
        self.ALL_PREDICTIONS[IMG_ID] = (
            self.ALL_PREDICTIONS["path"].str.split("/").str[-1].str.split(".").str[0]
        )
        self.ALL_PREDICTIONS = self.ALL_PREDICTIONS.set_index(IMG_ID)

        self.check_preds_paths()

    def quick_extract(self, file_path):
        # Read the file
        with open(file_path, "r") as file:
            lines = file.readlines()

            # Extract the values from the first column and count their occurrences
            first_column_values = [line.split()[0] for line in lines]
            counted_values = Counter(first_column_values).items()
            frame_id = file_path.split("/")[-1].split(".")[0]
            return {frame_id: dict(counted_values)}

    def quick_extract_worker(self, subset):
        counters = []
        for path in tqdm(subset):
            counters.append(self.quick_extract(path))
        return counters

    # Function to extract yhat from the given file path
    def extract_w_conf(self, file_path):
        d = pd.read_csv(
            file_path,
            sep=" ",
            names=["class_type", "dummy1", "dummy2", "dummy3", "dummy4", "conf"],
            engine="pyarrow",
        )

        d = d[["class_type", "conf"]]
        d = d.groupby("class_type").agg("size")

        # set all columns names to string versions of the class id
        d = d.to_frame().T
        d.columns = [str(x) for x in d.columns]

        d[IMG_ID] = file_path.split("/")[-1].split(".")[0]

        d = d.set_index(IMG_ID)

        d_dict = d.to_dict(orient="tight")
        del d
        return d_dict

    def check_preds_paths(self):
        len_before_drop = len(self.ALL_PREDICTIONS)
        existing_paths = self.ALL_PREDICTIONS["path"].apply(os.path.exists)
        self.ALL_PREDICTIONS = self.ALL_PREDICTIONS[existing_paths]
        len_after_drop = len(self.ALL_PREDICTIONS)
        if len_before_drop != len_after_drop:
            self.log.warning(
                f"Dropped {len_before_drop - len_after_drop} rows due to missing paths"
            )

    def parse(self):
        subsets = np.array_split(self.ALL_PREDICTIONS["path"].tolist(), 96)
        self.ALL_DETECTIONS = Parallel(n_jobs=self.NUM_CORES)(
            delayed(self.quick_extract_worker)(subset)
            for subset in tqdm(subsets, desc="Processing batches")
        )
        self.ALL_DETECTIONS = list(chain(*self.ALL_DETECTIONS))

        self.ALL_DETECTIONS = pd.concat(
            [
                pd.DataFrame(l)
                for l in tqdm(
                    self.ALL_DETECTIONS, desc="Creating detections dataframe..."
                )
            ],
            axis=1,
        ).T

        # drop path column
        self.ALL_PREDICTIONS = self.ALL_PREDICTIONS.drop(columns=["path"])
        # turn columns with numbers into ints, but leave string columns alone
        # turn column names with numbers into ints

        self.ALL_DETECTIONS.columns = self.ALL_DETECTIONS.columns.map(int)

        # sort columns by class id
        self.ALL_DETECTIONS = self.ALL_DETECTIONS.reindex(
            sorted(self.ALL_DETECTIONS.columns), axis=1
        )

        # make sure output dir exists
        os.makedirs(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{self.DAY_OF_COVERAGE}", exist_ok=True)

        self.ALL_PREDICTIONS.merge(
            self.ALL_DETECTIONS, left_index=True, right_index=True
        ).to_csv(f"{INSTALL_DIR}/{PROJECT_NAME}/df/{self.DAY_OF_COVERAGE}/detections.csv")

        if len(self.ALL_DETECTIONS) != len(self.ALL_PREDICTIONS):
            self.log.error(
                f"Number of detections ({len(self.ALL_DETECTIONS)}) does not match number of predictions ({len(self.ALL_PREDICTIONS)})"
            )

        self.log.info(f"Saved detections for {self.DAY_OF_COVERAGE} to disk.")


def ui_wrapper(doc):
    p = Parser(doc)
    p.parse()


if __name__ == "__main__":
    fire.Fire(ui_wrapper)
