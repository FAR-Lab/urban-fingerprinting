# FARLAB - UrbanECG Project
# Developer: @mattwfranchi, with help from GitHub CoPilot
# Last Modified: 10/29/2023

# This script houses a utility class to generate samples of 311 complaint data.


# Module Imports
import os
import sys

sys.path.append(os.path.abspath(os.path.join("../")))
sys.path.append(os.path.abspath(os.path.join("../..")))
sys.path.append(os.path.abspath(os.path.join("../../..")))


from src.utils.logger import setup_logger

import pandas as pd


class NYC311Sampler:
    def __init__(self, path_to_311_complaints_data, write_mode):
        self.log = setup_logger(name=__name__)
        self.write_mode = write_mode
        if self.write_mode:
            self.log.info(
                "Write mode enabled. Will write sampled data to disk."
            )
        else:
            self.log.info(
                "Running in interactive mode. Data will not be saved to disk."
            )

        self.complaints_df = pd.read_csv(
            path_to_311_complaints_data, engine="pyarrow"
        )
        self.log.success(
            f"Read {len(self.complaints_df.index)} complaints from {path_to_311_complaints_data}"
        )
        self.log.info("Setup complete.")

    def sample(
        self,
        n=-1,
        col_to_sample_on="Complaint Type",
        value_to_filter_on="",
        seed=42,
    ):
        if value_to_filter_on == "":
            self.log.info(
                f"No value to filter on provided. Sampling {n} complaints from overall dataset..."
            )
            if n == -1: 
                n = len(self.complaints_df.index)
            return self.complaints_df.sample(n=n, random_state=seed)
        else:
            try:
                subset = self.complaints_df[
                    self.complaints_df[col_to_sample_on] == value_to_filter_on
                ]
                if n == -1: 
                    n = len(subset.index)
            except Exception as e:
                self.log.error(
                    f"Failed to filter on {col_to_sample_on} == {value_to_filter_on}: {e}"
                )
                return None

            self.log.info(
                f"Sampling {n} complaints from {col_to_sample_on} == {value_to_filter_on}..."
            )

            if self.write_mode:
                subset.sample(n=n, random_state=seed).to_csv(
                    f"../../data/coords/{col_to_sample_on}_{value_to_filter_on.replace('/','-')}_{n}.csv",
                    index=False,
                )
            else:
                return subset.sample(n=n, random_state=seed)

    def samples_for_all_col_values(
        self, n=-1, col_to_sample_on="Complaint Type", seed=42
    ):
        values = self.complaints_df[col_to_sample_on].unique()
        self.log.info(f"Sampling from all values of {col_to_sample_on}...")
        if self.write_mode:
            for value in values:
                
                self.log.info(
                    f"Sampling {n} complaints from {col_to_sample_on} == {value}..."
                )
                self.sample(
                    n=n,
                    col_to_sample_on=col_to_sample_on,
                    value_to_filter_on=value,
                    seed=seed,
                )
        else:
            return {
                value: self.sample(
                    n=n,
                    col_to_sample_on=col_to_sample_on,
                    value_to_filter_on=value,
                    seed=seed,
                )
                for value in values
            }