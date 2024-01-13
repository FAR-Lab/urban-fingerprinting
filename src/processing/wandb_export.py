# FARLAB - UrbanECG Project 
# Developer: @mattwfranchi 
# Last Edited: 1/13/2024 

import os 
import wandb 

import pandas as pd

class WandBExport:
    def __init__(self, org_name, proj_name): 
        self.proj_name = proj_name 
        self.org_name = org_name
        self.api = wandb.Api()
       

        

        self.runs = []  


    def get_project_runs(self): 
        self.runs = self.api.runs(self.proj_name)
        print(self.runs)
        # get ids 
        self.run_ids = [run.id for run in self.runs]
        self.run_configs = [run.config for run in self.runs]

        # get run names 
        self.run_names = [run.name for run in self.runs]
        print(self.run_names)

        # get run metrics
        self.run_metrics = [run.summary_metrics for run in self.runs]

        self.run_metrics = pd.DataFrame(self.run_metrics)
        
        self.run_configs = pd.DataFrame(self.run_configs)

        self.run_metrics.insert(0, "run_name", self.run_names)

        self.run_data = pd.concat([self.run_metrics, self.run_configs, ], axis=1)

        dict_cols = [col for col in self.run_data.columns if self.run_data[col].apply(lambda x: isinstance(x, dict)).any()]

        the_rest = [col for col in self.run_data.columns if col not in dict_cols]
        the_rest = self.run_data[the_rest]

        to_add = []
        for col in dict_cols: 
            
            expanded = pd.json_normalize(self.run_data[col])
            # prefix col name to every column in expanded 
            expanded.columns = [f"{col}.{sub_col}" for sub_col in expanded.columns]
            to_add.append(expanded)

        to_add.insert(0, the_rest)
        self.run_data = pd.concat(to_add, axis=1)

        print(self.run_data)
        print(self.run_data.columns.values)

        self.run_data.to_csv("wandb_export.csv")
        

    def export(self, run_id, path):
        pass


if __name__ == "__main__": 
    wandb_export = WandBExport("urbanekg", "street-flooding")
    wandb_export.get_project_runs()