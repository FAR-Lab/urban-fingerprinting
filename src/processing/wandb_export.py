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
        # get ids 
        self.run_ids = [run.id for run in self.runs]
        self.run_configs = [run.config for run in self.runs]

        # get run names 
        self.run_names = [run.name for run in self.runs]
        print(self.run_names)

        # only keep finished runs 
        self.runs = [run for run in self.runs if run.state == "finished"]

        # all runs 
        all_run_history = [] 

        # get run metrics history 
        for run in self.runs: 
            run_metrics = run.history()
            run_metrics['run_name'] = run.name
            run_metrics['run_id'] = run.id
            os.makedirs("../data", exist_ok=True)
            run_metrics.to_csv(f"../data/{run.name}.csv")
            all_run_history.append(run_metrics)
        
        all_run_history = pd.concat(all_run_history)

        all_run_history.to_csv("../data/all_run_history.csv")
        
        
            
        

    
        

    def export(self, run_id, path):
        pass


if __name__ == "__main__": 
    wandb_export = WandBExport("urbanekg", "StreetFlooding-DynamicF1-NoValUpweight-RandomSearch")
    wandb_export.get_project_runs()