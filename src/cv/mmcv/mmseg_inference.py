# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/14/2023

# PREREQUISITES
# bdd100k-mmseg conda environment MUST BE ACTIVATED

import os 
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.logger import setup_logger
from mmseg.apis import MMSegInferencer


class UKG_MMSegInferencer(MMSegInferencer):
    def __init__(self, *args, **kwargs):
        super(MMSegInferencer, self).__init__(*args, **kwargs)
        self.log = setup_logger("UKG MMSegInferencer")
        self.log.success("UKG MMSegInferencer initialized.")



if __name__ == '__main__': 
    ukgmsi = UKG_MMSegInferencer()