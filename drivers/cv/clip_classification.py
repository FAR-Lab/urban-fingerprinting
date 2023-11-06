# FARLAB - UrbanECG 
# Developer: @mattwfranchi, with help from Github Copilot 
# Last Edited: 11/06/2023 

# This script houses a driver for the clip zero shot classification task on a set of images. 

# Module Imports 
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv.clip_zeroshot import CLIPZeroShot


# Driver 

if __name__ == '__main__': 
    frames_dir = sys.argv[1] 
    
    # take rest of sysargv as choices 
    choices = sys.argv[2:]

    # coerce choices to list of strings 
    choices = [str(choice) for choice in choices]

    # Initialize the CLIPZeroShot class 
    clipzs = CLIPZeroShot(frames_dir, choices=choices)