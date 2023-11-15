# FARLAB - UrbanECG 
# Developer: @mattwfranchi, with help from Github Copilot 
# Last Edited: 11/06/2023 

# This script houses a driver for the clip zero shot classification task on a set of images. 

# Module Imports 
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cv.clip_zeroshot import CLIPZeroShot


# Driver 

if __name__ == '__main__': 
    frames_dir = sys.argv[1] 
    
    # take output prefix as second sysargv
    output_prefix = sys.argv[2]

    # take rest of sysargv as choices 
    choices = sys.argv[3:]

    # coerce choices to list of strings 
    choices = [str(choice) for choice in choices]

    # drop single or double quotes from choices
    choices = [choice.replace('"', '').replace("'", "") for choice in choices]

    # drop empty strings from choices
    choices = [choice for choice in choices if choice != '']
    
    print(choices)

    # Initialize the CLIPZeroShot class 
    clipzs = CLIPZeroShot(frames_dir, output_prefix, choices=choices)

    # Run the zero shot classification task
    clipzs.zeroshot_all()