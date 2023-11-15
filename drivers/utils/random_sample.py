# FARLAB - UrbanECG
# Developer: @mattwfranchi, with help from GitHub Copilot
# Last Edited: 11/07/2023

# This script houses a driver to randomly sample a dataset of images. 

# Module Imports
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

from user.params.io import TOP_LEVEL_FRAMES_DIR

from src.utils.logger import setup_logger
from src.utils.timer import timer

from src.utils.data_pull.random_sample_DoC import ImagePull

if __name__ == '__main__': 

    @timer 
    def main():
        ip = ImagePull(TOP_LEVEL_FRAMES_DIR, sys.argv[2])
        ip(sys.argv[1], sys.argv[3])

    main()

