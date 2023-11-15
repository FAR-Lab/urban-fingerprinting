# FARLAB - UrbanECG 
# Developer: @mattwfranchi
# Last Edited: 11/08/2023 

# This script houses a driver for the train-test split on G class. 

# Module Imports
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
)

from src.utils.train_test_split_on_G import TTSplit_G


if __name__ == '__main__': 

    ttsplit_G = TTSplit_G(["2023-09-29"], output_prefix="flooding_train_test_split")
    ttsplit_G()