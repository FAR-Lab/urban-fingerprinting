# FARLAB - UrbanECG 
# Developer: @mattwfranchi 
# Last Edited: 12/11/2023 

# This script sets the parameter PROJECT_NAME in /src/user/params/io.py to the desired project name.
# If no argument is given, we reset PROJECT_NAME to the default value of "default".

# Import Packages
import os
import sys
import argparse

sys.path.append(os.path.join("..", ".."))

from src.utils.logger import setup_logger


# Define main function
if __name__ == '__main__':

    logger = setup_logger("toggle_project")
    logger.setLevel("INFO")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Set PROJECT_NAME for output storage to the desired project name.")
    parser.add_argument("-n", "--name", type=str, default="default", help="Name of project to set PROJECT_NAME to.")
    args = parser.parse_args()

    # Set project name
    PROJECT_NAME = args.name

    # Set path to io.py
    io_path = os.path.join("..", "..", "user", "params", "io.py")

    # Read in io.py
    with open(io_path, "r") as io_file:
        io_lines = io_file.readlines()

    # Find line with PROJECT_NAME
    for i, line in enumerate(io_lines):
        if "PROJECT_NAME" in line:
            io_lines[i] = f"PROJECT_NAME = '{PROJECT_NAME}'\n"

    # Write out io.py
    with open(io_path, "w") as io_file:
        io_file.writelines(io_lines)

    # Print confirmation message
    logger.success(f"PROJECT_NAME set to {PROJECT_NAME} in {io_path}.")
