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
from user.params.io import INSTALL_DIR 
import user.params.io as iofile

import importlib 


# implement same functionality in function called set_project 
# that takes in a project name and sets PROJECT_NAME to that name
# if no argument is given, we reset PROJECT_NAME to the default value of "default"
def set_project(project_name="default"):
    logger = setup_logger("toggle_project")
    logger.setLevel("INFO")

    # Set path to io.py
    io_path = os.path.join(INSTALL_DIR, "user", "params", "io.py")

    # Read in io.py
    with open(io_path, "r") as io_file:
        io_lines = io_file.readlines()

    # Find line with PROJECT_NAME
    for i, line in enumerate(io_lines):
        if "PROJECT_NAME" in line:
            io_lines[i] = f"PROJECT_NAME = '{project_name}'\n"

    # Write out io.py
    with open(io_path, "w") as io_file:
        io_file.writelines(io_lines)

    # Print confirmation message
    importlib.reload(iofile)
    logger.success(f"PROJECT_NAME set to {project_name} in {io_path}")

def reset_project():
    set_project()

# Define main function
if __name__ == '__main__':

    logger = setup_logger("toggle_project")
    logger.setLevel("INFO")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Set PROJECT_NAME for output storage to the desired project name.")
    parser.add_argument("-n", "--name", type=str, default="default", help="Name of project to set PROJECT_NAME to.")
    args = parser.parse_args()

    # Set project name variable
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
