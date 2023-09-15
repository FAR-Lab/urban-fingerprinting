# FARLAB -- UrbanECG Project
# Dev: Matt Franchi, help from GitHub Copilot 
# Last Edited: 09/14/2023 

# This script is used to generate a CSV of metadata and frame counts per day of coverage in the Nexar dashcam image dataset. 

import glob 
import os
from concurrent.futures import ProcessPoolExecutor
import logging 
import csv 
import datetime
import fire 

# Set up logging
logging.basicConfig(level=logging.INFO)
# set logging name
logger = logging.getLogger(__name__)


# Constant for number of works to use in parallel, should be equal to number of cores on machine
NUM_CORES = os.getenv("SLURM_CPUS_ON_NODE")
# Check if none
if NUM_CORES is None:
    # Set to 8
    NUM_CORES = 8


def get_frame_counts_worker(folder):
    """Get frame counts for all videos in a given folder. 

    Args:
        folder (str): Path to directory containing videos. 

    Returns:
        dict: Integer of frame (image) counts in the folder
    """

    # Check if folder exists
    if not os.path.exists(folder):
        raise ValueError("Folder does not exist.")

    # Check if folder is a directory
    if not os.path.isdir(folder):
        raise ValueError(f"Folder {folder} is not a directory.")
    
    # Glob 'folder' for all .jpg files 
    files = glob.glob(os.path.join(folder, "*/*.jpg"))

    # Return length of files list
    return len(files)


def get_md_counts_worker(md_csv):

    # If md_csv is None, return 0
    if md_csv is None:
        return 0

    # Check if md_csv exists
    if not os.path.exists(md_csv):
        logger.warning(f"Metadata CSV: {md_csv} does not exist.")
    
    # Open md_csv as a file object
    with open(md_csv, "r") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader)
        # Count number of rows
        num_rows = sum(1 for row in reader)
    
    # close file object
    f.close()


    # Return number of rows
    return num_rows

   
def get_data_counts(day_of_coverage, num_workers=8):
    # Glob all h3-6 hexagon directories within the given day of coverage 
    hex_dirs = glob.glob(os.path.join(day_of_coverage, "*"))
    # remove any non-directories 
    hex_dirs = [x for x in hex_dirs if os.path.isdir(x)]

    logger.info(f"Number of hex_dirs: {len(hex_dirs)}")

    # Allocate a ProcessPoolExecutor with num_workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map get_frame_counts_worker to all hex_dirs
        frame_counts = executor.map(get_frame_counts_worker, hex_dirs)

        # create copy of hex_dirs that points to metadata CSVs
        hex_dirs_md = hex_dirs
        # glob the csv in each hex_dir
        for i, hex_dir in enumerate(hex_dirs_md):
            # get list of csvs in hex_dir
            csvs = glob.glob(os.path.join(hex_dir, "*.csv"))
            # check if there is more than one csv
            if len(csvs) > 1:
                # raise error
                raise ValueError("More than one CSV in hex_dir.")
            # check if there is no csv
            elif len(csvs) == 0:
                # log warning 
                logger.warning(f"No CSV in hex_dir: {hex_dir}")
                # set hex_dirs_md[i] to None
                hex_dirs_md[i] = None
            else:
                # grab path of first csv 
                hex_dirs_md[i] = csvs[0]

        # Map get_md_counts_worker to all hex_dirs_md
        md_counts = executor.map(get_md_counts_worker, hex_dirs_md)

    # Return a dictionary of totals
    return {
        "total_frames": sum(frame_counts),
        "total_md": sum(md_counts)
    }


# Function to create date range in specific format 
def daterange(start_date, end_date, format="%Y-%m-%d"):
    """Create a date range in a specific format. 

    Args:
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        format (str): Format to return dates in. 

    Yields:
        str: Date in format (default YYYY-MM-DD)
    """
    # Convert start_date to datetime object
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    # Convert end_date to datetime object
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    # Iterate through date range
    for n in range(int((end_date - start_date).days)):
        # Yield date in format
        yield (start_date + datetime.timedelta(n)).strftime(format)


# Function to get data counts for a given date range
def get_data_counts_in_range(root, start_date, end_date, num_workers=8):
    """Get data counts for a given date range. 

    Args:
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        num_workers (int, optional): Number of workers to use in parallel. Defaults to 8.

    Returns:
        dict: Dictionary of total frames and total metadata counts for each date in the range.
    """
    # Create empty dictionary to store data counts
    data_counts = {}

    # Iterate through date range
    for date in daterange(start_date, end_date):
        # Create path to day of coverage
        day_of_coverage = os.path.join(root, date)
        # Get data counts for date
        data_counts[date] = get_data_counts(day_of_coverage, num_workers=num_workers)
        # Log date and data counts
        logger.info(f"Date: {date}, Data Counts: {data_counts[date]}")
    
    # Return data_counts
    return data_counts


# Function to turn data_counts into a CSV
def data_counts_to_csv(data_counts, csv_path):
    """Turn data counts into a CSV. 

    Args:
        data_counts (dict): Dictionary of data counts.
        csv_path (str): Path to save CSV to. 
    """
    # Open csv_path as a file object
    with open(csv_path, "w") as f:
        # Create csv writer
        writer = csv.writer(f)
        # Write header
        writer.writerow(["date", "total_frames", "total_md"])
        # Iterate through data_counts
        for date, counts in data_counts.items():
            # Write row
            writer.writerow([date, counts["total_frames"], counts["total_md"]])
    # Close file object
    f.close()


if __name__ == "__main__":
    # log number of cores, and start message 
    logger.info("Starting frame count script...")
    logger.info(f"Number of cores: {NUM_CORES}")

    data_counts = fire.Fire(get_data_counts_in_range)

    # prompt user if they want to save data counts to csv
    save_to_csv = input("Save data counts to CSV? (y/n): ")
    # check if user wants to save to csv (case insensitive)
    if save_to_csv.lower() == "y":
        # get path to save csv to
        csv_path = input("Enter path to save CSV to: ")
        # check if csv_path exists
        if os.path.exists(csv_path):
            # raise error
            raise ValueError(f"CSV Path: {csv_path} already exists.")
        # save data counts to csv
        data_counts_to_csv(data_counts, csv_path)
        # log success message
        logger.info(f"Data counts saved to CSV: {csv_path}")
    else:
        # log message
        logger.info("Data counts not saved to CSV.")
    # log end message
    logger.info("Frame count script complete.")


    

   


