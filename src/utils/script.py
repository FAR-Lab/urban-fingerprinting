# Function to count all files in directory
def count_files_in_dir(directory):
    # Check if directory exists
    if not os.path.exists(directory):
        raise ValueError("Directory does not exist.")

    # Check if directory is a directory
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} is not a directory.")

    # Glob 'directory' for all files
    files = glob.glob(os.path.join(directory, "*"))

    # Return length of files list
    return len(files)
