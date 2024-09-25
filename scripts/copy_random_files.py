import os
import random
import shutil

def copy_random_files(src_dir, dest_dir, n):
    # Ensure source directory exists
    if not os.path.isdir(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return

    # Ensure destination directory exists, create if not
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # Check if there are enough files to copy
    if len(files) < n:
        print(f"Not enough files to copy. Found {len(files)} files, but {n} were requested.")
        return

    # Select n random files
    random_files = random.sample(files, n)

    # Copy each selected file to the destination directory
    for file in random_files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))
        print(f"Copied {file} to {dest_dir}")

# Specify source and destination directories and number of files to copy
src_dir = '/media/george-vengrovski/disk2/zebra_finch/aws_specs'
dest_dir = '/media/george-vengrovski/disk2/training_song_detector/aws_zf_labeled'
n = 400  # Number of files to copy

copy_random_files(src_dir, dest_dir, n)
