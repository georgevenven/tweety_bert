import os
import numpy as np
from tqdm import tqdm
import argparse
import random
import shutil

def count_timebins(dir_path):
    # Checking if the directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    timebins_dict = {}

    # Retrieve all npz files in the directory
    npz_files = [file for file in os.listdir(dir_path) if file.endswith('.npz')]

    # Iterate over each file in the directory with a progress bar
    for file in tqdm(npz_files, desc="Processing files"):
        file_path = os.path.join(dir_path, file)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the npz file
        with np.load(file_path, allow_pickle=True) as data:
            # Store the number of timebins for each file
            timebins_dict[file] = data['s'].shape[1]
    
    return timebins_dict
    
def generate_random_folds(timebins_dict, target_timebins, dir_path, temp_folds_path):
    files = list(timebins_dict.keys())
    random.shuffle(files)  # Shuffle files to ensure randomness

    folds = []
    current_fold = []
    current_timebins = 0

    for file in files:
        if current_timebins >= target_timebins:
            folds.append(current_fold)
            current_fold = []
            current_timebins = 0

        current_fold.append(file)
        current_timebins += timebins_dict[file]

    # Add the last fold if it has any files
    if current_fold:
        folds.append(current_fold)

    # Create or clear the temp_folds_path directory
    if os.path.exists(temp_folds_path):
        shutil.rmtree(temp_folds_path)
    os.makedirs(temp_folds_path)

    fold_paths = []  # List to store paths to fold directories

    # Move files into fold directories
    for i, fold in enumerate(folds):
        fold_dir = os.path.abspath(os.path.join(temp_folds_path, f'fold_{i+1}'))
        os.makedirs(fold_dir, exist_ok=True)
        fold_paths.append(fold_dir)  # Add fold directory path to the list
        for file in fold:
            src = os.path.join(dir_path, file)
            dst = os.path.join(fold_dir, file)
            if os.path.exists(src):
                shutil.move(src, dst)
            else:
                print(f"File not found during move: {src}")

    # Debugging: Print the fold paths
    print("Fold directories created:", fold_paths)

    # Change this line to return a newline-separated string
    return '\n'.join(fold_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count timebins in npz files and generate random folds.')
    parser.add_argument('--dir_path', type=str, help='Path to the directory containing npz files')
    parser.add_argument('--target_timebins', type=int, default=1000, help='Target number of timebins per fold')
    parser.add_argument('--temp_folds_path', type=str, help='Path to create temporary fold directories')
    args = parser.parse_args()

    timebins_dict = count_timebins(args.dir_path)
    for filename, timebins in timebins_dict.items():
        print(f"{filename}: {timebins} timebins")

    fold_paths = generate_random_folds(timebins_dict, args.target_timebins, args.dir_path, args.temp_folds_path)
    # Print each fold path on a new line
    print("FOLD_PATHS_START")
    print(fold_paths)
    print("FOLD_PATHS_END")
