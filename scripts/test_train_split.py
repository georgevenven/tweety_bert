import os
import shutil
import random
from tqdm import tqdm

def split_dataset(folder_path, train_ratio, train_folder_dest, test_folder_dest, move_files=False):
    """
    Splits the npz files in the given folder into train and test sets based on the specified ratio
    and either copies or moves them to specified train and test destination folders based on the move_files flag.

    Parameters:
    folder_path (str): The path to the folder containing the dataset.
    train_ratio (float): The ratio of npz files to be included in the train set.
    train_folder_dest (str): The path to the destination train folder.
    test_folder_dest (str): The path to the destination test folder.
    move_files (bool): If True, files will be moved instead of copied. Defaults to False.
    """
    # Create train and test directories in the specified destination folders
    os.makedirs(train_folder_dest, exist_ok=True)
    os.makedirs(test_folder_dest, exist_ok=True)

    # List all npz files in the source folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.npz')]

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate number of files for the train set
    train_size = int(len(all_files) * train_ratio)

    # Split files
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    # Either move or copy files to respective destination directories based on move_files flag
    for file in tqdm(train_files, desc="Processing train files"):
        src_file_path = os.path.join(folder_path, file)
        dest_file_path = os.path.join(train_folder_dest, file)
        if move_files:
            shutil.move(src_file_path, dest_file_path)
        else:
            shutil.copy2(src_file_path, dest_file_path)

    for file in tqdm(test_files, desc="Processing test files"):
        src_file_path = os.path.join(folder_path, file)
        dest_file_path = os.path.join(test_folder_dest, file)
        if move_files:
            shutil.move(src_file_path, dest_file_path)
        else:
            shutil.copy2(src_file_path, dest_file_path)

# Example usage with moving files
split_dataset('/media/george-vengrovski/Extreme SSD/5509_data/USA_5509_Specs', 0.8, '/media/george-vengrovski/Extreme SSD/5509_data/USA_5509_Train', '/media/george-vengrovski/Extreme SSD/5509_data/USA_5509_Test', move_files=False)