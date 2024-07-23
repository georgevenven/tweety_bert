import numpy as np
import os
import shutil
import random
from tqdm import tqdm

def train_test_subsetting(src, dst, params):
    """
    Generates train and validation folders for dataset subsetting.
    :param src: Source directory path.
    :param dst: Destination directory path where the folders will be created.
    :param params: Dictionary containing the following keys:
                   - 'n_iterations': Number of iterations to perform the random subsetting (default: 5).
                   - 'subset_file_numbers': List of subset file numbers to evaluate (default: [1]).
    """
    if not os.path.exists(src):
        raise ValueError(f"Source directory {src} does not exist.")
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)

    # Set default values if not provided in params
    n_iterations = params.get('n_iterations', 5)
    subset_file_numbers = params.get('subset_file_numbers', [1])

    # List all npz files in the source directory
    npz_files = [item for item in os.listdir(src) if item.endswith(".npz")]

    # Extract the name of the source directory for folder naming
    src_dir_name = os.path.basename(src)

    # Perform subsetting for each subset file number
    for subset_file_number in subset_file_numbers:
        print(f"Subset File Number: {subset_file_number}")

        for iteration in range(1, n_iterations + 1):
            print(f"Iteration {iteration}")

            # Randomly select files for the training set
            train_files = random.sample(npz_files, min(subset_file_number, len(npz_files)))
            # Use the remaining files for the validation set
            val_files = [file for file in npz_files if file not in train_files]

            # Create train and validation directories using the destination directory path
            train_dir = os.path.join(dst, f"{src_dir_name}_train_{subset_file_number}_iteration{iteration}")
            val_dir = os.path.join(dst, f"{src_dir_name}_val_{subset_file_number}_iteration{iteration}")

            # Copy train files to train directory
            os.makedirs(train_dir, exist_ok=True)
            for file in tqdm(train_files, desc="Copying train files"):
                src_path = os.path.join(src, file)
                dst_path = os.path.join(train_dir, file)
                shutil.copy2(src_path, dst_path)

            # Copy validation files to validation directory
            os.makedirs(val_dir, exist_ok=True)
            for file in tqdm(val_files, desc="Copying validation files"):
                src_path = os.path.join(src, file)
                dst_path = os.path.join(val_dir, file)
                shutil.copy2(src_path, dst_path)

# Usage
src = "/media/george-vengrovski/disk1/yarden_OG_llb3"
dst = "/media/george-vengrovski/disk1/supervised_eval_dataset"
params = {
    'n_iterations': 3,
    'subset_file_numbers': [1, 10, 100] # Specific file numbers instead of percentages
}
train_test_subsetting(src, dst, params)

# src = "/media/george-vengrovski/disk2/canary_yarden/llb16_npz_files"
# dst = "/media/george-vengrovski/disk1/supervised_eval_dataset"
# params = {
#     'n_iterations': 3,
#     'subset_file_numbers': [1, 10, 100] # Specific file numbers instead of percentages
# }
# train_test_subsetting(src, dst, params)

# src = "/media/george-vengrovski/disk2/canary_yarden/llb11_npz_files"
# dst = "/media/george-vengrovski/disk1/supervised_eval_dataset"
# params = {
#     'n_iterations': 3,
#     'subset_file_numbers': [1, 10, 100] # Specific file numbers instead of percentages
# }
# train_test_subsetting(src, dst, params)
