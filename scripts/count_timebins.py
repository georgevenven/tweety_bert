import os
import numpy as np
from tqdm import tqdm

dir_path = '/home/george-vengrovski/Documents/data/llb16_data_matrices'

# Checking if the directory exists
if not os.path.exists(dir_path):
    raise FileNotFoundError(f"Directory not found: {dir_path}")

timebins = 0 

# Retrieve all npz files in the directory
npz_files = [file for file in os.listdir(dir_path) if file.endswith('.npz')]

# Iterate over each file in the directory with a progress bar
for file in tqdm(npz_files, desc="Processing files"):
    file_path = os.path.join(dir_path, file)
    
    # Load the npz file
    with np.load(file_path, allow_pickle=True) as data:
        # Storing data in dictionary
        timebins += data['s'].shape[1]

print(timebins)
