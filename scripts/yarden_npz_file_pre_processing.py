import os
import numpy as np
from tqdm import tqdm

# Directory
directory_path = '/media/george-vengrovski/Extreme SSD/yarden_data/llb16_data_matrices/llb16_data_matrices'

# Get the list of files in the directory
files = os.listdir(directory_path)

# Iterate over files in directory with tqdm
for file in tqdm(files, desc="Processing files"):
    file_path = os.path.join(directory_path, file)
    
    # Remove .mat files, keep .npz files
    if file.endswith('.mat'):
        os.remove(file_path)
    elif file.endswith('.npz'):
        # Load the .npz file
        npz_file = np.load(file_path, allow_pickle=True)
        
        #['t', 'f', 's', 'labels']

        labels = npz_file['labels'].reshape(-1)

        # Create vocalizations array
        vocalizations = np.ones(labels.shape)

        # Save the new .npz file with 's', 'labels', and 'vocalizations'
        np.savez(file_path, s=npz_file['s'], labels=labels, vocalizations=vocalizations)
