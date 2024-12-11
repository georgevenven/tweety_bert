"""
This script analyzes NPZ files containing spectrograms from UMAP folds to calculate statistics
about the number of time bins (first dimension) across all folds. It loops through all NPZ 
files in a specified directory, extracts the shape of each spectrogram stored under key 's',
and calculates the range and average of time bins across all folds. This information is 
useful for understanding the temporal variation in the spectrograms and ensuring consistency
across different UMAP fold files.
"""

import os
import numpy as np

# Directory containing the NPZ files
data_dir = "/media/george-vengrovski/George-SSD/folds_for_paper_llb"  # Replace with your actual directory path

# List to store first dimensions
first_dimensions = []

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.npz'):
        # Construct full file path
        file_path = os.path.join(data_dir, filename)
        
        # Load the NPZ file
        data = np.load(file_path)
        
        # Get and store the first dimension
        if 's' in data:
            spec_shape = data['s'].shape
            first_dimensions.append(spec_shape[0])
            print(f"File: {filename} - Spectrogram shape: {spec_shape}")
        else:
            print(f"File: {filename} - No spectrogram ('s') found in data")

# Calculate statistics
if first_dimensions:
    min_dim = min(first_dimensions)
    max_dim = max(first_dimensions)
    avg_dim = sum(first_dimensions) / len(first_dimensions)
    
    print("\nStatistics for first dimensions:")
    print(f"Range: {min_dim} to {max_dim}")
    print(f"Average: {avg_dim:.2f}")
