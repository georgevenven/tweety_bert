import os
import numpy as np
from tqdm import tqdm

def get_directory_size(directory_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def convert_npz_to_uncompressed_npz(directory_path):
    # Ensure the directory path exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"The directory {directory_path} does not exist.")

    # Get initial disk space
    initial_size = get_directory_size(directory_path)
    print(f"Initial disk space: {initial_size / (1024 * 1024):.2f} MB")

    # Iterate over all files in the directory with a progress bar
    for filename in tqdm(os.listdir(directory_path), desc="Processing files"):
        if filename.endswith('.npz'):
            npz_file_path = os.path.join(directory_path, filename)
            
            # Load the .npz file
            data = np.load(npz_file_path)

            # Save the data as an uncompressed .npz file
            uncompressed_npz_file_path = os.path.join(directory_path, f"{filename[:-4]}_uncompressed.npz")
            np.savez(uncompressed_npz_file_path, **data)

            # Remove the uncompressed .npz file
            os.remove(npz_file_path)

    # Get final disk space
    final_size = get_directory_size(directory_path)
    print(f"Final disk space: {final_size / (1024 * 1024):.2f} MB")
    print(f"Disk space change: {(final_size - initial_size) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    # Example usage
    directory_path = "/home/george-vengrovski/Documents/data/llb16_no_threshold_no_norm_test"
    convert_npz_to_uncompressed_npz(directory_path)