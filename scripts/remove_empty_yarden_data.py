import numpy as np
import os
from pathlib import Path

def process_directory(directory_path):
    """
    Process NPZ files in a directory to remove unlabeled YARDEN data files.
     
    Some YARDEN dataset files contain only unlabeled data (no labels of syllables),
    which can skew metrics and evaluations when training models. This script
    identifies and removes such files to ensure more accurate model assessment.
    The script recursively searches through the given directory and its subdirectories
    for .npz files, checks their labels, and removes files that only contain
    unlabeled data points (labels of 0 or -1).
    
    Args:
        directory_path (str): Path to the directory to process
        
    Returns:
        None: Prints deletion status and final count of removed files
    """
    deleted_count = 0
    
    # Recursively find all .npz files
    for npz_file in Path(directory_path).rglob('*.npz'):
        try:
            # Load the NPZ file
            data = np.load(npz_file)
            
            # Get unique labels
            unique_labels = np.unique(data['labels'])
            data.close()  # Close the file to prevent memory leaks
            
            # Check if only contains 0 or -1
            if set(unique_labels).issubset({0, -1}):
                os.remove(npz_file)
                deleted_count += 1
                print(f"Deleted: {npz_file}")
                
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    print(f"\nTotal files deleted: {deleted_count}")

if __name__ == "__main__":
    # Replace with your directory path
    directory_path = "/media/george-vengrovski/George-SSD/llb_stuff"
    process_directory(directory_path)