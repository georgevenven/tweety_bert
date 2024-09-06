import os
import numpy as np

def delete_corrupted_npz_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npz'):
                file_path = os.path.join(dirpath, filename)
                
                # Check if file is 0 bytes
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    print(f"Deleted 0 byte file: {file_path}")
                    continue
                
                # Try to open the file
                try:
                    with np.load(file_path) as npz_file:
                        # If we can load the file, it's not corrupted
                        pass
                except Exception as e:
                    os.remove(file_path)
                    print(f"Deleted corrupted file: {file_path}")
                    print(f"Error: {str(e)}")

if __name__ == "__main__":
    root_directory = "/media/george-vengrovski/disk2/training_song_detector/pretrain_dataset_train"
    delete_corrupted_npz_files(root_directory)
