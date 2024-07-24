import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_file(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        
        if 's' in data.keys():
            s_data = data['s']
            if s_data.ndim > 1:
                # Use boolean indexing to filter out rows with all zeros
                mask = ~np.all(s_data == 0, axis=1)
                processed_s_data = s_data[mask]

                # Save the processed data back to the .npz file
                new_data = {key: data[key] for key in data.keys()}
                new_data['s'] = processed_s_data
                np.savez(file_path, **new_data)
                return f"Processed and saved {file_path}"
            else:
                return f"The 's' data in {file_path} is not 2-dimensional"
        else:
            return f"No 's' key found in {file_path}"
    except Exception as e:
        return f"Failed to process {file_path} due to {e}"

def process_npz_files(folder_path, max_workers=4):
    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.npz')]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, file_path): file_path for file_path in file_paths}
        for future in as_completed(futures):
            print(future.result())

# Set the folder path where the .npz files are located
folder_path = "/media/george-vengrovski/disk1/5288_specs_processed"

# Process the .npz files in the folder
process_npz_files(folder_path, max_workers=16)
