import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            arrays = dict(data)
            
            if 'vocalization' in arrays:
                # if all the vocalization is 0, then delete the file
                if np.all(arrays['vocalization'] == 0):
                    os.remove(file_path)
                    return False
            
            if 'vocalization' in arrays:
                arrays['labels'] = np.zeros(len(arrays['vocalization']))
            
            np.savez(file_path, **arrays)
            return True
    except Exception:
        return False

def process_npz_files(folder_path, max_workers=16):
    npz_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.npz')]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in npz_files]
        
        successful = 0
        failed = 0
        
        with tqdm(total=len(npz_files), desc="Processing .npz files") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
    
    print(f"Processing complete. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    folder_path = "/media/george-vengrovski/Extreme SSD/5509_data/USA_5509_Specs"
    process_npz_files(folder_path)