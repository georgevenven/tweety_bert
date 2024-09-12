import os
import numpy as np
from zipfile import BadZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file_pair(file1_path, file2_path):
    try:
        with np.load(file1_path) as data1, np.load(file2_path) as data2:
            updated_data = {key: data2[key] for key in data2.files}
            
            if 'song' in data1 and 'vocalization' in updated_data:
                updated_data['vocalization'] = data1['song']
                updated_data['labels'] = data1['song']
                
                np.savez(file2_path, **updated_data)
                return f"Updated {os.path.basename(file2_path)} with song data from {os.path.basename(file1_path)}"
            else:
                return f"Required arrays not found in {os.path.basename(file1_path)} or {os.path.basename(file2_path)}"
    except BadZipFile:
        os.remove(file2_path)
        return f"BadZipFile error encountered for {os.path.basename(file2_path)}. Deleted and skipped this file."
    except Exception as e:
        return f"Error processing {os.path.basename(file2_path)}: {str(e)}. Skipped this file."

def match_and_update_files(folder1, folder2):
    file_pairs = []
    file2_dict = {f.split('_')[0] + '_' + f.split('_')[1]: f for f in os.listdir(folder2) if f.endswith('.npz')}
    
    for file1 in os.listdir(folder1):
        if file1.endswith('.npz'):
            prefix = '_'.join(file1.split('_')[:2])
            if prefix in file2_dict:
                file_pairs.append((
                    os.path.join(folder1, file1),
                    os.path.join(folder2, file2_dict[prefix])
                ))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file_pair, f1, f2) for f1, f2 in file_pairs]
        for future in as_completed(futures):
            print(future.result())

# Usage
folder1 = '/media/george-vengrovski/Rose-SSD/to_be_labeled_specs_better_Specs'
folder2 = '/media/george-vengrovski/Rose-SSD/high_quality_dataset_to_be_labeled'
match_and_update_files(folder1, folder2)