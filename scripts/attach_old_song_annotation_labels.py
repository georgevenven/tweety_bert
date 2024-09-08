import os
import numpy as np

def match_and_update_files(folder1, folder2):
    for file1 in os.listdir(folder1):
        if file1.endswith('.npz'):
            # Extract the part before the second underscore
            prefix = '_'.join(file1.split('_')[:2])
            
            # Find matching file in folder2
            for file2 in os.listdir(folder2):
                if file2.startswith(prefix) and file2.endswith('.npz'):
                    # Load both files
                    with np.load(os.path.join(folder1, file1)) as data1, np.load(os.path.join(folder2, file2)) as data2:
                        # Create a new dictionary with updated data
                        updated_data = {key: data2[key] for key in data2.files}
                        
                        # Update 'vocalization' array with 'song' array from file1
                        if 'song' in data1 and 'vocalization' in updated_data:
                            updated_data['vocalization'] = data1['song']
                            updated_data['labels'] = data1['song']
                            
                            # Save updated file2
                            np.savez(os.path.join(folder2, file2), **updated_data)
                            print(f"Updated {file2} with song data from {file1}")
                        else:
                            print(f"Required arrays not found in {file1} or {file2}")
                    break
            else:
                print(f"No matching file found for {file1}")

# Usage
folder1 = '/media/george-vengrovski/disk2/training_song_detector/wav_and_npz_files'
folder2 = '/media/george-vengrovski/disk2/training_song_detector/Ellen_Specs/train'
match_and_update_files(folder1, folder2)