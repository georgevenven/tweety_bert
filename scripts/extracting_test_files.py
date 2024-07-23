import os
import shutil

# Directory paths
source_dir = "/media/george-vengrovski/disk1/combined_song_data_1_test"
file_dir = "/media/george-vengrovski/disk2/canary_yarden/combined_yarden_specs"
destination_dir = "files/goliath_llb3_eval"

# Get a list of filenames starting with "llb3" from the source directory
filenames = [f for f in os.listdir(source_dir) if f.startswith("llb3")]

# Copy the files from the file directory to the destination directory
for filename in filenames:
    # Modify the filename to include ".wav" before ".npz"
    modified_filename = filename.replace(".npz", ".wav.npz")
    
    source_path = os.path.join(file_dir, modified_filename)
    destination_path = os.path.join(destination_dir, modified_filename)
    
    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"Copied: {modified_filename}")
    else:
        print(f"File not found: {modified_filename}")