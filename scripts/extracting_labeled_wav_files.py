import pandas as pd
import shutil
import os

# Define the directories
csv_file_path = '/media/george-vengrovski/disk2/canary_yarden/llb3_data/llb3_annot.csv'  # Path to the directory containing the CSV file
dir2 = '/media/george-vengrovski/disk2/canary_yarden/llb3_data/llb3_songs'  # Directory where the audio files are located
dir3 = '/media/george-vengrovski/disk2/canary_yarden/llb3_files_with_reattached_labels'  # Destination directory for the audio files

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract the 'audio file' column
audio_files = df['audio_file'].tolist()

# Iterate over each audio file name
for audio_file in audio_files:
    # Construct the full path to the audio file in dir2
    source_path = os.path.join(dir2, audio_file)
    
    # Check if the file exists in dir2
    if os.path.exists(source_path):
        # Construct the destination path in dir3
        destination_path = os.path.join(dir3, audio_file)
        
        # Move the file from dir2 to dir3
        shutil.move(source_path, destination_path)
        print(f"Moved: {audio_file}")
    else:
        print(f"File not found: {audio_file}")
