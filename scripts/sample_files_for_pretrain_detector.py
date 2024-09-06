import os
import random
import shutil

def sample_wav_files(source_folder, destination_folder, num_samples):
    # List to store all wav file paths
    all_wav_files = []

    # Walk through the directory structure
    for bird_id in os.listdir(source_folder):
        bird_path = os.path.join(source_folder, bird_id)
        if os.path.isdir(bird_path):
            for day_id in os.listdir(bird_path):
                day_path = os.path.join(bird_path, day_id)
                if os.path.isdir(day_path):
                    for file in os.listdir(day_path):
                        if file.endswith('.wav'):
                            all_wav_files.append(os.path.join(day_path, file))

    # Randomly select the specified number of files
    selected_files = random.sample(all_wav_files, min(num_samples, len(all_wav_files)))

    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Move selected files to the destination folder
    for file in selected_files:
        shutil.move(file, destination_folder)

    print(f"Moved {len(selected_files)} files to {destination_folder}")

# Example usage
source_folder = '/media/george-vengrovski/disk2/canary/unsorted'
destination_folder = '/media/george-vengrovski/disk2/training_song_detector/pretrain_dataset'
num_samples = 100000

sample_wav_files(source_folder, destination_folder, num_samples)