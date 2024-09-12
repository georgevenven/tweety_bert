import soundfile as sf
import numpy as np
import os
import shutil
from tqdm import tqdm

def has_audio(file_path, threshold=0):
    with sf.SoundFile(file_path, 'r') as wav_file:
        data = wav_file.read(dtype='int16')
        if wav_file.channels > 1:
            data = data[:, 0]
    return np.max(np.abs(data)) > threshold

def process_wav_files(file_paths, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files_with_audio = 0
    files_without_audio = 0

    for directory in tqdm(file_paths, desc="Processing directories"):
        for root, _, files in os.walk(directory):
            for file in tqdm(files, desc=f"Processing files in {os.path.basename(root)}", leave=False):
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(root, file)
                    if has_audio(file_path):
                        shutil.copy(file_path, output_folder)
                        files_with_audio += 1
                    else:
                        files_without_audio += 1

    return files_with_audio, files_without_audio

# List of WAV file paths to check
file_paths = [
    "/media/george-vengrovski/disk2/wolf_stuff/YNP Howl Clips/Breeding (Jan-March)",
    "/media/george-vengrovski/disk2/wolf_stuff/YNP Howl Clips/Nonbreeding",
    "/media/george-vengrovski/disk2/wolf_stuff/noise (1) (1)/noise",
    "/home/george-vengrovski/Downloads/Clips"
    # Add more file paths here
]

output_folder = "/media/george-vengrovski/disk2/wolf_stuff/files_with_audio"

files_with_audio, files_without_audio = process_wav_files(file_paths, output_folder)

print(f"Files with audio: {files_with_audio}")
print(f"Files without audio: {files_without_audio}")
