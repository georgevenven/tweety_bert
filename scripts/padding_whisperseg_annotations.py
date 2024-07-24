import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

specs_dir = "/media/george-vengrovski/disk1/5288_specs"
output_dir = "/media/george-vengrovski/disk1/5288_specs_processed"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_file(file):
    if file.endswith(".npz"):
        input_path = os.path.join(specs_dir, file)
        output_path = os.path.join(output_dir, file)

        try:
            # Load the npz file with allow_pickle=True
            try:
                with np.load(input_path, allow_pickle=True) as data:
                    spectrogram = data['s']
                    vocalization = data['vocalization']
                    labels = data.get('labels', np.zeros_like(vocalization))  # Create labels if not present
            except Exception as e:
                print(f"Error loading file {file}: {str(e)}")
                return

            # Skip file if all vocalization numbers are zero
            if np.all(vocalization == 0):
                return

            # Optimize padding operation
            indices = np.where(vocalization == 1)[0]
            padded_vocalization = np.zeros_like(vocalization)
            start = np.maximum(indices[:, np.newaxis] - 50, 0)
            end = np.minimum(indices[:, np.newaxis] + 51, len(vocalization))
            for s, e in zip(start.flat, end.flat):
                padded_vocalization[s:e] = 1

            # Save as npz file
            np.savez_compressed(output_path, s=spectrogram, vocalization=padded_vocalization, labels=labels)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

def process_batch(files):
    for file in files:
        process_file(file)

# Get the list of files
files = [file for file in os.listdir(specs_dir) if file.endswith(".npz")]

# Define batch size
batch_size = 100  # Adjust based on your system's capabilities
batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]

# Use ProcessPoolExecutor to process batches in parallel
with ProcessPoolExecutor() as executor:
    list(tqdm(executor.map(process_batch, batches), total=len(batches)))

print("Processing complete. Output files are in:", output_dir)