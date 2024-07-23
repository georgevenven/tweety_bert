import numpy as np
import pandas as pd
import os

def label_spectrogram(npz_dir, annotation_csv, default_sample_rate=44100, NFFT=1024, step_size=128):
    # Load the annotations CSV
    annotations = pd.read_csv(annotation_csv)

    # Calculate seconds per time bin in the spectrogram
    seconds_per_time_bin = step_size / default_sample_rate
    print(seconds_per_time_bin)

    # Iterate over npz files
    for npz_file in os.listdir(npz_dir):
        if npz_file.endswith('.npz'):
            file_path = os.path.join(npz_dir, npz_file)
            data = np.load(file_path, allow_pickle=True)
            spectrogram_data = data['s']  # Assuming 's' is the correct key for the spectrogram

            # Extract the basename without the .npz extension
            base_filename = npz_file[:-6]
            base_filename += ".wav"  
            print(base_filename)
            # Get corresponding rows from the annotations DataFrame
            relevant_annotations = annotations[annotations['audio_file'].str.contains(base_filename)]

            if relevant_annotations.empty:
                print(f"No labels found for {npz_file}. File will not be saved.")
                continue

            # Initialize the label matrix as a 1D array with the same number of time frames as the spectrogram
            label_matrix = np.zeros(spectrogram_data.shape[1], dtype=int)

            # Group annotations by label and process each group
            grouped_annotations = relevant_annotations.groupby('label')
            for label, group in grouped_annotations:
                # Use onset_s and offset_s for seconds
                initial_onset_seconds = group['onset_s'].min()
                final_offset_seconds = group['offset_s'].max()

                # Convert to indices within the spectrogram
                onset_index = int(initial_onset_seconds / seconds_per_time_bin)
                offset_index = int(final_offset_seconds / seconds_per_time_bin)

                # Fill the label matrix with the label value within the event's time frame
                label_matrix[onset_index:offset_index] = label

            # Overwrite the old npz file with the new attribute
            np.savez(file_path, s=spectrogram_data, labels=label_matrix)

            # For demonstration, just print out confirmation
            print(f"Labels added for {npz_file}: Label Matrix Shape {label_matrix.shape}, Spectrogram Shape {spectrogram_data.shape}")


# Example usage
npz_directory = "/media/george-vengrovski/disk1/yarden_128step_test"
annotation_csv_path = '/media/george-vengrovski/disk2/canary_yarden/llb3_data/llb3_annot.csv'
label_spectrogram(npz_directory, annotation_csv_path)
