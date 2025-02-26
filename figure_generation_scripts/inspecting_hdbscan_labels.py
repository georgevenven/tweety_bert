import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def plot_longest_segments_by_label(file_path, output_file_path):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # HDBSCAN labels
    hdbscan_colors = data["hdbscan_colors"].item()  # Load HDBSCAN colors

    unique_labels = np.unique(labels)
    max_segments = 4  # Maximum number of segments to display per label

    # Create a dictionary to store the longest segments for each label
    longest_segments = {label: [] for label in unique_labels}

    # Find the longest continuous segments for each label
    current_label = None
    start_idx = 0
    for i in range(len(labels)):
        if labels[i] != current_label:
            if current_label is not None:
                segment_length = i - start_idx
                longest_segments[current_label].append((start_idx, segment_length))
                # Sort and keep only the longest segments
                longest_segments[current_label].sort(key=lambda x: x[1], reverse=True)
                longest_segments[current_label] = longest_segments[current_label][:max_segments]
            current_label = labels[i]
            start_idx = i
    # Handle the last segment
    segment_length = len(labels) - start_idx
    longest_segments[current_label].append((start_idx, segment_length))
    longest_segments[current_label].sort(key=lambda x: x[1], reverse=True)
    longest_segments[current_label] = longest_segments[current_label][:max_segments]

    # Set up the figure for plotting
    num_rows = len(unique_labels)
    fig, axes = plt.subplots(nrows=num_rows, ncols=max_segments, figsize=(20, 5 * num_rows))
    
    if num_rows == 1:  # Adjust axes array if there's only one label
        axes = [axes]

    for label_idx, label in enumerate(unique_labels):
        for segment_idx, (start, length) in enumerate(longest_segments[label][:max_segments]):
            spec_slice = spec[start:start+length].T
            ax = axes[label_idx][segment_idx] if num_rows > 1 else axes[segment_idx]
            ax.imshow(spec_slice, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Label {label}, Length {length}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.show()

# File paths
file_path = "temp/UMAP_FILES/llb3_3440_2018_05_01_10_43_15_segment_0.npz"
output_file_path = "spectrogram_segments.png"

# Call the function
plot_longest_segments_by_label(file_path, output_file_path)
