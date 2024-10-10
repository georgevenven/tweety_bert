import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

def plot_spectrogram_with_labels(file_path, segment_length):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # Integer labels per timepoint
    ground_truth_labels = data["ground_truth_labels"]
    embedding = data["embedding_outputs"]
    ground_truth_colors = data["ground_truth_colors"].item()  # Load ground truth colors from the NPZ file
    hdbscan_colors = data["hdbscan_colors"].item()  # Load HDBSCAN colors from the NPZ file
    print(embedding.shape)

    # Randomly select the starting point of the segment
    start_idx = random.randint(0, spec.shape[0] - segment_length)
    end_idx = start_idx + segment_length

    spec_slice = spec[start_idx:end_idx, :]  # Take all frequency bins, but only the selected segment
    labels_slice = labels[start_idx:end_idx]  # Take the labels for the selected segment
    ground_truth_labels_slice = ground_truth_labels[start_idx:end_idx]  # Take the ground truth labels for the selected segment
    spec_slice = spec_slice.T
    print(spec_slice.shape)
    print(labels_slice.shape)

    # Set up the figure and gridspec
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[20, 2, 2], hspace=0.05)

    # Create a spectrogram axis
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(spec_slice, aspect='auto', origin='lower')
    ax0.axis('off')

    # Create an axis for the HDBSCAN labels color bar
    ax1 = fig.add_subplot(gs[1])
    hdbscan_label_colors = [hdbscan_colors[label] for label in labels_slice]
    hdbscan_label_colors_rgb = [mcolors.to_rgb(color) for color in hdbscan_label_colors]
    ax1.imshow([hdbscan_label_colors_rgb], aspect='auto', origin='lower')
    ax1.axis('off')

    # Create an axis for the ground truth labels color bar
    ax2 = fig.add_subplot(gs[2])
    ground_truth_label_colors = [ground_truth_colors[label] for label in ground_truth_labels_slice]
    ground_truth_label_colors_rgb = [mcolors.to_rgb(color) for color in ground_truth_label_colors]
    ax2.imshow([ground_truth_label_colors_rgb], aspect='auto', origin='lower')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

# Load the NPZ file and call the function to plot
file_path = "/media/george-vengrovski/disk1/test_specs/9.11-12-12-59--12.7076_segment_0.npz"
segment_length = 1500
plot_spectrogram_with_labels(file_path, segment_length)