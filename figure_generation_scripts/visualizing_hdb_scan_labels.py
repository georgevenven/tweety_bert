import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.analysis import syllable_to_phrase_labels

def plot_spectrogram_with_labels(file_path, segment_length):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # Integer labels per timepoint (-1 for noise)
    ground_truth_labels = data["ground_truth_labels"]
    embedding = data["embedding_outputs"]
    print(embedding.shape)

    # Convert syllable labels to phrase labels
    ground_truth_labels = syllable_to_phrase_labels(ground_truth_labels, silence=0)

    # Increment ground truth labels by 1 to match Script 1 adjustments
    ground_truth_labels = ground_truth_labels + 1

    # === HDBSCAN Labels Coloring ===
    # Filter out noise points and increment labels by 1 (matching UMAP plot logic)
    mask = labels != -1
    adjusted_labels = labels.copy()
    adjusted_labels[mask] += 1  # Increment to avoid negative indices

    # Create the same colormap as in UMAP plots
    unique_labels = np.unique(adjusted_labels[adjusted_labels != -1])
    num_labels = len(unique_labels)
    cmap = plt.colormaps['tab20']
    label_to_color = {label: np.array(cmap(i)) for i, label in enumerate(unique_labels)}
    
    # Assign black color to noise points (-1)
    label_to_color[-1] = np.array([0.0, 0.0, 0.0, 1.0])

    # Map labels to colors and ensure numpy array format
    hdbscan_label_colors_rgb_full = np.array([label_to_color[label] for label in adjusted_labels])

    # === Ground Truth Labels Coloring ===
    # Load ground truth colors and ensure proper shape
    ground_truth_colors = data["ground_truth_colors"]
    
    # Print debug information
    print("Original ground truth colors shapes:")
    for i, color in enumerate(ground_truth_colors):
        print(f"Color {i}: {np.array(color).shape}, {color}")
    
    # Convert colors to proper numpy arrays and ensure each has shape (4,)
    ground_truth_colors = []
    for color in data["ground_truth_colors"]:
        # Handle different possible color formats
        color_array = np.array(color, dtype=np.float32)
        if color_array.shape == ():  # scalar
            color_array = np.array([color_array, color_array, color_array, 1.0], dtype=np.float32)
        elif len(color_array) == 3:  # RGB
            color_array = np.append(color_array, 1.0).astype(np.float32)
        elif len(color_array) == 4:  # RGBA
            color_array = color_array.astype(np.float32)
        else:
            raise ValueError(f"Unexpected color format: {color_array}")
        ground_truth_colors.append(color_array)
    
    # Set black color for index 1
    ground_truth_colors[1] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    # Print debug information after conversion
    print("\nProcessed ground truth colors shapes:")
    for i, color in enumerate(ground_truth_colors):
        print(f"Color {i}: {color.shape}, {color}")
    
    # Stack all colors into a single numpy array with shape (n_colors, 4)
    ground_truth_colors = np.stack(ground_truth_colors)
    print("\nFinal ground truth colors shape:", ground_truth_colors.shape)
    
    # Map ground truth labels to colors
    max_label_index = len(ground_truth_colors) - 1
    label_indices = np.clip(ground_truth_labels, 0, max_label_index)
    ground_truth_label_colors_rgb_full = ground_truth_colors[label_indices]

    # Ensure both color arrays have consistent shape (n_timepoints, 4)
    assert ground_truth_label_colors_rgb_full.shape[-1] == 4
    assert hdbscan_label_colors_rgb_full.shape[-1] == 4

    # Randomly select the starting point of the segment
    start_idx = random.randint(0, spec.shape[0] - segment_length)
    end_idx = start_idx + segment_length

    spec_slice = spec[start_idx:end_idx, :]  # Select the segment
    adjusted_labels_slice = adjusted_labels[start_idx:end_idx]  # Adjusted HDBSCAN labels for the segment
    ground_truth_labels_slice = ground_truth_labels[start_idx:end_idx]  # Ground truth labels for the segment
    spec_slice = spec_slice.T  # Transpose for correct orientation
    print(spec_slice.shape)
    print(adjusted_labels_slice.shape)

    # Extract the corresponding colors for the sliced data
    hdbscan_label_colors_rgb_slice = hdbscan_label_colors_rgb_full[start_idx:end_idx]
    ground_truth_label_colors_rgb_slice = ground_truth_label_colors_rgb_full[start_idx:end_idx]

    # Convert to NumPy arrays with shape (1, length, 4)
    # Ensure arrays are properly shaped before adding the new axis
    hdbscan_label_colors_rgb_array = np.array(hdbscan_label_colors_rgb_slice)[np.newaxis, :, :]
    ground_truth_label_colors_rgb_array = np.array(ground_truth_label_colors_rgb_slice)[np.newaxis, :, :]

    # === Plotting ===
    # Set up the figure and gridspec
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[20, 2, 2], hspace=0.05)

    # Plot the spectrogram
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(spec_slice, aspect='auto', origin='lower')
    ax0.axis('off')

    # Plot the HDBSCAN labels color bar
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(hdbscan_label_colors_rgb_array, aspect='auto', origin='lower')
    ax1.axis('off')

    # Plot the ground truth labels color bar
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(ground_truth_label_colors_rgb_array, aspect='auto', origin='lower')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

# Load the NPZ file and call the function to plot
file_path = "files/llb3.npz"
segment_length = 1000
plot_spectrogram_with_labels(file_path, segment_length)
