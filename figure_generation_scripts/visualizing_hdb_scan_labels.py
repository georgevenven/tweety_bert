import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Now import from src
from data_class import SongDataSet_Image, CollateFunction
from analysis import ComputerClusterPerformance

# Create an instance of ComputerClusterPerformance
# We can pass an empty list since we're only using the method
cluster_performance = ComputerClusterPerformance([])

def plot_spectrogram_with_labels(file_path, segment_length, start_idx, save_path, smoothing=True):
    # Load data from the .npz file
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]  # Spectrogram data
    labels = data["hdbscan_labels"]  # Integer labels per timepoint (-1 for noise)
    ground_truth_labels = data["ground_truth_labels"]
    embedding = data["embedding_outputs"]
    print(embedding.shape)

    # Convert syllable labels to phrase labels using the class method
    if smoothing:
        labels = cluster_performance.majority_vote(labels, window_size=150)
    ground_truth_labels = cluster_performance.syllable_to_phrase_labels(ground_truth_labels, silence=0)

    # === HDBSCAN Labels Coloring ===
    # Filter out noise points first, then increment (to match UMAP plot logic exactly)
    mask = labels != -1
    adjusted_labels = labels.copy()
    adjusted_labels[mask] += 1  # Increment non-noise labels by 1

    # Create the same colormap as in UMAP plots
    plot_labels = adjusted_labels[mask]  # Only non-noise labels
    unique_labels = np.unique(plot_labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    
    # Create color array for all points (including noise)
    hdbscan_label_colors_rgb_full = np.zeros((len(adjusted_labels), 4))
    hdbscan_label_colors_rgb_full[~mask] = [0.0, 0.0, 0.0, 1.0]  # Noise points in black
    hdbscan_label_colors_rgb_full[mask] = [label_to_color[label] for label in adjusted_labels[mask]]

    # === Ground Truth Labels Coloring ===
    # Increment ground truth labels by 1 first (to match UMAP plots)
    ground_truth_labels = ground_truth_labels + 1

    # Process ground truth colors exactly as in UMAP plots
    ground_truth_colors = list(data["ground_truth_colors"])
    ground_truth_colors[1] = [0.0, 0.0, 0.0, 1.0]  # Set black color for index 1

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
    
    # Stack all colors into a single numpy array with shape (n_colors, 4)
    ground_truth_colors = np.stack(ground_truth_colors)
    
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

    # Extract the corresponding colors for the sliced data
    hdbscan_label_colors_rgb_slice = hdbscan_label_colors_rgb_full[start_idx:end_idx]
    ground_truth_label_colors_rgb_slice = ground_truth_label_colors_rgb_full[start_idx:end_idx]

    # Convert to NumPy arrays with shape (1, length, 4)
    # Ensure arrays are properly shaped before adding the new axis
    hdbscan_label_colors_rgb_array = np.array(hdbscan_label_colors_rgb_slice)[np.newaxis, :, :]
    ground_truth_label_colors_rgb_array = np.array(ground_truth_label_colors_rgb_slice)[np.newaxis, :, :]

    # === Plotting ===
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
    
    # Save the figure instead of showing it
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def generate_all_spectrograms(file_path, segment_length, step_size, output_dir, smoothing=True):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    spec = data["s"]
    total_length = spec.shape[0]
    
    # Calculate number of segments
    num_segments = (total_length - segment_length) // step_size + 1
    
    print(f"Generating {num_segments} spectrograms...")
    
    # Generate spectrograms for each segment
    for i in range(num_segments):
        start_idx = i * step_size
        save_path = os.path.join(output_dir, f"spectrogram_{i:05d}.png")
        print(f"Generating spectrogram {i+1}/{num_segments}", end='\r')
        plot_spectrogram_with_labels(file_path, segment_length, start_idx, save_path, smoothing=smoothing)
    
    print("\nDone!")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spectrograms with HDBSCAN and ground truth labels.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the .npz file.")
    parser.add_argument("--segment_length", type=int, default=1000, help="Length of each spectrogram window.")
    parser.add_argument("--step_size", type=int, default=250, help="How much to slide the window by.")
    parser.add_argument("--output_dir", type=str, default="imgs/all_spec_plus_labels", help="Directory to save output images.")
    parser.add_argument("--no_smoothing", action="store_true", help="Turn off label smoothing (majority vote). Smoothing is ON by default.")

    args = parser.parse_args()

    generate_all_spectrograms(
        args.file_path,
        args.segment_length,
        args.step_size,
        args.output_dir,
        smoothing=not args.no_smoothing
    )
