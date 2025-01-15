import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from src.analysis import ComputerClusterPerformance

# Adjust these paths as needed
input_path = "/media/george-vengrovski/George-SSD/folds_for_paper_llb"
output_dir_multiple = "imgs/umap_plots/umap_folds"  # For multiple files

# Ensure Matplotlib does not attempt to show windows
plt.ioff()

def plot_embeddings(embeddings, labels, title, point_size, alpha, is_hdbscan, ground_truth_colors, base_name, output_directory, is_phrase=False):
    """
    Create scatter plots of embeddings colored by different label types.

    Parameters
    ----------
    embeddings : ndarray
        2D array of shape (n_samples, 2) containing embedding coordinates (e.g., from UMAP).
    labels : ndarray
        Integer labels for each point (ground truth or HDBSCAN clusters).
    title : str
        Plot title.
    point_size : int
        Size of the points in the scatter plot.
    alpha : float
        Alpha (transparency) of the points.
    is_hdbscan : bool
        Whether labels are from HDBSCAN (True) or ground truth (False).
    ground_truth_colors : list
        List of colors for ground truth labels.
    base_name : str
        Base name for output file naming.
    output_directory : str
        Directory to save the output plots.
    is_phrase : bool
        Whether the labels are phrase-level (True) or syllable-level (False).
    """
    fig = plt.figure(figsize=(8, 8), dpi=300)
    ax = fig.add_subplot(111)
    
    # Set consistent margins
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    if is_hdbscan:
        # Filter out noise points (-1) and increment labels by 1
        mask = labels != -1
        plot_embeddings = embeddings[mask]
        plot_labels = labels[mask] + 1  # Increment labels by 1 to avoid negative indices

        # Map HDBSCAN labels to colors using a colormap
        unique_labels = np.unique(plot_labels)
        num_labels = len(unique_labels)
        cmap = plt.cm.get_cmap('tab20', num_labels)
        label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in plot_labels]

        plot_type = 'hdbscan'
    else:
        # For ground truth labels, map labels directly to ground_truth_colors
        plot_embeddings = embeddings
        plot_labels = labels.astype(int)

        # Ensure label indices are within the bounds of ground_truth_colors
        max_label_index = len(ground_truth_colors) - 1
        label_indices = [label if label <= max_label_index else label % len(ground_truth_colors) for label in plot_labels]

        colors = [ground_truth_colors[label] for label in label_indices]
        plot_type = 'phrase_labels' if is_phrase else 'ground_truth'

    plt.scatter(
        plot_embeddings[:, 0],
        plot_embeddings[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        edgecolors='none'
    )

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Embedding Dimension 1', fontsize=14)
    plt.ylabel('Embedding Dimension 2', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(output_directory, f'{base_name}_embedding_plot_{plot_type}.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory


def process_file(file_path, output_directory=None):
    """
    Process a single NPZ file, plotting embeddings with different label types.

    Parameters
    ----------
    file_path : str
        Path to the .npz file to be processed.
    output_directory : str, optional
        Directory to save output files. If None, saves in the current directory.
    """
    if output_directory is None:
        output_directory = "."

    # Load data
    f = np.load(file_path, allow_pickle=True)
    ground_truth = f["ground_truth_labels"]
    embeddings = f["embedding_outputs"]
    hdbscan_labels = f["hdbscan_labels"]
    ground_truth_colors = f["ground_truth_colors"]

    # Increment ground truth labels by 1 to match script 1
    ground_truth = ground_truth + 1

    # Convert ground_truth_colors to a list and set the color at index 1 to black
    ground_truth_colors = list(ground_truth_colors)
    ground_truth_colors[1] = [0.0, 0.0, 0.0]  # Set color at index 1 to black (RGB)

    # Extract base name from file path
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create ComputerClusterPerformance instance for label processing
    cluster_performance = ComputerClusterPerformance([file_path])
    
    # Convert ground truth to phrase labels and apply majority vote

    # silence is usually 0, but since we are using 1 to represent silence, we need to set silence to 1

    # phrase_labels = cluster_performance.syllable_to_phrase_labels(ground_truth, silence=1)
    hdbscan_labels = cluster_performance.fill_noise_with_nearest_label(hdbscan_labels)
    phrase_labels = cluster_performance.syllable_to_phrase_labels(ground_truth, silence=1)

    # phrase_labels = cluster_performance.majority_vote(phrase_labels, window_size=0)  # Adjust window size as needed

    # Plot original ground truth labels
    plot_embeddings(
        embeddings=embeddings,
        labels=ground_truth,
        title='Embeddings Colored by Ground Truth Labels',
        point_size=10,
        alpha=0.1,
        is_hdbscan=False,
        ground_truth_colors=ground_truth_colors,
        base_name=base_name,
        output_directory=output_directory
    )

    # Plot phrase-level ground truth labels
    plot_embeddings(
        embeddings=embeddings,
        labels=phrase_labels,
        title='Embeddings Colored by Phrase Labels',
        point_size=10,
        alpha=0.1,
        is_hdbscan=False,
        ground_truth_colors=ground_truth_colors,
        base_name=base_name,
        output_directory=output_directory,
        is_phrase=True
    )

    # Plot HDBSCAN labels
    plot_embeddings(
        embeddings=embeddings,
        labels=hdbscan_labels,
        title='Embeddings Colored by HDBSCAN Labels',
        point_size=10,
        alpha=0.1,
        is_hdbscan=True,
        ground_truth_colors=ground_truth_colors,
        base_name=base_name,
        output_directory=output_directory
    )


# Determine if input_path is a directory or a file
if os.path.isdir(input_path):
    # Process multiple files
    npz_files = glob(os.path.join(input_path, "*.npz"))
    # Output directory for multiple mode
    multi_output_directory = os.path.join("imgs", "umap_plots", "umap_folds")
    for file in npz_files:
        process_file(file, output_directory=multi_output_directory)
else:
    # Process a single file
    process_file(input_path, output_directory=".")
