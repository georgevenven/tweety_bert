import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Adjust these paths as needed
input_path = "/media/george-vengrovski/George-SSD/folds_for_paper_llb"
output_dir_multiple = "imgs/umap_plots/umap_folds"  # For multiple files

# Ensure Matplotlib does not attempt to show windows
plt.ioff()

def plot_embeddings(embeddings, labels, title, point_size, alpha, is_hdbscan, ground_truth_colors, base_name, output_directory):
    """
    Create scatter plots of embeddings colored either by ground truth or HDBSCAN labels.

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
    """
    plt.figure(figsize=(8, 8), dpi=300)
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
        plot_type = 'ground_truth'

    plt.scatter(
        plot_embeddings[:, 0],
        plot_embeddings[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        edgecolors='none'
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Embedding Dimension 1', fontsize=14)
    plt.ylabel('Embedding Dimension 2', fontsize=14)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    output_file = os.path.join(output_directory, f'{base_name}_embedding_plot_{plot_type}.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory


def process_file(file_path, output_directory=None):
    """
    Process a single NPZ file, plotting embeddings colored by both ground truth and HDBSCAN labels.

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

    # Plot embeddings colored by ground truth labels
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

    # Plot embeddings colored by HDBSCAN labels
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
