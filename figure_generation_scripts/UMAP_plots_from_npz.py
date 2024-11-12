import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load data
file_path = "/media/george-vengrovski/Diana-SSD/GEORGE/temp2rawumap/llb16_for_paper_raw_spec.npz"
f = np.load(file_path, allow_pickle=True)
ground_truth = f["ground_truth_labels"]
embeddings = f["embedding_outputs"]
hdbscan_labels = f["hdbscan_labels"]
ground_truth_colors = f["ground_truth_colors"]

# Increment ground truth labels by 1 to match Script 1
ground_truth = ground_truth + 1

# Convert ground_truth_colors to a list and set the color at index 1 to black
ground_truth_colors = list(ground_truth_colors)
ground_truth_colors[1] = [0.0, 0.0, 0.0]  # Set color at index 1 to black (RGB)

# Extract base name from file path
base_name = os.path.splitext(os.path.basename(file_path))[0]

# Function to create scatter plots
def plot_embeddings(embeddings, labels, title, point_size=1, alpha=0.5, is_hdbscan=False):
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

    else:
        # For ground truth labels, map labels directly to ground_truth_colors
        plot_embeddings = embeddings
        plot_labels = labels.astype(int)

        # Ensure label indices are within the bounds of ground_truth_colors
        max_label_index = len(ground_truth_colors) - 1
        label_indices = [label if label <= max_label_index else label % len(ground_truth_colors) for label in plot_labels]

        colors = [ground_truth_colors[label] for label in label_indices]

    scatter = plt.scatter(
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
    plot_type = 'hdbscan' if is_hdbscan else 'ground_truth'
    plt.savefig(f'{base_name}_embedding_plot_{plot_type}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

# Plot embeddings colored by ground truth labels
plot_embeddings(
    embeddings,
    ground_truth,
    title='Embeddings Colored by Ground Truth Labels',
    point_size=10,
    alpha=0.1,
    is_hdbscan=False
)

# Plot embeddings colored by HDBSCAN labels
plot_embeddings(
    embeddings,
    hdbscan_labels,
    title='Embeddings Colored by HDBSCAN Labels',
    point_size=10,
    alpha=0.1,
    is_hdbscan=True
)
