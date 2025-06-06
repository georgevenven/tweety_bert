import os
import sys
import argparse

# Get the directory of the current script and the project root
figure_generation_dir = os.path.dirname(__file__)
project_root = os.path.dirname(figure_generation_dir)

# Change the working directory to the project root
os.chdir(project_root)

# Add src directory to Python path
sys.path.append("src")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

# Make sure Matplotlib is using an interactive backend:
# (If you're in a pure headless environment, you won't see the figure.)
matplotlib.use("TkAgg")  # or Qt5Agg, etc., depending on your environment

from matplotlib.widgets import RectangleSelector
from analysis import ComputerClusterPerformance
from data_class import SongDataSet_Image, CollateFunction

# ------------------ CONFIGURABLE PATHS ------------------ #
# Either a .npz file (interactive cropping) or folder
# input_path = "files/LLB3_Untrained.npz"
# output_dir = "imgs/umap_plots"
# -------------------------------------------------------- #

def interactive_crop(embeddings, colors):
    """
    Display an interactive figure so the user can 'drag a rectangle'
    to define a crop region. Returns (x_min, x_max, y_min, y_max).
    """

    # We'll store the rectangle coordinates here
    crop_coords = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1], c=colors, s=10, alpha=0.1
    )
    ax.set_title("Drag to select crop region. Close window when done.")

    # RectangleSelector callback: on release, save the bounding box
    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        crop_coords["x_min"], crop_coords["x_max"] = sorted([x1, x2])
        crop_coords["y_min"], crop_coords["y_max"] = sorted([y1, y2])

    # Create the rectangle selector with updated parameters
    rect_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],  # Only respond to left mouse button
        minspanx=0.0,
        minspany=0.0,
        spancoords="data",
        interactive=True
    )

    plt.show()  # Blocks until window is closed

    # If the user never dragged a box, fall back to full extent
    if crop_coords["x_min"] is None or crop_coords["y_min"] is None:
        x_min, x_max = np.min(embeddings[:, 0]), np.max(embeddings[:, 0])
        y_min, y_max = np.min(embeddings[:, 1]), np.max(embeddings[:, 1])
    else:
        x_min, x_max = crop_coords["x_min"], crop_coords["x_max"]
        y_min, y_max = crop_coords["y_min"], crop_coords["y_max"]

    return x_min, x_max, y_min, y_max


def plot_embeddings(
    embeddings,
    labels,
    title,
    is_hdbscan,
    ground_truth_colors,
    base_name,
    output_directory,
    bounding_box=None,
):
    """
    Plots both 'uncropped' and 'cropped' (if bounding_box is provided).
    bounding_box is (x_min, x_max, y_min, y_max).
    """

    # ------ CHOOSE COLORS FOR EACH POINT ------
    if is_hdbscan:
        # exclude noise
        mask = labels != -1
        plot_embeddings = embeddings[mask]
        plot_labels = labels[mask] + 1  # shift no negative indices
        unique = np.unique(plot_labels)
        cmap = plt.cm.get_cmap("tab20", len(unique))
        label_to_color = {u: cmap(i) for i, u in enumerate(unique)}
        colors = [label_to_color[lbl] for lbl in plot_labels]
        suffix = "hdbscan"
    else:
        plot_embeddings = embeddings
        plot_labels = labels.astype(int)
        max_label_idx = len(ground_truth_colors) - 1
        label_indices = [
            lbl if lbl <= max_label_idx else lbl % len(ground_truth_colors)
            for lbl in plot_labels
        ]
        colors = [ground_truth_colors[idx] for idx in label_indices]
        suffix = "ground_truth"  # or "phrase_labels" outside

    # ------ UNCROPPED PLOT ------
    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(
        plot_embeddings[:, 0],
        plot_embeddings[:, 1],
        c=colors,
        s=10,
        alpha=0.1,
        edgecolors="none",
    )
    plt.title(title, fontsize=32)
    plt.xlabel("UMAP 1", fontsize=24)
    plt.ylabel("UMAP 2", fontsize=24)
    plt.xticks([])  # Remove tick marks but keep axis
    plt.yticks([])  # Remove tick marks but keep axis

    uncropped_file = os.path.join(
        output_directory, f"{base_name}_embedding_plot_{suffix}.png"
    )
    plt.savefig(uncropped_file, bbox_inches="tight", dpi=300)
    plt.close()

    # ------ CROPPED PLOT (if bounding_box given) ------
    if bounding_box:
        x_min, x_max, y_min, y_max = bounding_box
        plt.figure(figsize=(8, 8), dpi=300)
        plt.scatter(
            plot_embeddings[:, 0],
            plot_embeddings[:, 1],
            c=colors,
            s=10,
            alpha=0.1,
            edgecolors="none",
        )
        plt.title(title + " (Cropped)", fontsize=32)
        plt.xlabel("UMAP 1", fontsize=24)
        plt.ylabel("UMAP 2", fontsize=24)
        plt.xticks([])  # Remove tick marks but keep axis
        plt.yticks([])  # Remove tick marks but keep axis
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        cropped_file = os.path.join(
            output_directory, f"{base_name}_embedding_plot_{suffix}_cropped.png"
        )
        plt.savefig(cropped_file, bbox_inches="tight", dpi=300)
        plt.close()


def process_file(file_path, output_directory=None, bounding_box=None):
    """
    Load data, produce uncropped plots, and produce cropped plots if bounding_box is provided.
    """

    if output_directory is None:
        output_directory = "."

    data = np.load(file_path, allow_pickle=True)
    embeddings = data["embedding_outputs"]
    ground_truth = data["ground_truth_labels"] + 1  # shift by 1
    hdbscan_labels = data["hdbscan_labels"]
    ground_truth_colors = list(data["ground_truth_colors"])
    # set color[1] = black
    ground_truth_colors[1] = [0.0, 0.0, 0.0]

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Create cluster performance object
    cluster_perf = ComputerClusterPerformance([file_path])
    hdbscan_labels = cluster_perf.fill_noise_with_nearest_label(hdbscan_labels)
    phrase_labels = cluster_perf.syllable_to_phrase_labels(ground_truth, silence=1)

    # ----- PLOT GROUND TRUTH -----
    plot_embeddings(
        embeddings=embeddings,
        labels=ground_truth,
        title="Embeddings (Ground Truth)",
        is_hdbscan=False,
        ground_truth_colors=ground_truth_colors,
        base_name=base_name,
        output_directory=output_directory,
        bounding_box=bounding_box,
    )

    # ----- PLOT PHRASE-LEVEL -----
    plot_embeddings(
        embeddings=embeddings,
        labels=phrase_labels,
        title="Embeddings (Phrase)",
        is_hdbscan=False,
        ground_truth_colors=ground_truth_colors,
        base_name=base_name + "_phrase",
        output_directory=output_directory,
        bounding_box=bounding_box,
    )

    # ----- PLOT HDBSCAN LABELS -----
    plot_embeddings(
        embeddings=embeddings,
        labels=hdbscan_labels,
        title="Embeddings (HDBSCAN)",
        is_hdbscan=True,
        ground_truth_colors=ground_truth_colors,
        base_name=base_name + "_hdbscan",
        output_directory=output_directory,
        bounding_box=bounding_box,
    )


if __name__ == "__main__":
    # --- Add argparse setup ---
    parser = argparse.ArgumentParser(description="Generate UMAP plots from NPZ files.")
    parser.add_argument("input_path", type=str, help="Path to the input NPZ file or folder.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output plots.")
    args = parser.parse_args()
    # --- End argparse setup ---

    if os.path.isdir(args.input_path):
        npz_files = glob(os.path.join(args.input_path, "*.npz"))
        out_dir = os.path.join(args.output_dir, "umap_folds")
        for fpath in npz_files:
            process_file(fpath, output_directory=out_dir, bounding_box=None)
    else:
        if not args.input_path.endswith(".npz"):
            print(f"Error: Must provide a .npz file or a folder, got: {args.input_path}")
            sys.exit(1)

        npz_data = np.load(args.input_path, allow_pickle=True)
        embeddings = npz_data["embedding_outputs"]
        ground_truth = npz_data["ground_truth_labels"] + 1
        ground_truth_colors = list(npz_data["ground_truth_colors"])
        ground_truth_colors[1] = [0, 0, 0]
        label_indices = ground_truth.astype(int)
        label_indices = [
            lbl if lbl < len(ground_truth_colors) else lbl % len(ground_truth_colors)
            for lbl in label_indices
        ]
        preview_colors = [ground_truth_colors[lbl] for lbl in label_indices]

        print("Opening interactive window for bounding box selection...")
        x_min, x_max, y_min, y_max = interactive_crop(embeddings, preview_colors)
        bounding_box = (x_min, x_max, y_min, y_max)
        print(f"Selected bounding box = {bounding_box}")

        file_name = os.path.splitext(os.path.basename(args.input_path))[0]
        out_dir = os.path.join(args.output_dir, file_name)
        process_file(
            args.input_path,
            output_directory=out_dir,
            bounding_box=bounding_box,
        )
