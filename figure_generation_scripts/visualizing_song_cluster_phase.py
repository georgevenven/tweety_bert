"""
this module provides an interactive umap visualization tool that loads a .npz file containing:
    - embedding_outputs: a 2d array with umap coordinates,
    - s: spectrogram data,
    - ground_truth_labels: original ground truth labels (if available),
    - hdbscan_labels: alternative clustering labels from hdbscan,
    - dataset_indices: indices used for group coloring,
    - file_indices: supplementary indices.

the tool displays a scatter plot of the umap projection and enables interactive selection via a lasso tool.
upon selection, it generates a composite figure that includes:
    - a phase plot (using a color gradient based on point positions),
    - a scatter plot colored by either ground truth/hdbscan labels or, if enabled, by dataset indices,
    - an image of the corresponding spectrogram region,
    - phase gradient and ground truth (or dataset group) color bars.

coloring behavior:
    - if used_group_coloring is true (default: false), the dataset_indices field is used for coloring with a tab20 palette.
    - otherwise, if all ground truth labels are zero, hdbscan_labels are used (with negative labels rendered as grey) 
      and a tab20 palette is applied.
    - if valid ground truth labels are present, they are incremented by one and colored using the ground_truth_colors 
      provided in the file (if available) or a default palette.

usage:
    instantiate the umapselector class with the path to the .npz file, an optional max_length (default 500), and 
    an optional used_group_coloring flag (default false). then call plot_umap_with_selection() to start the interactive session.

note:
    the lasso widget is stored in an instance variable to prevent garbage collection.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import os
import random
import string
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

class UMAPSelector:
    def __init__(self, file_path, max_length=500, used_group_coloring=False):
        data = np.load(file_path, allow_pickle=True)
        self.embedding = data["embedding_outputs"]
        self.spec = data["s"]
        self.labels = data["ground_truth_labels"]
        self.file_indices = data["file_indices"]
        self.used_group_coloring = used_group_coloring

        if self.used_group_coloring:
            self.labels_for_color = data["dataset_indices"]
            cmap = plt.get_cmap("tab20")
            self.ground_truth_colors = [mcolors.to_hex(cmap(i)) for i in range(20)]
            self.using_hdbscan = False
        else:
            if np.all(self.labels == 0):
                print("all ground truth labels are 0; using hdbscan_labels instead")
                self.labels_for_color = data["hdbscan_labels"]
                self.using_hdbscan = True
                cmap = plt.get_cmap("tab20")
                self.ground_truth_colors = [mcolors.to_hex(cmap(i)) for i in range(20)]
            else:
                self.using_hdbscan = False
                self.labels_for_color = self.labels + 1
                if "ground_truth_colors" in data:
                    colors = list(data["ground_truth_colors"])
                    if len(colors) > 1:
                        colors[1] = [0.0, 0.0, 0.0]
                    self.ground_truth_colors = colors
                else:
                    self.ground_truth_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        self.selected_points = None
        self.selected_embedding = None
        self.max_length = max_length
        self.lasso = None  # preserve lasso widget

    def get_color(self, label):
        try:
            label = int(label)
        except:
            label = 0
        if not self.used_group_coloring:
            if self.using_hdbscan and label < 0:
                return "#7f7f7f"
        return self.ground_truth_colors[label % len(self.ground_truth_colors)]

    def find_random_contiguous_region(self, points, labels):
        if len(points) > self.max_length:
            start_index = random.randint(0, len(points) - self.max_length)
            return points[start_index:start_index + self.max_length]
        else:
            return points

    def onselect(self, verts):
        path = Path(verts)
        mask = path.contains_points(self.embedding)
        self.selected_points = np.where(mask)[0]
        if len(self.selected_points) > 0:
            self.selected_embedding = self.embedding[self.selected_points]
            self.plot_selected_region()

    def plot_umap_with_selection(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(self.labels_for_color)
        for label in unique_labels:
            mask = self.labels_for_color == label
            color = self.get_color(label)
            label_type = "dataset" if self.used_group_coloring else ("hdbscan" if self.using_hdbscan else "groundtruth")
            ax.scatter(
                self.embedding[mask, 0],
                self.embedding[mask, 1],
                s=20,
                alpha=0.1,
                color=color,
                label=f'{label_type} label {label}'
            )
        self.lasso = LassoSelector(ax, self.onselect)
        ax.set_title('umap projection', fontsize=14)
        ax.set_xlabel('umap 1', fontsize=12)
        ax.set_ylabel('umap 2', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_selected_region(self):
        if self.selected_points is not None:
            region = self.find_random_contiguous_region(self.selected_points, self.labels)
            if len(region) == 0:
                return
            spec_region = self.spec[region]
            selected_embedding = self.embedding[self.selected_points]
            selected_region_embedding = self.embedding[region]
            base_label = self.labels_for_color[region[0]]
            if self.used_group_coloring:
                label_str = f"dataset_{base_label}"
            else:
                label_str = f"hdbscan_{base_label}" if self.using_hdbscan else f"groundtruth_{base_label}"
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

            x_coords = selected_embedding[:, 0]
            y_coords = selected_embedding[:, 1]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_norm = (x_coords - x_min) / (x_max - x_min)
            y_norm = (y_coords - y_min) / (y_max - y_min)

            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(
                2, 2,
                width_ratios=[2.64, 6.75],
                height_ratios=[1, 1],
                hspace=0.3,
                top=0.95
            )

            ax_points_gradient = fig.add_subplot(gs[0, 0])
            ax_points_groundtruth = fig.add_subplot(gs[1, 0])
            ax_spec = fig.add_subplot(gs[:, 1])

            normalized_coords = np.column_stack((x_norm, y_norm))
            grad_colors = np.zeros((len(normalized_coords), 3))
            grad_colors[:, 0] = normalized_coords[:, 0]
            grad_colors[:, 1] = normalized_coords[:, 1]

            ax_points_gradient.scatter(x_norm, y_norm, s=6, c=grad_colors, alpha=0.6)
            ax_points_gradient.set_aspect('equal')
            ax_points_gradient.set_title('phase', fontsize=24)
            ax_points_gradient.set_xticks([])
            ax_points_gradient.set_yticks([])

            scatter_colors = [self.get_color(label) for label in self.labels_for_color[self.selected_points]]
            ax_points_groundtruth.scatter(x_norm, y_norm, s=6, c=scatter_colors, alpha=0.6)
            ax_points_groundtruth.set_aspect('equal')
            ax_points_groundtruth.set_title('ground truth', fontsize=24)
            ax_points_groundtruth.set_xticks([])
            ax_points_groundtruth.set_yticks([])

            ax_spec.imshow(spec_region[:, :250].T, aspect='auto', origin='lower', cmap='viridis')
            ax_spec.set_xticks([])
            ax_spec.set_yticks([])
            ax_spec.set_xlabel('')
            ax_spec.set_ylabel('')

            divider = make_axes_locatable(ax_spec)
            ax_gradient = divider.append_axes("bottom", size="12.5%", pad=0.525)
            region_x_coords = selected_region_embedding[:, 0]
            region_y_coords = selected_region_embedding[:, 1]
            region_x_norm = (region_x_coords - x_min) / (x_max - x_min)
            region_y_norm = (region_y_coords - y_min) / (y_max - y_min)
            normalized_region_coords = np.column_stack((region_x_norm, region_y_norm))
            region_grad_colors = np.zeros((len(normalized_region_coords), 3))
            region_grad_colors[:, 0] = normalized_region_coords[:, 0]
            region_grad_colors[:, 1] = normalized_region_coords[:, 1]
            ax_gradient.imshow([region_grad_colors], aspect='auto')
            ax_gradient.set_axis_off()
            ax_gradient.set_title('phase gradient', fontsize=24, y=-0.65)

            ax_groundtruth = divider.append_axes("bottom", size="12.5%", pad=0.84)
            gt_segment_colors = [self.get_color(label) for label in self.labels_for_color[region]]
            gt_segment_colors_rgb = [mcolors.to_rgb(color) for color in gt_segment_colors]
            ax_groundtruth.imshow([gt_segment_colors_rgb], aspect='auto')
            ax_groundtruth.set_axis_off()
            ax_groundtruth.set_title('ground truth class', fontsize=24, y=-0.65)

            plt.tight_layout()
            os.makedirs("imgs/selected_regions", exist_ok=True)
            fig.savefig(f"imgs/selected_regions/{random_name}_{label_str}.png")
            plt.close(fig)

# usage example
file_path = "/media/george-vengrovski/flash-drive/DOI_data_/USA5506_PrePostDOI.npz"
selector = UMAPSelector(file_path, max_length=1000, used_group_coloring=False)
selector.plot_umap_with_selection()
