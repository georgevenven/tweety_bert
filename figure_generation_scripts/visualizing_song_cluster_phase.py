import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import os
import random
import string
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import time

class UMAPSelector:
    def __init__(self, file_path, max_length=500):
        data = np.load(file_path, allow_pickle=True)
        self.embedding = data["embedding_outputs"]
        self.spec = data["s"]
        self.labels = data["ground_truth_labels"]
        self.ground_truth_labels = data["ground_truth_labels"]
        self.ground_truth_colors = data["ground_truth_colors"].item()
        self.selected_points = None
        self.selected_embedding = None
        self.max_length = max_length

    def find_random_contiguous_region(self, points, labels):
        if len(points) > self.max_length:
            start_index = random.randint(0, len(points) - self.max_length)
            end_index = start_index + self.max_length
            random_region = points[start_index:end_index]
            return random_region
        else:
            random_region = points
            return random_region


    def onselect(self, verts):
        path = Path(verts)
        mask = path.contains_points(self.embedding)
        self.selected_points = np.where(mask)[0]
        if len(self.selected_points) > 0:
            self.selected_embedding = self.embedding[self.selected_points]
            self.plot_selected_region()

    def plot_umap_with_selection(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(self.ground_truth_labels)
        for label in unique_labels:
            mask = self.ground_truth_labels == label
            color = self.ground_truth_colors[label]
            ax.scatter(self.embedding[mask, 0], self.embedding[mask, 1], s=20, alpha=0.1, color=color, label=f'Ground Truth Label {label}')
        lasso = LassoSelector(ax, self.onselect)
        ax.set_title('UMAP Projection', fontsize=14)
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_selected_region(self):
        if self.selected_points is not None:
            region = self.find_random_contiguous_region(self.selected_points, self.labels)
            if len(region) == 0:
                return
            spec_region = self.spec[region]
            selected_embedding = self.embedding[self.selected_points]  # Use all selected points
            selected_region_embedding = self.embedding[region]  # Embedding points corresponding to the spectrogram
            ground_truth_label = self.ground_truth_labels[region[0]]
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

            x_coords = selected_embedding[:, 0]  # Use all selected points
            y_coords = selected_embedding[:, 1]  # Use all selected points
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_norm = (x_coords - x_min) / (x_max - x_min)
            y_norm = (y_coords - y_min) / (y_max - y_min)

            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 6.75], height_ratios=[1, 1])  # Slightly increased width ratio for UMAP plots for a bit more space, adjusted by 5%

            ax_points_gradient = fig.add_subplot(gs[0, 0])
            ax_points_groundtruth = fig.add_subplot(gs[1, 0])
            ax_spec = fig.add_subplot(gs[:, 1])

            # Calculate the color gradient based on the normalized coordinates of all selected points
            normalized_coords = np.column_stack((x_norm, y_norm))
            colors = np.zeros((len(normalized_coords), 3))
            colors[:, 0] = normalized_coords[:, 0]  # Purple hue based on x-coordinate
            colors[:, 1] = normalized_coords[:, 1]  # Yellow hue based on y-coordinate

            ax_points_gradient.scatter(x_norm, y_norm, s=5, c=colors, alpha=0.5)
            ax_points_gradient.set_aspect('equal')
            ax_points_gradient.set_title('Phase', fontsize=18)
            ax_points_gradient.set_xlabel('Normalized X', fontsize=14)
            ax_points_gradient.set_ylabel('Normalized Y', fontsize=14)
            ax_points_gradient.tick_params(axis='both', which='major', labelsize=12)

            ground_truth_colors = [self.ground_truth_colors[label] for label in self.ground_truth_labels[self.selected_points]]
            ax_points_groundtruth.scatter(x_norm, y_norm, s=5, c=ground_truth_colors, alpha=0.5)
            ax_points_groundtruth.set_aspect('equal')
            ax_points_groundtruth.set_title('Ground Truth', fontsize=18)
            ax_points_groundtruth.set_xlabel('Normalized X', fontsize=14)
            ax_points_groundtruth.set_ylabel('Normalized Y', fontsize=14)
            ax_points_groundtruth.tick_params(axis='both', which='major', labelsize=12)

            ax_spec.imshow(spec_region[:, :250].T, aspect='auto', origin='lower', cmap='viridis')
            ax_spec.set_xticks([])  # Remove x-axis ticks
            ax_spec.set_yticks([])  # Remove y-axis ticks
            ax_spec.set_xlabel('')  # Remove x-axis label
            ax_spec.set_ylabel('')  # Remove y-axis label

            divider = make_axes_locatable(ax_spec)
            ax_gradient = divider.append_axes("bottom", size="12.5%", pad=0.525)  # Slightly increased padding for a bit more space, adjusted by 5%

            # Calculate the color gradient based on the normalized coordinates of the points corresponding to the spectrogram
            region_x_coords = selected_region_embedding[:, 0]
            region_y_coords = selected_region_embedding[:, 1]
            region_x_norm = (region_x_coords - x_min) / (x_max - x_min)
            region_y_norm = (region_y_coords - y_min) / (y_max - y_min)
            normalized_region_coords = np.column_stack((region_x_norm, region_y_norm))
            region_colors = np.zeros((len(normalized_region_coords), 3))
            region_colors[:, 0] = normalized_region_coords[:, 0]  # Purple hue based on x-coordinate
            region_colors[:, 1] = normalized_region_coords[:, 1]  # Yellow hue based on y-coordinate

            ax_gradient.imshow([region_colors], aspect='auto')
            ax_gradient.set_axis_off()
            ax_gradient.set_title('Phase Gradient', fontsize=24, y=-0.65)  # Adjusted title position for more space

            ax_groundtruth = divider.append_axes("bottom", size="12.5%", pad=0.84)  # Slightly increased padding for a bit more space, adjusted by 5%
            gt_segment_colors = [self.ground_truth_colors[label] for label in self.ground_truth_labels[region]]
            gt_segment_colors_rgb = [mcolors.to_rgb(color) for color in gt_segment_colors]
            ax_groundtruth.imshow([gt_segment_colors_rgb], aspect='auto')
            ax_groundtruth.set_axis_off()
            ax_groundtruth.set_title('Ground Truth Class', fontsize=24, y=-0.65)  # Adjusted title position for more space

            plt.tight_layout()

            os.makedirs("imgs/selected_regions", exist_ok=True)
            fig.savefig(f"imgs/selected_regions/{random_name}_groundtruth_{ground_truth_label}.png")
            plt.close(fig)

file_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_temp.npz"
selector = UMAPSelector(file_path, max_length=500)
selector.plot_umap_with_selection()