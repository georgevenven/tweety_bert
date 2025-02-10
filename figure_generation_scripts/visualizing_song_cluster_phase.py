"""
Unified UMAP visualization with both:
 - Original single-figure approach (if collage_mode=False),
 - New "collage" approach (if collage_mode=True),
 - Proper HDBSCAN negative => gray,
 - Original logic for ground truth or hdbscan labeling,
 - Purple/green heatmaps for used_group_coloring=True
   both in the main UMAP and in the single-figure "dataset" subplot.
 
Now, in collage mode, we also pad the phase gradient and label rows to self.max_length,
so that they align with the black-padded spectrogram length.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import os
import random
import string
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

class UMAPSelector:
    def __init__(
        self, 
        file_path, 
        max_length=500, 
        used_group_coloring=False,
        collage_mode=False
    ):
        data = np.load(file_path, allow_pickle=True)
        self.embedding = data["embedding_outputs"]
        self.spec = data["s"]
        self.labels = data["ground_truth_labels"]
        self.file_indices = data["file_indices"]  # needed for collage subplots
        self.used_group_coloring = used_group_coloring
        self.collage_mode = collage_mode
        self.max_length = max_length
        self.lasso = None  # preserve Lasso widget

        # Decide label field & base palette
        if self.used_group_coloring:
            # We'll do purple/green heatmaps for 0,1 => purple, 2,3 => green
            self.labels_for_color = data["dataset_indices"]
            self.using_hdbscan = False
            # We'll override coloring in get_color()
            self.ground_truth_colors = ["purple", "green"]  
        else:
            # Original logic if not group coloring
            if np.all(self.labels == 0):
                print("all ground truth labels are 0; using hdbscan_labels instead")
                self.labels_for_color = data["hdbscan_labels"]
                self.using_hdbscan = True

                # fallback palette from tab20
                tab20_cmap = plt.get_cmap("tab20")
                self.ground_truth_colors = [
                    mcolors.to_hex(tab20_cmap(i)) for i in range(20)
                ]
            else:
                self.using_hdbscan = False
                self.labels_for_color = self.labels + 1

                # if 'ground_truth_colors' is present, adapt them
                if "ground_truth_colors" in data:
                    colors = list(data["ground_truth_colors"])
                    if len(colors) > 1:
                        # ensure second color is black
                        colors[1] = [0.0, 0.0, 0.0]
                    self.ground_truth_colors = colors
                else:
                    self.ground_truth_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        # Will be filled by Lasso
        self.selected_points = None
        self.selected_embedding = None

    def get_color(self, label):
        """
        If used_group_coloring=True => 
            label in [0,1] => "purple", label in [2,3] => "green", else => "gray".
        Else if using hdbscan and label < 0 => "#7f7f7f" (gray).
        Else => index into self.ground_truth_colors.
        """
        try:
            label = int(label)
        except:
            label = 0

        if self.used_group_coloring:
            # Purple/green for 0,1 / 2,3, else gray
            if label in [0, 1]:
                return "purple"
            elif label in [2, 3]:
                return "green"
            else:
                return "gray"
        else:
            # If using HDBSCAN => negative => gray
            if self.using_hdbscan and label < 0:
                return "#7f7f7f"
            # else index ground_truth_colors
            return self.ground_truth_colors[label % len(self.ground_truth_colors)]

    def onselect(self, verts):
        """
        Handler for the Lasso selection.
        Calls single-figure or collage as needed.
        """
        path = Path(verts)
        mask = path.contains_points(self.embedding)
        self.selected_points = np.where(mask)[0]
        if len(self.selected_points) == 0:
            return
        self.selected_embedding = self.embedding[self.selected_points]

        if self.collage_mode:
            self.plot_selected_region_collage()
        else:
            self.plot_selected_region()

    def plot_umap_with_selection(self):
        """
        Main UMAP display: 
         - used_group_coloring=True => purple/green heatmap,
         - else => scatter + hdbscan or ground_truth coloring.
        """
        if self.used_group_coloring:
            self.plot_umap_heatmap_with_selection()
        else:
            self.plot_umap_scatter_with_selection()

    def plot_umap_scatter_with_selection(self):
        """
        Original scatter approach if used_group_coloring=False.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = np.unique(self.labels_for_color)

        for label in unique_labels:
            mask = self.labels_for_color == label
            color = self.get_color(label)

            label_type = "hdbscan" if self.using_hdbscan else "groundtruth"
            ax.scatter(
                self.embedding[mask, 0],
                self.embedding[mask, 1],
                s=20,
                alpha=0.1,
                color=color,
                label=f"{label_type} label {label}"
            )

        self.lasso = LassoSelector(ax, self.onselect)
        ax.set_title("UMAP Projection", fontsize=14)
        ax.set_xlabel("umap 1", fontsize=12)
        ax.set_ylabel("umap 2", fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_umap_heatmap_with_selection(self):
        """
        If used_group_coloring=True => Purple/Green 2D histogram overlay
        with brightness factor and invisible scatter for Lasso.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        nbins = 300
        brightness_factor = 2.0

        x_min, x_max = self.embedding[:,0].min(), self.embedding[:,0].max()
        y_min, y_max = self.embedding[:,1].min(), self.embedding[:,1].max()
        xedges = np.linspace(x_min, x_max, nbins+1)
        yedges = np.linspace(y_min, y_max, nbins+1)

        # label 0,1 => "before" => purple; 2,3 => green
        before_mask = np.isin(self.labels_for_color, [0,1])
        after_mask  = np.isin(self.labels_for_color, [2,3])

        hist_before, _, _ = np.histogram2d(
            self.embedding[before_mask, 0] if np.any(before_mask) else [],
            self.embedding[before_mask, 1] if np.any(before_mask) else [],
            bins=[xedges, yedges]
        )
        hist_after, _, _ = np.histogram2d(
            self.embedding[after_mask, 0] if np.any(after_mask) else [],
            self.embedding[after_mask, 1] if np.any(after_mask) else [],
            bins=[xedges, yedges]
        )

        if hist_before.max() > 0:
            hist_before /= hist_before.max()
        if hist_after.max() > 0:
            hist_after /= hist_after.max()

        # Build an RGB image: purple => R+B, green => G
        rgb = np.zeros((nbins, nbins, 3))
        rgb[..., 0] = hist_before
        rgb[..., 2] = hist_before
        rgb[..., 1] = hist_after

        rgb *= brightness_factor
        np.clip(rgb, 0, 1, out=rgb)

        ax.imshow(
            rgb.transpose((1,0,2)),
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower"
        )

        # invisible scatter for Lasso
        ax.scatter(self.embedding[:,0], self.embedding[:,1], s=0.1, alpha=0.0)
        self.lasso = LassoSelector(ax, self.onselect)

        ax.set_title("UMAP Heatmap (Purple=0/1, Green=2/3)", fontsize=14)
        ax.set_xlabel("umap 1", fontsize=12)
        ax.set_ylabel("umap 2", fontsize=12)
        plt.tight_layout()
        plt.show()

    def find_random_contiguous_region(self, points):
        """
        For the single-figure approach, pick up to 'max_length' contiguous points.
        """
        if len(points) > self.max_length:
            start_index = random.randint(0, len(points) - self.max_length)
            return points[start_index : start_index + self.max_length]
        else:
            return points

    def plot_selected_region(self):
        """
        Original single-figure approach:
          - If used_group_coloring=True => "dataset" subplot as a purple/green heatmap 
          - If used_group_coloring=False => normal scatter (ground truth or hdbscan).
        """
        if self.selected_points is None or len(self.selected_points) == 0:
            return

        region = self.find_random_contiguous_region(self.selected_points)
        if len(region) == 0:
            return

        spec_region = self.spec[region]
        selected_embedding = self.embedding[self.selected_points]
        selected_region_embedding = self.embedding[region]

        # Label for saved figure
        base_label = self.labels_for_color[region[0]]
        if self.used_group_coloring:
            label_str = f"dataset_{base_label}"
        else:
            label_str = (
                f"hdbscan_{base_label}" if self.using_hdbscan 
                else f"groundtruth_{base_label}"
            )
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        # Phase/dataset subplots: normalize the entire selected region
        x_coords = selected_embedding[:, 0]
        y_coords = selected_embedding[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        if x_min==x_max:
            x_norm = np.zeros_like(x_coords)
        else:
            x_norm = (x_coords - x_min)/(x_max - x_min)
        if y_min==y_max:
            y_norm = np.zeros_like(y_coords)
        else:
            y_norm = (y_coords - y_min)/(y_max - y_min)

        fig = plt.figure(figsize=(14,8))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[2.64, 6.75],
            height_ratios=[1, 1],
            hspace=0.3,
            top=0.95
        )

        ax_phase = fig.add_subplot(gs[0,0])
        ax_dataset = fig.add_subplot(gs[1,0])
        ax_spec = fig.add_subplot(gs[:,1])

        # 1) Phase subplot
        phase_colors = np.zeros((len(x_norm),3))
        phase_colors[:,0] = x_norm
        phase_colors[:,1] = y_norm
        ax_phase.scatter(x_norm, y_norm, s=6, c=phase_colors, alpha=0.6)
        ax_phase.set_aspect('equal')
        ax_phase.set_title("phase", fontsize=24)
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])

        # 2) If used_group_coloring => purple/green heatmap; else scatter
        if not self.used_group_coloring:
            scatter_colors = [self.get_color(lbl) for lbl in self.labels_for_color[self.selected_points]]
            ax_dataset.scatter(x_norm, y_norm, s=6, c=scatter_colors, alpha=0.6)
            ax_dataset.set_title(
                "ground truth" if not self.using_hdbscan else "hdbscan",
                fontsize=24
            )
        else:
            brightness_factor = 2.0
            nbins = 300
            xedges = np.linspace(0,1,nbins+1)
            yedges = np.linspace(0,1,nbins+1)

            sub_labels = self.labels_for_color[self.selected_points]
            before_mask = np.isin(sub_labels, [0,1])
            after_mask  = np.isin(sub_labels, [2,3])

            hist_before, _, _ = np.histogram2d(
                x_norm[before_mask], y_norm[before_mask], 
                bins=[xedges, yedges]
            )
            hist_after, _, _ = np.histogram2d(
                x_norm[after_mask], y_norm[after_mask], 
                bins=[xedges, yedges]
            )
            if hist_before.max()>0:
                hist_before /= hist_before.max()
            if hist_after.max()>0:
                hist_after /= hist_after.max()

            rgb = np.zeros((nbins, nbins, 3))
            rgb[...,0] = hist_before  # R
            rgb[...,2] = hist_before  # => purple
            rgb[...,1] = hist_after   # => green

            rgb *= brightness_factor
            np.clip(rgb, 0, 1, out=rgb)

            ax_dataset.imshow(
                rgb.transpose((1,0,2)),
                extent=[0,1,0,1],
                origin="lower"
            )
            ax_dataset.set_title("dataset", fontsize=24)

        ax_dataset.set_aspect('equal')
        ax_dataset.set_xticks([])
        ax_dataset.set_yticks([])

        # 3) Spectrogram
        ax_spec.imshow(spec_region[:, :250].T, aspect='auto', origin='lower', cmap='viridis')
        ax_spec.set_xticks([])
        ax_spec.set_yticks([])

        # color bars below spectrogram
        divider = make_axes_locatable(ax_spec)
        ax_gradient = divider.append_axes("bottom", size="12.5%", pad=0.525)

        # region-based color bars
        rx_coords = selected_region_embedding[:,0]
        ry_coords = selected_region_embedding[:,1]
        rx_min, rx_max = rx_coords.min(), rx_coords.max()
        ry_min, ry_max = ry_coords.min(), ry_coords.max()

        if rx_min==rx_max:
            rx_norm = np.zeros_like(rx_coords)
        else:
            rx_norm = (rx_coords - rx_min)/(rx_max - rx_min)
        if ry_min==ry_max:
            ry_norm = np.zeros_like(ry_coords)
        else:
            ry_norm = (ry_coords - ry_min)/(ry_max - ry_min)

        region_phase_colors = np.zeros((len(rx_norm),3))
        region_phase_colors[:,0] = rx_norm
        region_phase_colors[:,1] = ry_norm
        ax_gradient.imshow([region_phase_colors], aspect='auto')
        ax_gradient.set_axis_off()
        ax_gradient.set_title("phase gradient", fontsize=24, y=-0.65)

        ax_groundtruth = divider.append_axes("bottom", size="12.5%", pad=0.84)
        reg_labels = self.labels_for_color[region]
        reg_colors = [mcolors.to_rgb(self.get_color(lbl)) for lbl in reg_labels]
        ax_groundtruth.imshow([reg_colors], aspect='auto')
        ax_groundtruth.set_axis_off()
        ax_groundtruth.set_title(
            "dataset group" if self.used_group_coloring else "ground truth class",
            fontsize=24, y=-0.65
        )

        plt.tight_layout()
        os.makedirs("imgs/selected_regions", exist_ok=True)
        fig.savefig(f"imgs/selected_regions/{random_name}_{label_str}.png")
        plt.close(fig)

    def plot_selected_region_collage(self):
        """
        Large "ultrawide" collage mode: 
          - Left column => two subplots stacked: phase + dataset (heatmap if used_group_coloring)
          - Right column => up to 9 spectrogram subplots from unique songs
          - Only display subregions >= 100 points
          - Randomize the order of songs
          - Pad each spectrogram with black on top
          - Also pad the phase gradient + label color arrays to self.max_length
        """
        if self.selected_points is None or len(self.selected_points) == 0:
            return

        # label for file
        first_pt = self.selected_points[0]
        base_label = self.labels_for_color[first_pt]
        if self.used_group_coloring:
            label_str = f"dataset_{base_label}"
        else:
            label_str = (
                f"hdbscan_{base_label}" if self.using_hdbscan 
                else f"groundtruth_{base_label}"
            )
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        # Build a large figure with minimal margins
        fig = plt.figure(figsize=(60, 18))
        outer_gs = fig.add_gridspec(
            1, 2, 
            width_ratios=[1, 3], 
            wspace=0.05,
            top=0.95,
            bottom=0.05,
            left=0.005,
            right=0.995
        )

        left_gs = outer_gs[0,0].subgridspec(2,1, hspace=0.05)
        right_gs= outer_gs[0,1].subgridspec(3,3, wspace=0.2, hspace=0.2)

        ax_phase   = fig.add_subplot(left_gs[0, 0])
        ax_dataset = fig.add_subplot(left_gs[1, 0])

        # Left column embedding
        sel_embed = self.embedding[self.selected_points]
        x_min, x_max = sel_embed[:,0].min(), sel_embed[:,0].max()
        y_min, y_max = sel_embed[:,1].min(), sel_embed[:,1].max()

        if x_min==x_max:
            x_norm = np.zeros_like(sel_embed[:,0])
        else:
            x_norm = (sel_embed[:,0]-x_min)/(x_max - x_min)
        if y_min==y_max:
            y_norm = np.zeros_like(sel_embed[:,1])
        else:
            y_norm = (sel_embed[:,1]-y_min)/(y_max - y_min)

        # Phase
        phase_colors = np.column_stack([x_norm, y_norm, np.zeros_like(x_norm)])
        ax_phase.scatter(x_norm, y_norm, s=6, c=phase_colors, alpha=0.7)
        ax_phase.set_title("Phase Plot", fontsize=30, pad=5)
        ax_phase.set_aspect("equal")
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])

        # Dataset or GT
        if not self.used_group_coloring:
            sc_colors = [self.get_color(lb) for lb in self.labels_for_color[self.selected_points]]
            ax_dataset.scatter(x_norm, y_norm, s=6, c=sc_colors, alpha=0.7)
            ax_dataset.set_title(
                "ground truth" if not self.using_hdbscan else "hdbscan",
                fontsize=24
            )
        else:
            brightness_factor = 2.0
            nbins = 300
            before_mask = np.isin(self.labels_for_color[self.selected_points], [0,1])
            after_mask  = np.isin(self.labels_for_color[self.selected_points], [2,3])

            xedges = np.linspace(0,1, nbins+1)
            yedges = np.linspace(0,1, nbins+1)

            hist_before, _, _ = np.histogram2d(
                x_norm[before_mask], y_norm[before_mask],
                bins=[xedges, yedges]
            )
            hist_after, _, _ = np.histogram2d(
                x_norm[after_mask], y_norm[after_mask],
                bins=[xedges, yedges]
            )
            if hist_before.max()>0:
                hist_before/=hist_before.max()
            if hist_after.max()>0:
                hist_after /=hist_after.max()

            rgb = np.zeros((nbins, nbins, 3))
            rgb[...,0] = hist_before
            rgb[...,2] = hist_before
            rgb[...,1] = hist_after
            rgb *= brightness_factor
            np.clip(rgb,0,1,out=rgb)

            ax_dataset.imshow(rgb.transpose((1,0,2)), extent=[0,1,0,1], origin="lower")
            ax_dataset.set_title(
                "Dataset (Purple=0/1, Green=2/3)", 
                fontsize=26, 
                pad=5
            )

        ax_dataset.set_aspect("equal")
        ax_dataset.set_xticks([])
        ax_dataset.set_yticks([])

        # Right column => up to 9 spectrogram subplots
        unique_songs = np.unique(self.file_indices[self.selected_points])
        np.random.shuffle(unique_songs)  # randomize order
        unique_songs = unique_songs[:9]  # limit to 9

        idx = 0
        for song_id in unique_songs:
            r = idx // 3
            c = idx % 3
            idx += 1

            ax_spec = fig.add_subplot(right_gs[r, c])
            pts_this_song = self.selected_points[self.file_indices[self.selected_points] == song_id]
            if len(pts_this_song)==0:
                ax_spec.set_title(f"Song {song_id} (No Points)")
                ax_spec.axis("off")
                continue

            subreg = self.find_random_contiguous_region(pts_this_song)
            # skip if < 100 points
            if len(subreg) < 100:
                ax_spec.set_title(f"Song {song_id} (<100 pts skipped)")
                ax_spec.axis("off")
                continue

            # Now we also respect self.max_length here for the time dimension
            raw_data = self.spec[subreg]  # shape: (time_len, freq_len)

            time_len = raw_data.shape[0]  
            freq_len = raw_data.shape[1]

            freq_cut = min(freq_len, 250)           # clamp freq to 250
            time_cut = min(time_len, self.max_length)  # clamp time to max_length

            # Create a black-padded array of shape (self.max_length, freq_cut)
            padded_spectrogram = np.zeros((self.max_length, freq_cut), dtype=raw_data.dtype)
            # Place the actual data at the bottom
            padded_spectrogram[-time_cut:, :freq_cut] = raw_data[-time_cut:, :freq_cut]

            # Show it (transpose => freq is vertical dimension)
            ax_spec.imshow(
                padded_spectrogram.T, 
                aspect="auto",
                origin="lower",
                cmap="viridis"
            )
            ax_spec.set_title(f"Song {song_id}", fontsize=20)
            ax_spec.set_xticks([])
            ax_spec.set_yticks([])

            divider = make_axes_locatable(ax_spec)
            ax_gradient = divider.append_axes("bottom", size="12%", pad=0.3)
            ax_labels   = divider.append_axes("bottom", size="12%", pad=0.6)

            sub_embed = self.embedding[subreg]
            sx_min, sx_max = sub_embed[:,0].min(), sub_embed[:,0].max()
            sy_min, sy_max = sub_embed[:,1].min(), sub_embed[:,1].max()

            if sx_min==sx_max:
                sx_norm = np.zeros_like(sub_embed[:,0])
            else:
                sx_norm = (sub_embed[:,0]-sx_min)/(sx_max - sx_min)
            if sy_min==sy_max:
                sy_norm = np.zeros_like(sub_embed[:,1])
            else:
                sy_norm = (sub_embed[:,1]-sy_min)/(sy_max - sy_min)

            # Build the non-padded "phase gradient" array for these points
            pc = np.column_stack([sx_norm, sy_norm, np.zeros_like(sx_norm)])

            # Create a padded version of shape (self.max_length, 3) => black/zero at the top
            pc_padded = np.zeros((self.max_length, 3))
            real_len = len(subreg)
            pc_padded[-real_len:] = pc  # place real data at the bottom

            ax_gradient.imshow([pc_padded], aspect="auto")
            ax_gradient.set_axis_off()
            ax_gradient.set_title("Phase Gradient", fontsize=14, y=-0.65)

            # Build the non-padded label color array
            sub_labs = self.labels_for_color[subreg]
            c_array = [mcolors.to_rgb(self.get_color(lb)) for lb in sub_labs]

            # Create a padded version of shape (self.max_length, 3)
            c_array_padded = np.zeros((self.max_length, 3))
            # Place real colors at bottom
            c_array_padded[-real_len:] = c_array

            ax_labels.imshow([c_array_padded], aspect="auto")
            ax_labels.set_axis_off()
            ax_labels.set_title(
                "Dataset Group" if self.used_group_coloring else "Ground Truth Class",
                fontsize=14, y=-0.65
            )

        # Updated title to focus on group color
        plt.suptitle(
            "Group Color Collage View", 
            fontsize=42,
            y=0.99
        )
        plt.tight_layout()
        os.makedirs("imgs/selected_regions", exist_ok=True)
        fig.savefig(f"imgs/selected_regions/collage_{random_name}_{label_str}.png", dpi=150)
        plt.close(fig)


# Usage example:
file_path = "/media/george-vengrovski/66AA-9C0A/DOI_npz_files/USA5506_PrePostDOI.npz"
selector = UMAPSelector(
    file_path=file_path, 
    max_length=1000, 
    used_group_coloring=True,   
    collage_mode=True
)
selector.plot_umap_with_selection()
