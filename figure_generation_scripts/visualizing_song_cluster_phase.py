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

CHANGES MADE (WITHOUT REMOVING EXISTING FUNCTIONALITY):
1. When padding, place black on the *right* instead of the left/top, i.e. preserve data on the left side.
2. If a sample is too small (<100 pts) and is excluded, keep looking for another sample to possibly fill up all 9 collage slots.
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
        phase_colors = np.column_stack([x_norm, y_norm, np.zeros_like(x_norm)])
        ax_phase.scatter(x_norm, y_norm, s=8, c=phase_colors, alpha=0.7)  # Slightly reduced point size
        ax_phase.set_title("A. Embedding\n(Colored by Location)", fontsize=14, pad=5, fontweight='bold')  # Two-line title
        
        # Force the phase plot to be a perfect square with dimensions 100x100
        ax_phase.set_xlim(0, 1)
        ax_phase.set_ylim(0, 1)
        ax_phase.set_aspect("equal")
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])

        # 2) If used_group_coloring => purple/green heatmap; else scatter
        if not self.used_group_coloring:
            scatter_colors = [self.get_color(lbl) for lbl in self.labels_for_color[self.selected_points]]
            ax_dataset.scatter(x_norm, y_norm, s=8, c=scatter_colors, alpha=0.7)  # Slightly reduced point size
            ax_dataset.set_title(
                "B. Embedding\n(Colored by Label)",
                fontsize=14,  # Two-line title
                fontweight='bold'
            )
        else:
            brightness_factor = 8.0
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
            ax_dataset.set_title(
                "B. Embedding\n(Colored by Label)", 
                fontsize=14,  # Two-line title
                pad=5,
                fontweight='bold'
            )

        ax_dataset.set_aspect('equal')
        ax_dataset.set_xticks([])
        ax_dataset.set_yticks([])

        # Add 'B' label to the dataset plot
        ax_dataset.text(-0.1, 1.0, 'B', transform=ax_dataset.transAxes, 
                       va='center', ha='right', fontsize=14, fontweight='bold')

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
          - Right section => 2x3 grid of spectrogram subplots (6 total) from unique songs
          - Only display subregions >= 100 points
          - Randomize the order of songs
          - Pad each spectrogram with black on the right
          - Also pad the phase gradient + label colors to self.max_length in the same manner
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

        # Build a large figure with minimal margins - adjusted for 8.5x11 paper ratio
        fig = plt.figure(figsize=(11, 8.5))
        
        # Add more space at the top to prevent title from being cut off
        outer_gs = fig.add_gridspec(
            1, 2, 
            width_ratios=[1, 2.5], 
            wspace=0.1,  # Reduced spacing between left and right sections
            top=0.88,    # Reduced from 0.92 to 0.88 to add significantly more space at the top
            bottom=0.02,
            left=0.02,
            right=0.98
        )

        left_gs = outer_gs[0,0].subgridspec(2,1, hspace=0.1)  # Reduced spacing between phase and dataset plots
        right_gs= outer_gs[0,1].subgridspec(3,2, wspace=0.15, hspace=0.2)  # Changed to 3 rows, 2 columns

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
        ax_phase.scatter(x_norm, y_norm, s=8, c=phase_colors, alpha=0.7)  # Slightly reduced point size
        ax_phase.set_title("A. Embedding\n(Colored by Location)", fontsize=14, pad=5, fontweight='bold')  # Two-line title
        
        # Force the phase plot to be a perfect square with dimensions 100x100
        ax_phase.set_xlim(0, 1)
        ax_phase.set_ylim(0, 1)
        ax_phase.set_aspect("equal")
        ax_phase.set_xticks([])
        ax_phase.set_yticks([])

        # Dataset or GT
        if not self.used_group_coloring:
            sc_colors = [self.get_color(lb) for lb in self.labels_for_color[self.selected_points]]
            ax_dataset.scatter(x_norm, y_norm, s=8, c=sc_colors, alpha=0.7)  # Slightly reduced point size
            ax_dataset.set_title(
                "B. Embedding\n(Colored by Label)",
                fontsize=14,  # Two-line title
                fontweight='bold'
            )
        else:
            # Increased brightness factor specifically for the dataset plot
            brightness_factor = 4.0  # Increased from 2.0 to 4.0 for more vibrant colors
            nbins = 100  # Set to 100 for 100x100 heatmap
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
            
            # Normalize each histogram separately to make colors more vibrant
            if hist_before.max()>0:
                hist_before /= hist_before.max()
            if hist_after.max()>0:
                hist_after /= hist_after.max()

            # Create RGB image with more saturated colors
            rgb = np.zeros((nbins, nbins, 3))
            # For purple (more saturated)
            rgb[...,0] = hist_before * 0.8  # R component (slightly reduced for more saturated purple)
            rgb[...,2] = hist_before        # B component
            # For green (more saturated)
            rgb[...,1] = hist_after         # G component

            # Apply brightness factor and clip
            rgb *= brightness_factor
            np.clip(rgb, 0, 1, out=rgb)

            ax_dataset.imshow(rgb.transpose((1,0,2)), extent=[0,1,0,1], origin="lower")
            ax_dataset.set_title(
                "B. Embedding\n(Colored by Label)", 
                fontsize=14,  # Two-line title
                pad=5,
                fontweight='bold'
            )

        # Add 'B' label to the dataset plot
        ax_dataset.text(-0.1, 1.0, 'B', transform=ax_dataset.transAxes, 
                       va='center', ha='right', fontsize=14, fontweight='bold')

        # Force the dataset plot to be a perfect square with dimensions 100x100
        ax_dataset.set_xlim(0, 1)
        ax_dataset.set_ylim(0, 1)
        ax_dataset.set_aspect("equal")
        ax_dataset.set_xticks([])
        ax_dataset.set_yticks([])

        # Collage: up to 6 spectrogram subplots (2x3 grid). But if a sample is too small (<100 pts),
        # skip it and try the next. We continue until we either find 6 valid songs or run out.
        all_songs = np.unique(self.file_indices[self.selected_points])
        np.random.shuffle(all_songs)

        valid_songs = []
        for s in all_songs:
            pts_this_song = self.selected_points[self.file_indices[self.selected_points] == s]
            if len(pts_this_song) >= 100:
                valid_songs.append(s)
            if len(valid_songs) == 6:  # Changed from 9 to 6
                break

        # Add 'C' label and title above the spectrograms section
        if len(valid_songs) > 0:
            # Calculate the center position of the right section (spectrograms)
            # This ensures the title is centered over just the spectrogram columns
            right_center = 0.5 + 1/(2*3.5)  # Adjusted based on width_ratios=[1, 2.5]
            
            # Add centered title for the spectrogram collage section, split into two lines
            # Position adjusted to 0.94 to place it in the middle of the top margin
            fig.text(right_center, 0.94, 'C. Spectrogram Collage\nBreeding Season (Purple), Non-breeding Season (Green)', 
                    fontsize=14, fontweight='bold', ha='center')

        # We will create up to 6 subplots for valid songs
        for i in range(6):  # Changed from 9 to 6
            ax_spec = fig.add_subplot(right_gs[i//2, i%2])  # Changed from i//3, i%3 to i//2, i%2
            if i >= len(valid_songs):
                # no more valid songs to show
                ax_spec.axis("off")
                continue

            song_id = valid_songs[i]
            pts_this_song = self.selected_points[self.file_indices[self.selected_points] == song_id]

            subreg = self.find_random_contiguous_region(pts_this_song)
            # Should be >= 100 if we put it in valid_songs
            raw_data = self.spec[subreg]  # shape: (time_len, freq_len)

            time_len = raw_data.shape[0]  
            freq_len = raw_data.shape[1]

            freq_cut = min(freq_len, 250)            # clamp freq dimension
            time_cut = min(time_len, self.max_length)  # clamp time dimension

            # Create a black-padded array of shape (self.max_length, freq_cut)
            # CHANGED: put actual data in the *left* portion (so black is on the right).
            padded_spectrogram = np.zeros((self.max_length, freq_cut), dtype=raw_data.dtype)
            padded_spectrogram[:time_cut, :freq_cut] = raw_data[:time_cut, :freq_cut]

            ax_spec.imshow(
                padded_spectrogram.T, 
                aspect="auto",
                origin="lower",
                cmap="viridis"
            )
            # Remove the song title to make spectrograms taller
            ax_spec.set_xticks([])
            ax_spec.set_yticks([])

            # Also pad the phase gradient and label colors to self.max_length in the same manner
            divider = make_axes_locatable(ax_spec)
            ax_gradient = divider.append_axes("bottom", size="12%", pad=0.3)  # Reduced size and padding
            ax_labels   = divider.append_axes("bottom", size="12%", pad=0.1)  # Reduced padding between bars

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

            # Create a padded version of shape (self.max_length, 3) => black/zero on the *right* side
            # but we handle time as vertical dimension, so to preserve the same logic
            # we fill from top to bottom. We'll keep consistent with how we padded spectrogram:
            pc_padded = np.zeros((self.max_length, 3))
            real_len = len(subreg)
            pc_padded[:real_len] = pc  # place real data from row 0 to row real_len

            ax_gradient.imshow([pc_padded], aspect="auto")
            ax_gradient.set_axis_off()

            # Build the non-padded label color array
            sub_labs = self.labels_for_color[subreg]
            c_array = [mcolors.to_rgb(self.get_color(lb)) for lb in sub_labs]

            c_array_padded = np.zeros((self.max_length, 3))
            c_array_padded[:real_len] = c_array

            ax_labels.imshow([c_array_padded], aspect="auto")
            ax_labels.set_axis_off()
            
            # Add C and D labels next to the gradient and label bars
            ax_gradient.text(-0.05, 0.5, 'E', transform=ax_gradient.transAxes, 
                            va='center', ha='right', fontsize=12, fontweight='bold')
            ax_labels.text(-0.05, 0.5, 'F', transform=ax_labels.transAxes, 
                          va='center', ha='right', fontsize=12, fontweight='bold')

        plt.tight_layout()
        os.makedirs("imgs/selected_regions", exist_ok=True)
        fig.savefig(f"imgs/selected_regions/collage_{random_name}_{label_str}.png", dpi=300)  # Increased DPI for print quality
        plt.close(fig)

if __name__ == "__main__":
    file_path = "files/Calls_Test.npz"
    selector = UMAPSelector(
        file_path=file_path, 
        max_length=100, 
        used_group_coloring=False,   
        collage_mode=True
    )
    selector.plot_umap_with_selection()
