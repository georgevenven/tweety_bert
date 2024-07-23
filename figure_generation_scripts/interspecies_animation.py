import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import librosa
import librosa.display

def load_data(file_path, spectrogram_file_paths):
    data = np.load(file_path, allow_pickle=True)
    embedding = data["embedding_outputs"]
    ground_truth_labels = data["ground_truth_labels"]
    ground_truth_colors = data["ground_truth_colors"].item()
    ground_truth_colors[int(1)] = "#000000"  # Add black to dictionary with key 0
    hdbscan_labels = data["hdbscan_labels"]

    spectrograms = []
    for spectrogram_file_path in spectrogram_file_paths:
        spectrogram = np.load(spectrogram_file_path)
        spectrograms.append(spectrogram)

    return embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, spectrograms

def create_animated_gifs(embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, spectrograms, output_paths, points_per_frame=1, spectrogram_length=200):
    # Normalize the embedding coordinates
    x_coords = embedding[:, 0]
    y_coords = embedding[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_norm = (x_coords - x_min) / (x_max - x_min)
    y_norm = (y_coords - y_min) / (y_max - y_min)

    # Get unique ground truth labels
    unique_labels = np.unique(ground_truth_labels)

    # Calculate the number of frames and time bins
    num_frames = len(embedding)
    num_time_bins = spectrograms[0].shape[1]

    # Calculate the starting frame based on the spectrogram length
    half_length = spectrogram_length // 2
    start_frame = half_length

    # Adjust the total number of frames
    total_frames = num_frames - start_frame

    # Create the animation for the UMAP embedding
    fig_anim, ax_anim = plt.subplots(figsize=(12, 12), dpi=80)
    def animate_umap(frame):
        default_size = 60
        start_idx = frame + start_frame
        end_idx = min(start_idx + points_per_frame, len(embedding))
        colors = np.array(['black'] * len(embedding))
        sizes = np.ones(len(embedding)) * default_size
        colors[start_idx:end_idx] = [ground_truth_colors[label] for label in ground_truth_labels[start_idx:end_idx]]
        sizes[start_idx:end_idx] = 280

        ax_anim.clear()
        for label in unique_labels:
            mask = ground_truth_labels == label
            color = ground_truth_colors[label]
            ax_anim.scatter(x_norm[mask], y_norm[mask], s=sizes[mask], alpha=0.5, color=color, label=f'Ground Truth Label {label}')

        ax_anim.set_aspect('equal')
        ax_anim.set_title('Single Song UMAP Embedding', fontsize=24)
        ax_anim.set_xlabel('UMAP 1', fontsize=24)
        ax_anim.set_ylabel('UMAP 2', fontsize=24)
        ax_anim.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        for spine in ax_anim.spines.values():
            spine.set_linewidth(2)

    anim_umap = animation.FuncAnimation(fig_anim, animate_umap, frames=total_frames, interval=50)
    anim_umap.save(output_paths[0], writer='pillow')
    plt.close(fig_anim)

    # Create the animation for the first spectrogram
    fig_spec1, ax_spec1 = plt.subplots(figsize=(12, 6), dpi=80)
    def animate_spec1(frame):
        center_time_bin = frame + start_frame
        half_length = spectrogram_length // 2
        start_time_bin = max(0, center_time_bin - half_length)
        end_time_bin = min(center_time_bin + half_length, num_time_bins)

        ax_spec1.clear()
        librosa.display.specshow(spectrograms[0][:, start_time_bin:end_time_bin], x_axis=None, y_axis='linear', ax=ax_spec1)
        ax_spec1.set_title('Spectrogram 1', fontsize=24)
        ax_spec1.set_xlabel('Time', fontsize=24)
        ax_spec1.set_ylabel('Frequency', fontsize=24)
        ax_spec1.set_xticks([])

    anim_spec1 = animation.FuncAnimation(fig_spec1, animate_spec1, frames=total_frames, interval=50)
    anim_spec1.save(output_paths[1], writer='pillow')
    plt.close(fig_spec1)

    # Create the animation for the second spectrogram
    fig_spec2, ax_spec2 = plt.subplots(figsize=(12, 6), dpi=80)
    def animate_spec2(frame):
        center_time_bin = frame + start_frame
        half_length = spectrogram_length // 2
        start_time_bin = max(0, center_time_bin - half_length)
        end_time_bin = min(center_time_bin + half_length, num_time_bins)

        vmin = np.min(spectrograms[1])
        vmax = np.max(spectrograms[1])

        ax_spec2.clear()
        librosa.display.specshow(spectrograms[1][:, start_time_bin:end_time_bin], x_axis=None, y_axis='linear', ax=ax_spec2, vmin=vmin, vmax=vmax)
        ax_spec2.set_title('Spectrogram 2', fontsize=24)
        ax_spec2.set_xlabel('Time', fontsize=24)
        ax_spec2.set_ylabel('Frequency', fontsize=24)
        ax_spec2.set_xticks([])

    anim_spec2 = animation.FuncAnimation(fig_spec2, animate_spec2, frames=total_frames, interval=50)
    anim_spec2.save(output_paths[2], writer='pillow')
    plt.close(fig_spec2)

if __name__ == "__main__":
    file_path = "files/labels_Single_Song.npz"
    spectrogram_file_paths = ["/media/george-vengrovski/disk1/yarden_OG_llb3/llb3_0064_2018_04_23_17_30_36_spec.npy", "/home/george-vengrovski/Documents/projects/tweety_bert_paper/concatenated_outputs.npy"]  # Replace with the paths to your precomputed spectrograms
    output_paths = ["umap_animation.gif", "spectrogram1_animation.gif", "spectrogram2_animation.gif"]
    points_per_frame = 2
    spectrogram_length = 100  # Set the desired spectrogram length

    embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, spectrograms = load_data(file_path, spectrogram_file_paths)
    create_animated_gifs(embedding, ground_truth_labels, ground_truth_colors, hdbscan_labels, spectrograms, output_paths, points_per_frame, spectrogram_length)