import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import os
from collections import Counter
import umap
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
import glasbey
from sklearn.metrics.cluster import completeness_score
import seaborn as sns
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
import pickle
from itertools import cycle
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)


def average_colors_per_sample(ground_truth_labels, cmap):
    """
    Averages the colors for each prediction point based on the ground truth labels.
    
    Parameters:
    - ground_truth_labels: numpy array of shape (samples, labels_per_sample)
    - cmap: matplotlib.colors.ListedColormap object

    Returns:
    - averaged_colors: numpy array of shape (samples, 3) containing the averaged RGB colors
    """
    # Initialize an array to hold the averaged colors
    averaged_colors = np.zeros((ground_truth_labels.shape[0], 3))

    for i, labels in enumerate(ground_truth_labels):
        # Retrieve the colors for each label using the colormap
        colors = cmap(labels / np.max(ground_truth_labels))[:, :3]  # Normalize labels and exclude alpha channel

        # Average the colors for the current sample
        averaged_colors[i] = np.mean(colors, axis=0)

    return averaged_colors

def load_data(data_dir, context=1000):
    """
    Create a DataLoader from your custom dataset.
    """
    dataset = SongDataSet_Image(
        data_dir, num_classes=50, infinite_loader=False,
        pitch_shift=False, segment_length=None
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)
    return loader

def generate_hdbscan_labels(array, min_samples=1, min_cluster_size=5000):
    """
    Generate labels for data points using HDBSCAN.
    """
    import hdbscan

    hdbscan_model = hdbscan.HDBSCAN(
        min_samples=min_samples, min_cluster_size=min_cluster_size
    )
    labels = hdbscan_model.fit_predict(array)
    print(f"discovered labels {np.unique(labels)}")
    return labels

def plot_umap_projection(
    model,
    device,
    data_dirs,
    category_colors_file="test_llb16",
    samples=1e6,
    file_path='category_colors.pkl',
    layer_index=None,
    dict_key=None,
    context=1000,
    save_name=None,
    raw_spectogram=False,
    save_dict_for_analysis=True,
    remove_non_vocalization=True,
    min_cluster_size=500
):
    """
    This function:
    1) Iterates over multiple data directories.
    2) Extracts data in context-sized chunks, runs inference if needed.
    3) Optionally filters non-vocal segments.
    4) If the total data collected goes beyond 'samples', it truncates the
       final batch (i.e., the last song) so that we don't exceed the budget.
    5) Runs UMAP dimensionality reduction.
    6) Clusters via HDBSCAN.
    7) Plots results and saves arrays.

    Key difference from the random-subsample version:
    - We never pick random frames to reach the 'samples' limit. Instead, as soon
      as we exceed 'samples' in the last song, we slice only the portion of that
      last batch needed and then stop collecting further data.
    """

    # 1. Convert 'samples' to an integer, if float was given.
    samples = int(samples)

    # 2. Prepare lists that hold aggregated data across all directories
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = []
    vocalization_arr = []
    file_indices_arr = []
    dataset_indices_arr = []

    # 3. For storing which file index maps to which path
    file_map = {}
    current_file_index = 0

    # 4. Make a directory for saving images
    if save_name is None:
        save_name = "umap_projection"
    experiment_dir = os.path.join("imgs", "umap_plots", save_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 5. Prepare a figure space (though real plotting happens after UMAP)
    plt.figure(figsize=(8, 6))

    # 6. If no data directories, bail
    if len(data_dirs) < 1:
        print("No data directories provided. Exiting.")
        return

    total_samples_collected = 0
    num_datasets = len(data_dirs)

    # if you still want some nominal logic about "samples per dataset," keep it:
    samples_per_dataset = samples // num_datasets

    # ------------------------------------------------
    # 7) LOAD EACH DATASET, BATCH BY BATCH
    # ------------------------------------------------
    for dataset_idx, data_dir in enumerate(data_dirs):
        print(f"Loading dataset {dataset_idx + 1}/{num_datasets} from {data_dir}")
        data_loader = load_data(data_dir=data_dir, context=context)
        data_loader_iter = iter(data_loader)

        dataset_samples_collected = 0

        # We'll keep iterating until we collect enough from this dataset,
        # or exhaust it, or hit the global limit
        while dataset_samples_collected < samples_per_dataset:
            try:
                data, ground_truth_label, vocalization, this_file_path = next(data_loader_iter)

                # skip if shape is suspicious
                if data.shape[1] < 100:
                    print("Skipping batch because data.shape[1] < 100")
                    continue

                # skip if we've already handled this file
                if this_file_path in file_map.values():
                    print(f"Skipping file {this_file_path}, already in file_map.")
                    continue

                # record this new file
                file_map[current_file_index] = this_file_path

                # number of classes
                num_classes = ground_truth_label.shape[-1]
                original_data_length = data.shape[1]
                original_label_length = ground_truth_label.shape[1]

                # pad to multiple of context
                pad_size_time = (context - (original_data_length % context)) % context
                if pad_size_time > 0:
                    data = F.pad(data, (0, 0, 0, pad_size_time), 'constant', 0)
                    vocalization = F.pad(vocalization, (0, pad_size_time), 'constant', 0)
                    ground_truth_label = F.pad(
                        ground_truth_label, (0, 0, 0, pad_size_time), 'constant', 0
                    )

                total_time = data.shape[1]
                num_times = total_time // context

                # reshape
                data = data.reshape(num_times, context, data.shape[-1])
                ground_truth_label = ground_truth_label.reshape(num_times, context, num_classes)
                vocalization = vocalization.reshape(num_times, context)

                # run inference if not raw
                if not raw_spectogram:
                    data_for_model = data.unsqueeze(1).permute(0, 1, 3, 2)
                    with torch.no_grad():
                        _, layers = model.inference_forward(data_for_model.to(device))

                    layer_output_dict = layers[layer_index]
                    output = layer_output_dict.get(dict_key, None)
                    if output is None:
                        print(f"Warning: invalid dict_key='{dict_key}' for this batch. Skipping.")
                        continue

                    b, t, f = output.shape
                    if b != num_times or t != context:
                        print("Warning: mismatch in model output shape. Skipping batch.")
                        continue

                    predictions_this_batch = output.reshape(b * t, f)
                    predictions_this_batch = predictions_this_batch[:original_data_length]
                else:
                    # raw spec
                    predictions_this_batch = data.reshape(num_times * context, data.shape[-1])
                    predictions_this_batch = predictions_this_batch[:original_data_length]

                # flatten labels, voc, etc.
                ground_truth_label_flat = ground_truth_label.reshape(
                    num_times * context, num_classes
                )[:original_label_length]
                vocalization_flat = vocalization.reshape(num_times * context)[:original_data_length]
                ground_truth_label_int = torch.argmax(
                    ground_truth_label_flat, dim=-1
                ).cpu().numpy()

                spec_this_batch = data.reshape(num_times * context, data.shape[-1])
                spec_this_batch = spec_this_batch[:original_data_length]

                # shape checks
                n_pred = len(predictions_this_batch)
                n_label = len(ground_truth_label_int)
                n_voc = len(vocalization_flat)
                n_spec = len(spec_this_batch)
                if not (n_pred == n_label == n_voc == n_spec):
                    print(
                        f"Shape mismatch: predictions={n_pred}, "
                        f"labels={n_label}, voc={n_voc}, spec={n_spec}. Skipping batch."
                    )
                    continue

                # now see if appending these frames will exceed our global sample limit
                # if so, we only keep the portion needed to reach 'samples'
                space_left = samples - total_samples_collected
                if space_left <= 0:
                    # we've already reached the global limit, just stop entirely
                    break

                if n_pred > space_left:
                    # we only want the first `space_left` frames from this last batch
                    predictions_this_batch = predictions_this_batch[:space_left]
                    ground_truth_label_int = ground_truth_label_int[:space_left]
                    spec_this_batch = spec_this_batch[:space_left]
                    vocalization_flat = vocalization_flat[:space_left]
                    n_pred = space_left  # now we truncated to exactly space_left

                # create file/dataset indices arrays for these frames
                file_indices_this_batch = np.full(
                    n_pred, current_file_index, dtype=np.int64
                )
                dataset_indices_this_batch = np.full_like(
                    file_indices_this_batch, dataset_idx, dtype=np.int64
                )

                # everything is consistent, append
                predictions_arr.append(predictions_this_batch)
                ground_truth_labels_arr.append(ground_truth_label_int)
                vocalization_arr.append(vocalization_flat.cpu().numpy())
                spec_arr.append(spec_this_batch)
                file_indices_arr.append(file_indices_this_batch)
                dataset_indices_arr.append(dataset_indices_this_batch)

                # increment counters
                current_file_index += 1
                dataset_samples_collected += n_pred
                total_samples_collected += n_pred

                # stop if we are at or past either the dataset or global limit
                if dataset_samples_collected >= samples_per_dataset:
                    break
                if total_samples_collected >= samples:
                    break

            except StopIteration:
                print(f"Exhausted dataset {data_dir} after {dataset_samples_collected} samples.")
                break

        # if the global limit is reached, stop reading new datasets
        if total_samples_collected >= samples:
            break

    # -------------------------
    # CONCATENATE RESULTS
    # -------------------------
    if len(predictions_arr) == 0:
        print("No data was collected at all. Exiting.")
        return

    predictions = np.concatenate(predictions_arr, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    vocalization_arr = np.concatenate(vocalization_arr, axis=0)
    spec_arr = np.concatenate(spec_arr, axis=0)
    file_indices = np.concatenate(file_indices_arr, axis=0)
    dataset_indices = np.concatenate(dataset_indices_arr, axis=0)

    print(f"Collected shapes:\n"
          f"  predictions: {predictions.shape}\n"
          f"  ground_truth_labels: {ground_truth_labels.shape}\n"
          f"  vocalization_arr: {vocalization_arr.shape}\n"
          f"  spec_arr: {spec_arr.shape}")

    if not (len(predictions) == len(ground_truth_labels) == len(vocalization_arr) == len(spec_arr)):
        print("Final shape mismatch detected. Not proceeding to UMAP.")
        return

    # ------------------------
    # OPTIONAL VOCAL FILTER
    # ------------------------
    if remove_non_vocalization:
        idx_vocal = np.where(vocalization_arr == 1)[0]
        print(f"Filtering by vocalization == 1, from {len(predictions)} to {len(idx_vocal)} samples.")
        predictions = predictions[idx_vocal]
        ground_truth_labels = ground_truth_labels[idx_vocal]
        spec_arr = spec_arr[idx_vocal]
        file_indices = file_indices[idx_vocal]
        dataset_indices = dataset_indices[idx_vocal]
        vocalization_arr = vocalization_arr[idx_vocal]

    if len(predictions) < 2:
        print("Insufficient data after vocalization filtering. Exiting.")
        return

    # final shape check
    n_final = len(predictions)
    print(f"Final data shape: {predictions.shape} samples by {predictions.shape[1]} features")

    # -----------------------------
    # RUN UMAP
    # -----------------------------
    if n_final <= 200:
        neighbor_count = max(2, n_final // 2)
        print(f"Warning: reducing n_neighbors to {neighbor_count} due to small dataset.")
    else:
        neighbor_count = 200

    print("Initializing UMAP reducer...")
    reducer = umap.UMAP(
        n_neighbors=neighbor_count,
        min_dist=0,
        n_components=2,
        metric='cosine'
    )
    print("Fitting UMAP...")
    embedding_outputs = reducer.fit_transform(predictions)
    print("UMAP fitting complete. Shape of embedding:", embedding_outputs.shape)

    # -----------------------------
    # HDBSCAN CLUSTERING
    # -----------------------------
    print("Generating HDBSCAN labels...")
    hdbscan_labels = generate_hdbscan_labels(
        embedding_outputs,
        min_samples=1,
        min_cluster_size=min_cluster_size
    )
    print("HDBSCAN done. Unique labels:", np.unique(hdbscan_labels))

    # -----------------------------
    # COLOR & PLOT
    # -----------------------------
    unique_clusters = np.unique(hdbscan_labels)
    unique_ground_truth_labels = np.unique(ground_truth_labels)

    def create_color_palette(n_colors):
        """
        Create an RGB color palette of size n_colors 
        using glasbey's distinct color approach.
        """
        colors = glasbey.create_palette(palette_size=n_colors)

        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip('#')
            return tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))

        return [hex_to_rgb(color) for color in colors]

    n_ground_truth = len(unique_ground_truth_labels)
    n_clusters = len(unique_clusters)
    ground_truth_colors = create_color_palette(n_ground_truth)
    hdbscan_colors = create_color_palette(n_clusters)

    # if -1 in cluster labels, color it black
    if -1 in unique_clusters:
        i_neg1 = np.where(unique_clusters == -1)[0][0]
        hdbscan_colors[i_neg1] = (0, 0, 0)

    # plot hdbscan
    fig1, ax1 = plt.subplots(figsize=(16, 16), facecolor='white')
    ax1.set_facecolor('white')
    sc1 = ax1.scatter(
        embedding_outputs[:, 0],
        embedding_outputs[:, 1],
        c=hdbscan_labels,
        s=10,
        alpha=0.1,
        cmap=mcolors.ListedColormap(hdbscan_colors)
    )
    ax1.set_title("HDBSCAN Discovered Labels", fontsize=48)
    ax1.set_xlabel("UMAP Dimension 1", fontsize=48)
    ax1.set_ylabel("UMAP Dimension 2", fontsize=48)
    ax1.tick_params(axis='both', which='both', bottom=False, left=False, 
                    labelbottom=False, labelleft=False)
    plt.tight_layout()
    hdbscan_figpath = os.path.join(experiment_dir, "hdbscan_labels.png")
    plt.savefig(hdbscan_figpath, facecolor=fig1.get_facecolor(), edgecolor='none')
    plt.close(fig1)

    # plot ground truth
    fig2, ax2 = plt.subplots(figsize=(16, 16), facecolor='white')
    ax2.set_facecolor('white')
    sc2 = ax2.scatter(
        embedding_outputs[:, 0],
        embedding_outputs[:, 1],
        c=ground_truth_labels,
        s=10,
        alpha=0.1,
        cmap=mcolors.ListedColormap(ground_truth_colors)
    )
    ax2.set_title("Ground Truth Labels", fontsize=48)
    ax2.set_xlabel("UMAP Dimension 1", fontsize=48)
    ax2.set_ylabel("UMAP Dimension 2", fontsize=48)
    ax2.tick_params(axis='both', which='both', bottom=False, left=False, 
                    labelbottom=False, labelleft=False)
    plt.tight_layout()
    gt_figpath = os.path.join(experiment_dir, "ground_truth_labels.png")
    plt.savefig(gt_figpath, facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close(fig2)

    # ---------------------
    # SAVE THE DATA
    # ---------------------
    save_path = os.path.join("files", f"{save_name}.npz")
    print(f"Saving arrays to {save_path} ...")
    np.savez(
        save_path,
        embedding_outputs=embedding_outputs,
        hdbscan_labels=hdbscan_labels,
        ground_truth_labels=ground_truth_labels,
        predictions=predictions,
        s=spec_arr,
        hdbscan_colors=hdbscan_colors,
        ground_truth_colors=ground_truth_colors,
        vocalization=vocalization_arr,
        file_indices=file_indices,
        dataset_indices=dataset_indices,
        file_map=file_map
    )
    print("Done!")



# def apply_windowing(arr, window_size, stride, flatten_predictions=False):
#     """
#     Apply windowing to the input array.

#     Parameters:
#     - arr: The input array to window, expected shape (num_samples, features) for predictions and (num_samples,) for labels.
#     - window_size: The size of each window.
#     - stride: The stride between windows.
#     - flatten_predictions: A boolean indicating whether to flatten the windowed predictions.

#     Returns:
#     - windowed_arr: The windowed version of the input array.
#     """
#     num_samples, features = arr.shape if len(arr.shape) > 1 else (arr.shape[0], 1)
#     num_windows = (num_samples - window_size) // stride + 1
#     windowed_arr = np.lib.stride_tricks.as_strided(
#         arr,
#         shape=(num_windows, window_size, features),
#         strides=(arr.strides[0] * stride, arr.strides[0], arr.strides[-1]),
#         writeable=False
#     )

#     if flatten_predictions and features > 1:
#         # Flatten each window for predictions
#         windowed_arr = windowed_arr.reshape(num_windows, -1)
    
#     return windowed_arr


# def sliding_window_umap(model, device, data_dir="test_llb16",
#                          remove_silences=False, samples=100, category_colors_file='category_colors.pkl', 
#                          layer_index=None, dict_key=None, time_bins_per_umap_point=100, 
#                          context=1000, save_dir=None, raw_spectogram=False, save_dict_for_analysis=False, compute_svm=False, color_scheme="Syllable", window_size=100, stride=1):
#     predictions_arr = []
#     ground_truth_labels_arr = []
#     spec_arr = [] 

#     # Reset Figure
#     plt.figure(figsize=(8, 6))

#     # to allow sci notation 
#     samples = int(samples)
#     total_samples = 0

#     data_loader = load_data(data_dir=data_dir, context=context)
#     data_loader_iter = iter(data_loader)

#     while total_samples < samples:
#         try:
#             # Retrieve the next batch
#             data, ground_truth_label = next(data_loader_iter)
            
#             # if smaller than context window, go to next song
#             if data.shape[-2] < context:
#                 continue 

#             # because network is made to work with batched data, we unsqueeze a dim and transpose the last two dims (usually handled by collate fn)
#             data = data.unsqueeze(0).permute(0,1,3,2)

#             # calculate the number of times a song 
#             num_times = data.shape[-1] // context
            
#             # removing left over timebins that do not fit in context window 
#             shave_index = num_times * context
#             data = data[:,:,:,:shave_index]

#             batch, channel, freq, time_bins = data.shape 

#             # cheeky reshaping operation to reshape the length of the song that is larger than the context window into multiple batches 
#             data = data.permute(0,-1, 1, 2)
#             data = data.reshape(num_times, context, channel, freq)
#             data = data.permute(0,2,3,1)

#             # reshaping g truth labels to be consistent 
#             batch, time_bins, labels = ground_truth_label.shape

#             # shave g truth labels 
#             ground_truth_label = ground_truth_label.permute(0,2,1)
#             ground_truth_label = ground_truth_label[:,:,:shave_index]

#             # cheeky reshaping operation to reshape the length of the song that is larger than the context window into multiple batches 
#             ground_truth_label = ground_truth_label.permute(0,2,1)
#             ground_truth_label = ground_truth_label.reshape(num_times, context, labels)
            
#         except StopIteration:
#             # if test set is exhausted, print the number of samples collected and stop the collection process
#             print(f"samples collected {len(ground_truth_labels_arr) * context}")
#             break

#         if raw_spectogram == False:
#             _, layers = model.inference_forward(data.to(device))

#             layer_output_dict = layers[layer_index]
#             output = layer_output_dict.get(dict_key, None)

#             if output is None:
#                 print(f"Invalid key: {dict_key}. Skipping this batch.")
#                 continue

#             batches, time_bins, features = output.shape 
#             # data shape [0] is the number of batches, 
#             predictions = output.reshape(batches, time_bins, features)
#             # combine the batches and number of samples per context window 
#             predictions = predictions.flatten(0,1)
#             predictions_arr.append(predictions.detach().cpu().numpy())

#         # remove channel dimension 
#         data = data.squeeze(1)
#         spec = data

#         # set the features (freq axis to be the last dimension)
#         spec = spec.permute(0, 2, 1)
#         # combine batches and timebins
#         spec = spec.flatten(0, 1)

#         ground_truth_label = ground_truth_label.flatten(0, 1)
#         ground_truth_label = torch.argmax(ground_truth_label, dim=-1)

#         spec_arr.append(spec.cpu().numpy())
#         ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
        
#         total_samples += spec.shape[0]

#     # convert the list of batch * samples * features to samples * features 
#     ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
#     spec_arr = np.concatenate(spec_arr, axis=0)

#     if not raw_spectogram:
#         predictions = np.concatenate(predictions_arr, axis=0)
#     else:
#         predictions = spec_arr

#     # razor off any extra datapoints 
#     if samples > len(predictions):
#         samples = len(predictions)
#     else:
#         predictions = predictions[:samples]
#         ground_truth_labels = ground_truth_labels[:samples]

#     print(predictions.shape)

#     # Ensure predictions are in the correct shape (num_samples, features) before windowing
#     predictions = apply_windowing(predictions, window_size, stride=stride, flatten_predictions=True)
#     ground_truth_labels = apply_windowing(ground_truth_labels.reshape(-1, 1), window_size, stride=stride, flatten_predictions=False)

#     ground_truth_labels = ground_truth_labels.squeeze()
    
#     # Fit the UMAP reducer       
#     reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')

#     embedding_outputs = reducer.fit_transform(predictions)
#     hdbscan_labels = generate_hdbscan_labels(embedding_outputs)

#     np.savez(f"save_dir", embedding_outputs=embedding_outputs, hdbscan_labels=hdbscan_labels, ground_truth_labels=ground_truth_labels, s=spec_arr)
    
#     # Assuming 'glasbey' is a predefined object with a method 'extend_palette'
#     cmap = glasbey.extend_palette(["#000000"], palette_size=30)
#     cmap = mcolors.ListedColormap(cmap)    

#     ground_truth_labels = average_colors_per_sample(ground_truth_labels, cmap)

#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

#     axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=hdbscan_labels, s=10, alpha=.1, cmap=cmap)
#     axes[0].set_title("HDBSCAN")

#     # Plot with the original color scheme
#     axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=ground_truth_labels, s=10, alpha=.1, cmap=cmap)
#     axes[1].set_title("Original Coloring")

#     # Adjust title based on 'raw_spectogram' flag
#     if raw_spectogram:
#         plt.title(f'UMAP of Spectogram', fontsize=14)
#     else:
#         plt.title(f'UMAP Projection of (Layer: {layer_index}, Key: {dict_key})', fontsize=14)

#     # Save the plot if 'save_dir' is specified, otherwise display it
#     if save_dir:
#         plt.savefig(save_dir, format='png')
#     else:
#         plt.show()

import numpy as np
from collections import Counter
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

class ComputerClusterPerformance():
    """
    This class computes clustering performance metrics such as 
    homogeneity, completeness, and V-measure using ground-truth 
    labels and HDBSCAN cluster assignments. 

    Key Modifications:
    - Noise points (where hdbscan_labels == -1) are no longer removed. 
      Instead, each noise point is replaced by the nearest non-noise 
      cluster label, if one can be found by looking to the left or right. 
      If no non-noise label can be found, it remains -1.
    - Silence frames were originally represented as 0. Now, silence 
      is represented as -1 to align with the concept of no assignment.
      The `syllable_to_phrase_labels` method is called with silence=-1, 
      ensuring phrase segmentation works with the updated silence value.
    """

    def __init__(self, labels_path=None):
        """
        Parameters:
        - labels_path: list of str
          List of paths to .npz files containing 
          'hdbscan_labels' and 'ground_truth_labels' arrays.
        """
        self.labels_paths = labels_path
            
        
    @staticmethod
    def syllable_to_phrase_labels(arr, silence=-1):
        """
        Convert a sequence of syllable labels into a sequence of phrase labels,
        merging silence bins with their nearest adjacent syllables.

        For each contiguous block of silence:
        - If it's bounded by the same label on both sides, assign that label to all.
        - If it's bounded by two different labels, assign each time-bin to the closer label;
        ties go to the left.
        - If it's at the beginning or end (missing one side), assign to the available side.

        Parameters
        ----------
        arr : np.ndarray
            Array of integer labels, where `silence` frames are indicated by `silence`.
        silence : int, optional
            Integer value representing silence, by default -1.

        Returns
        -------
        np.ndarray
            Array of phrase-level labels with silence frames appropriately merged.
        """
        new_arr = np.array(arr, dtype=int)
        length = len(new_arr)
        if length == 0:
            return new_arr  # Edge case: empty input

        # Helper function to find contiguous regions of silence
        def find_silence_runs(labels):
            runs = []
            in_silence = False
            start = None

            for i, val in enumerate(labels):
                if val == silence and not in_silence:
                    in_silence = True
                    start = i
                elif val != silence and in_silence:
                    runs.append((start, i - 1))
                    in_silence = False
            # If ended in silence
            if in_silence:
                runs.append((start, length - 1))
            return runs

        # Identify contiguous silence regions
        silence_runs = find_silence_runs(new_arr)

        for start_idx, end_idx in silence_runs:
            # Check left and right labels
            left_label = new_arr[start_idx - 1] if start_idx > 0 else None
            right_label = new_arr[end_idx + 1] if end_idx < length - 1 else None

            if left_label is None and right_label is None:
                # Entire array is silence or single region with no bounding labels
                # Do nothing or choose some default strategy
                continue
            elif left_label is None:
                # Leading silence; merge with right label
                new_arr[start_idx:end_idx+1] = right_label
            elif right_label is None:
                # Trailing silence; merge with left label
                new_arr[start_idx:end_idx+1] = left_label
            elif left_label == right_label:
                # Same label on both sides
                new_arr[start_idx:end_idx+1] = left_label
            else:
                # Different labels on both sides
                # Assign each bin to whichever side is closer; ties go left
                left_distances = np.arange(start_idx, end_idx + 1) - (start_idx - 1)
                right_distances = (end_idx + 1) - np.arange(start_idx, end_idx + 1)

                for i in range(start_idx, end_idx + 1):
                    # Distance from left non-silence is (i - (start_idx - 1))
                    dist_left = i - (start_idx - 1)
                    # Distance from right non-silence is ((end_idx + 1) - i)
                    dist_right = (end_idx + 1) - i

                    if dist_left < dist_right:
                        new_arr[i] = left_label
                    elif dist_right < dist_left:
                        new_arr[i] = right_label
                    else:
                        # Tie -> go left
                        new_arr[i] = left_label

        return new_arr

    @staticmethod
    def majority_vote(data, window_size=1):
        """
        Return an array of the same length as 'data',
        where each index i is replaced by the majority over
        a window around i. No padding is added.
        """
        from collections import Counter
        data = np.asarray(data)
        n = len(data)
        
        # If window_size=1, no smoothing
        if window_size <= 1 or n == 0:
            return data.copy()
        
        half_w = window_size // 2
        output = np.zeros_like(data)
        
        for i in range(n):
            # define start/end, clamped
            start = max(0, i - half_w)
            end   = min(n, i + half_w + 1)
            window = data[start:end]
            
            # majority
            c = Counter(window)
            major_label = max(c, key=c.get)  # picks the label with highest freq
            output[i] = major_label
        
        return output


    def fill_noise_with_nearest_label(self, labels):
        """
        For each noise point (labeled -1), find the nearest non-noise 
        label to the left or right and assign it to this point. If no 
        non-noise label is found, it remains -1.
        
        Parameters:
        - labels: np.ndarray
          Array of cluster labels where -1 indicates noise.

        Returns:
        - labels: np.ndarray
          Array with noise points replaced by the nearest non-noise labels, 
          when possible.
        """
        noise_indices = np.where(labels == -1)[0]
        for idx in noise_indices:
            # Search left
            left_idx = idx - 1
            while left_idx >= 0 and labels[left_idx] == -1:
                left_idx -= 1
            
            # Search right
            right_idx = idx + 1
            while right_idx < len(labels) and labels[right_idx] == -1:
                right_idx += 1
            
            # Compute distances if valid
            left_dist = (idx - left_idx) if (left_idx >= 0 and labels[left_idx] != -1) else np.inf
            right_dist = (right_idx - idx) if (right_idx < len(labels) and labels[right_idx] != -1) else np.inf

            # Assign based on nearest non-noise label
            if left_dist == np.inf and right_dist == np.inf:
                # No non-noise neighbors found, remain -1
                continue
            elif left_dist < right_dist:
                labels[idx] = labels[left_idx]
            elif right_dist < left_dist:
                labels[idx] = labels[right_idx]
            else:
                # Equidistant, pick left arbitrarily
                labels[idx] = labels[left_idx]

        return labels

    def compute_vmeasure_score(self):
        """
        Compute the homogeneity, completeness, and V-measure metrics for all 
        label files provided at initialization.

        Steps:
        1. Load HDBSCAN labels and ground-truth labels from npz files.
        2. Instead of removing noise points (HDBSCAN label = -1), replace them 
           with the nearest non-noise label if possible, or leave as -1 if not.
        3. Convert silence from 0 to -1 in ground_truth_labels for consistency.
        4. Convert ground_truth_labels to phrase labels using `syllable_to_phrase_labels` with silence=-1.
        5. Compute homogeneity, completeness, and v-measure using these cleaned labels.
        6. Return a dictionary of metrics, each containing a tuple of (mean, standard_error_of_mean).

        Returns:
        - metrics: dict
          {
            'Homogeneity': (mean_h, sem_h),
            'Completeness': (mean_c, sem_c),
            'V-measure': (mean_v, sem_v)
          }
        """
        homogeneity_scores = []
        completeness_scores = []
        v_measure_scores = []

        for path_index, path in enumerate(self.labels_paths):
            f = np.load(path)
            hdbscan_labels = f['hdbscan_labels']
            ground_truth_labels = f['ground_truth_labels']

            # Step 1: Identify noise points and handle them
            # We do NOT remove noise points now. Instead, we fill them in.
            hdbscan_labels = self.fill_noise_with_nearest_label(hdbscan_labels)

            # # Step 2: Convert silence from 0 to -1 in ground_truth_labels
            # ground_truth_labels[ground_truth_labels == 0] = -1

            # # Step 3: Apply majority voting to smooth HDBSCAN labels (optional)
            # hdbscan_labels = self.majority_vote(hdbscan_labels)
            
            # Step 4: Convert ground_truth_labels to phrase-level labels with silence = -1
            ground_truth_labels = self.syllable_to_phrase_labels(arr=ground_truth_labels, silence=0)

            # Compute metrics
            homogeneity = homogeneity_score(ground_truth_labels, hdbscan_labels)
            completeness = completeness_score(ground_truth_labels, hdbscan_labels)
            v_measure = v_measure_score(ground_truth_labels, hdbscan_labels)

            homogeneity_scores.append(homogeneity)
            completeness_scores.append(completeness)
            v_measure_scores.append(v_measure)

        # Calculate mean and standard error (SEM = std / sqrt(n))
        n = len(homogeneity_scores)
        metrics = {
            'Homogeneity': (np.mean(homogeneity_scores), np.std(homogeneity_scores, ddof=1) / np.sqrt(n)),
            'Completeness': (np.mean(completeness_scores), np.std(completeness_scores, ddof=1) / np.sqrt(n)),
            'V-measure': (np.mean(v_measure_scores), np.std(v_measure_scores, ddof=1) / np.sqrt(n))
        }

        return metrics
