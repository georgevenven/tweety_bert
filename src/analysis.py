import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import re 
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

def syllable_to_phrase_labels(arr, silence=-1):
    new_arr = np.array(arr, dtype=int)
    current_syllable = None
    start_of_phrase_index = None
    first_non_silence_label = None  # To track the first non-silence syllable

    for i, value in enumerate(new_arr):
        if value != silence and value != current_syllable:
            if start_of_phrase_index is not None:
                new_arr[start_of_phrase_index:i] = current_syllable
            current_syllable = value
            start_of_phrase_index = i
            
            if first_non_silence_label is None:  # Found the first non-silence label
                first_non_silence_label = value

    if start_of_phrase_index is not None:
        new_arr[start_of_phrase_index:] = current_syllable

    # Replace the initial silence with the first non-silence syllable label
    if new_arr[0] == silence and first_non_silence_label is not None:
        for i in range(len(new_arr)):
            if new_arr[i] != silence:
                break
            new_arr[i] = first_non_silence_label

    return new_arr

def load_data( data_dir, context=1000):
    dataset = SongDataSet_Image(data_dir, num_classes=50, infinite_loader = False)
    # collate_fn = CollateFunction(segment_length=context)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)
    return loader 

def generate_hdbscan_labels(array, min_samples=1, min_cluster_size=5000):
    """
    Generate labels for data points using the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering algorithm.

    Parameters:
    - array: ndarray of shape (n_samples, n_features)
      The input data to cluster.

    - min_samples: int, default=5
      The number of samples in a neighborhood for a point to be considered as a core point.

    - min_cluster_size: int, default=5
      The minimum number of points required to form a cluster.

    Returns:
    - labels: ndarray of shape (n_samples)
      Cluster labels for each point in the dataset. Noisy samples are given the label -1.
    """

    import hdbscan

    # Create an HDBSCAN object with the specified parameters.
    hdbscan_model = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)

    # Fit the model to the data and extract the labels.
    labels = hdbscan_model.fit_predict(array)

    print(f"discovered labels {np.unique(labels)}")

    return labels

def plot_umap_projection(model, device, data_dirs, samples=100, category_colors_file=None, layer_index=None, dict_key=None, 
                         context=1000, save_name=None, raw_spectogram=False, remove_non_vocalization=True, plot_comparison=False,
                         save_dict_for_analysis=True, truncate_to_smallest_group=False):
    all_predictions = []
    all_ground_truth_labels = []
    all_specs = []
    all_vocalizations = []
    dataloader_indices = []
    sample_ids = []
    data_dir_mapping = []  # New list to store data directory for each sample

    # Reset Figure
    plt.figure(figsize=(8, 6))

    num_groups = len(data_dirs)

    if plot_comparison and num_groups > 1:
        plot_comparison = True
    else:
        plot_comparison = False

    # Load category colors if file is provided
    if category_colors_file:
        with open(category_colors_file, 'rb') as f:
            category_colors = pickle.load(f)
        # Convert numpy array to list of tuples if necessary
        if isinstance(category_colors, np.ndarray):
            if category_colors.ndim == 2:
                category_colors = [tuple(color) for color in category_colors]
            elif category_colors.ndim == 1:
                category_colors = [tuple(int(x) for x in category_colors)]
        else:
            category_colors = [tuple(int(x) for x in category_colors)]
    else:
        category_colors = None

    sample_id_counter = 0  # Initialize a counter for unique sample IDs

    for dataloader_idx, data_dir in enumerate(data_dirs):
        data_loader = load_data(data_dir=data_dir, context=context)
        data_loader_iter = iter(data_loader)
        predictions_arr = []
        ground_truth_labels_arr = []
        spec_arr = []
        vocalization_arr = []

        # to allow sci notation 
        samples = int(samples)
        total_samples = 0

        while total_samples < samples:
            try:
                # Retrieve the next batch
                data, ground_truth_label, vocalization = next(data_loader_iter)

                num_classes = ground_truth_label.shape[-1]
                original_data_length = data.shape[1]
                original_label_length = ground_truth_label.shape[1]

                # Pad data to the nearest context size in the time dimension
                total_time = data.shape[1]
                pad_size_time = (context - (total_time % context)) % context
                data = F.pad(data, (0, 0, 0, pad_size_time), 'constant', 0)

                # Calculate the number of context windows in the song
                num_times = data.shape[1] // context

                batch, time_bins, freq = data.shape

                # Reshape data to fit into multiple context-sized batches
                data = data.reshape(batch * num_times, context, freq)

                # Pad ground truth labels to match data padding in time dimension
                total_length_labels = ground_truth_label.shape[1]
                pad_size_labels_time = (context - (total_length_labels % context)) % context
                ground_truth_label = F.pad(ground_truth_label, (0, 0, 0, pad_size_labels_time), 'constant', 0)
                vocalization = F.pad(vocalization, (0, pad_size_labels_time), 'constant', 0)

                ground_truth_label = ground_truth_label.reshape(batch * num_times, context, num_classes)
                vocalization = vocalization.reshape(batch * num_times, context)

            except StopIteration:
                print(f"Dataloader {dataloader_idx}: samples collected {len(ground_truth_labels_arr) * context}")
                break

            if not raw_spectogram:
                data = data.unsqueeze(1)
                data = data.permute(0, 1, 3, 2)

                _, layers = model.inference_forward(data.to(device))

                layer_output_dict = layers[layer_index]
                output = layer_output_dict.get(dict_key, None)

                if output is None:
                    print(f"Invalid key: {dict_key}. Skipping this batch.")
                    continue

                batches, time_bins, features = output.shape
                predictions = output.reshape(batches, time_bins, features)
                predictions = predictions.flatten(0, 1)
                predictions = predictions[:original_data_length]

                predictions_arr.append(predictions.detach().cpu().numpy())
            else:
                data = data.unsqueeze(1)
                data = data.permute(0, 1, 3, 2)

            data = data.squeeze(1)
            spec = data
            spec = spec.permute(0, 2, 1)

            spec = spec.flatten(0, 1)
            spec = spec[:original_data_length]

            ground_truth_label = ground_truth_label.flatten(0, 1)
            vocalization = vocalization.flatten(0, 1)

            ground_truth_label = ground_truth_label[:original_label_length]

            ground_truth_label = torch.argmax(ground_truth_label, dim=-1)

            spec_arr.append(spec.cpu().numpy())
            ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
            vocalization_arr.append(vocalization.cpu().numpy())
            
            # Add sample IDs and data directory for this batch
            batch_sample_ids = [sample_id_counter] * vocalization.shape[0]
            sample_ids.extend(batch_sample_ids)
            data_dir_mapping.extend([data_dir] * vocalization.shape[0])
            sample_id_counter += 1

            total_samples += spec.shape[0]

        # Convert the list of batch * samples * features to samples * features 
        ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
        spec_arr = np.concatenate(spec_arr, axis=0)
        vocalization_arr = np.concatenate(vocalization_arr, axis=0)
        
        if not raw_spectogram:
            predictions = np.concatenate(predictions_arr, axis=0)
        else:
            predictions = spec_arr

        # Filter for vocalization before any processing or visualization
        if remove_non_vocalization:
            vocalization_indices = np.where(vocalization_arr == 1)[0]
            predictions = predictions[vocalization_indices]
            ground_truth_labels = ground_truth_labels[vocalization_indices]
            spec_arr = spec_arr[vocalization_indices]
            sample_ids = [sample_ids[i] for i in vocalization_indices]
            data_dir_mapping = [data_dir_mapping[i] for i in vocalization_indices]

        all_predictions.append(predictions)
        all_ground_truth_labels.append(ground_truth_labels)
        all_specs.append(spec_arr)
        all_vocalizations.append(vocalization_arr)
        dataloader_indices.extend([dataloader_idx] * len(predictions))

    # Truncate to smallest group if requested
    if truncate_to_smallest_group:
        min_length = min(len(pred) for pred in all_predictions)
        
        # Find the last complete sample in each group
        truncated_lengths = []
        for i in range(len(all_predictions)):
            sample_ids_group = sample_ids[sum(len(p) for p in all_predictions[:i]):sum(len(p) for p in all_predictions[:i+1])]
            
            # Find the last complete sample that fits within min_length
            last_complete_sample = None
            for id in sorted(set(sample_ids_group), reverse=True):
                if sample_ids_group.count(id) <= min_length:
                    last_complete_sample = id
                    break
            
            if last_complete_sample is None:
                # If no complete sample fits, use the first sample
                last_complete_sample = sample_ids_group[0]
            
            truncated_length = sum(1 for id in sample_ids_group if id <= last_complete_sample)
            truncated_lengths.append(truncated_length)
        
        # Use the smallest truncated length that includes complete samples
        min_truncated_length = min(truncated_lengths)
        
        # Truncate all data to this length
        all_predictions = [pred[:min_truncated_length] for pred in all_predictions]
        all_ground_truth_labels = [labels[:min_truncated_length] for labels in all_ground_truth_labels]
        all_specs = [spec[:min_truncated_length] for spec in all_specs]
        all_vocalizations = [voc[:min_truncated_length] for voc in all_vocalizations]
        
        # Adjust other lists
        total_length = min_truncated_length * len(data_dirs)
        dataloader_indices = dataloader_indices[:total_length]
        sample_ids = sample_ids[:total_length]
        data_dir_mapping = data_dir_mapping[:total_length]

    # Combine all data
    combined_predictions = np.concatenate(all_predictions, axis=0)
    combined_ground_truth_labels = np.concatenate(all_ground_truth_labels, axis=0)
    combined_specs = np.concatenate(all_specs, axis=0)
    dataloader_indices = np.array(dataloader_indices)

    print(f"Shape of combined array for UMAP: {combined_predictions.shape}")

    # Fit the UMAP reducer
    reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')
    reducer_cluster = umap.UMAP(n_neighbors=200, min_dist=0, n_components=6, metric='cosine')

    embedding_outputs = reducer.fit_transform(combined_predictions)
    embedding_outputs_cluster = reducer_cluster.fit_transform(combined_predictions)

    hdbscan_labels = generate_hdbscan_labels(embedding_outputs_cluster, min_samples=1, min_cluster_size=int(combined_predictions.shape[0]/200))

    # Create colormaps
    cmap_dataloaders = plt.cm.get_cmap('tab10')
    dataloader_colors = cmap_dataloaders(np.linspace(0, 1, len(data_dirs)))

    # add the color black as silences
    cmap_ground_truth = glasbey.extend_palette(["#000000"], palette_size=30)
    cmap_ground_truth = mcolors.ListedColormap(cmap_ground_truth)

    # Compute unique labels and their corresponding colors for ground truth labels
    unique_ground_truth_labels = np.unique(combined_ground_truth_labels)
    ground_truth_label_colors = {label: cmap_ground_truth.colors[label % len(cmap_ground_truth.colors)] for label in unique_ground_truth_labels}

    # Create a colormap for HDBSCAN labels
    cmap_hdbscan = glasbey.extend_palette(["#FFFFFF"], palette_size=30)
    cmap_hdbscan = mcolors.ListedColormap(cmap_hdbscan)
    hdbscan_colors = np.array([cmap_hdbscan.colors[label % len(cmap_hdbscan.colors)] for label in hdbscan_labels])

    # Plot 1: Comparison plot (if applicable)
    if plot_comparison:
        fig, ax = plt.subplots(figsize=(16, 16), edgecolor='black', linewidth=2)
        colors = ['red', 'blue']  # Set colors to red and blue
        for idx, (color, data_dir) in enumerate(zip(colors, data_dirs)):
            mask = dataloader_indices == idx
            scatter = ax.scatter(embedding_outputs[mask, 0], embedding_outputs[mask, 1], 
                                 c=color, s=70, alpha=0.1, label=data_dir)
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlabel('UMAP 1', fontsize=48)
        ax.set_ylabel('UMAP 2', fontsize=48)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        ax.set_title("UMAP Projection Comparison", fontsize=48)
        ax.legend(fontsize=24, markerscale=2, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.savefig(save_name + "_comparison.png", bbox_inches='tight')
        plt.close()

    # Plot 2: Ground Truth Labels
    fig_ground_truth, ax_ground_truth = plt.subplots(figsize=(16, 16), edgecolor='black', linewidth=2)
    scatter_ground_truth = ax_ground_truth.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], 
                                                   c=combined_ground_truth_labels, s=70, alpha=0.1, cmap=cmap_ground_truth)
    ax_ground_truth.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_ground_truth.set_xlabel('UMAP 1', fontsize=48)
    ax_ground_truth.set_ylabel('UMAP 2', fontsize=48)
    for spine in ax_ground_truth.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    ax_ground_truth.set_title("Ground Truth Labels", fontsize=48)
    plt.tight_layout()
    plt.savefig(save_name + "_ground_truth.png")
    plt.close()

    # Plot 3: HDBSCAN Labels
    fig_hdbscan, ax_hdbscan = plt.subplots(figsize=(16, 16), edgecolor='black', linewidth=2)
    scatter_hdbscan = ax_hdbscan.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], 
                                         c=hdbscan_labels, s=70, alpha=0.1, cmap=cmap_hdbscan)
    ax_hdbscan.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_hdbscan.set_xlabel('UMAP 1', fontsize=48)
    ax_hdbscan.set_ylabel('UMAP 2', fontsize=48)
    for spine in ax_hdbscan.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    ax_hdbscan.set_title("HDBSCAN Discovered Labels", fontsize=48)
    plt.tight_layout()
    plt.savefig(save_name + "_hdbscan.png")
    plt.close()

    # Save the data for further analysis
    if save_dict_for_analysis:
        np.savez(f"files/labels_{save_name}", 
                 embedding_outputs=embedding_outputs,
                 hdbscan_labels=hdbscan_labels,
                 ground_truth_labels=combined_ground_truth_labels,
                 specs=combined_specs,
                 hdbscan_colors=hdbscan_colors,
                 ground_truth_colors=cmap_ground_truth.colors,
                 dataloader_indices=dataloader_indices,
                 dataloader_colors=dataloader_colors,
                 sample_ids=np.array(sample_ids),
                 data_dir_mapping=np.array(data_dir_mapping))

    print(f"Plots saved as {save_name}_comparison.png, {save_name}_ground_truth.png, and {save_name}_hdbscan.png")
    if save_dict_for_analysis:
        print(f"Data saved as files/labels_{save_name}.npz")

def apply_windowing(arr, window_size, stride, flatten_predictions=False):
    """
    Apply windowing to the input array.

    Parameters:
    - arr: The input array to window, expected shape (num_samples, features) for predictions and (num_samples,) for labels.
    - window_size: The size of each window.
    - stride: The stride between windows.
    - flatten_predictions: A boolean indicating whether to flatten the windowed predictions.

    Returns:
    - windowed_arr: The windowed version of the input array.
    """
    num_samples, features = arr.shape if len(arr.shape) > 1 else (arr.shape[0], 1)
    num_windows = (num_samples - window_size) // stride + 1
    windowed_arr = np.lib.stride_tricks.as_strided(
        arr,
        shape=(num_windows, window_size, features),
        strides=(arr.strides[0] * stride, arr.strides[0], arr.strides[-1]),
        writeable=False
    )

    if flatten_predictions and features > 1:
        # Flatten each window for predictions
        windowed_arr = windowed_arr.reshape(num_windows, -1)
    
    return windowed_arr


def sliding_window_umap(model, device, data_dir="test_llb16",
                         remove_silences=False, samples=100, file_path='category_colors.pkl', 
                         layer_index=None, dict_key=None, time_bins_per_umap_point=100, 
                         context=1000, save_dir=None, raw_spectogram=False, save_dict_for_analysis=False, compute_svm=False, color_scheme="Syllable", window_size=100, stride=1):
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = [] 

    # Reset Figure
    plt.figure(figsize=(8, 6))

    # to allow sci notation 
    samples = int(samples)
    total_samples = 0

    data_loader = load_data(data_dir=data_dir, context=context)
    data_loader_iter = iter(data_loader)

    while total_samples < samples:
        try:
            # Retrieve the next batch
            data, ground_truth_label = next(data_loader_iter)
            
            # if smaller than context window, go to next song
            if data.shape[-2] < context:
                continue 

            # because network is made to work with batched data, we unsqueeze a dim and transpose the last two dims (usually handled by collate fn)
            data = data.unsqueeze(0).permute(0,1,3,2)

            # calculate the number of times a song 
            num_times = data.shape[-1] // context
            
            # removing left over timebins that do not fit in context window 
            shave_index = num_times * context
            data = data[:,:,:,:shave_index]

            batch, channel, freq, time_bins = data.shape 

            # cheeky reshaping operation to reshape the length of the song that is larger than the context window into multiple batches 
            data = data.permute(0,-1, 1, 2)
            data = data.reshape(num_times, context, channel, freq)
            data = data.permute(0,2,3,1)

            # reshaping g truth labels to be consistent 
            batch, time_bins, labels = ground_truth_label.shape

            # shave g truth labels 
            ground_truth_label = ground_truth_label.permute(0,2,1)
            ground_truth_label = ground_truth_label[:,:,:shave_index]

            # cheeky reshaping operation to reshape the length of the song that is larger than the context window into multiple batches 
            ground_truth_label = ground_truth_label.permute(0,2,1)
            ground_truth_label = ground_truth_label.reshape(num_times, context, labels)
            
        except StopIteration:
            # if test set is exhausted, print the number of samples collected and stop the collection process
            print(f"samples collected {len(ground_truth_labels_arr) * context}")
            break

        if raw_spectogram == False:
            _, layers = model.inference_forward(data.to(device))

            layer_output_dict = layers[layer_index]
            output = layer_output_dict.get(dict_key, None)

            if output is None:
                print(f"Invalid key: {dict_key}. Skipping this batch.")
                continue

            batches, time_bins, features = output.shape 
            # data shape [0] is the number of batches, 
            predictions = output.reshape(batches, time_bins, features)
            # combine the batches and number of samples per context window 
            predictions = predictions.flatten(0,1)
            predictions_arr.append(predictions.detach().cpu().numpy())

        # remove channel dimension 
        data = data.squeeze(1)
        spec = data

        # set the features (freq axis to be the last dimension)
        spec = spec.permute(0, 2, 1)
        # combine batches and timebins
        spec = spec.flatten(0, 1)

        ground_truth_label = ground_truth_label.flatten(0, 1)
        ground_truth_label = torch.argmax(ground_truth_label, dim=-1)

        spec_arr.append(spec.cpu().numpy())
        ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
        
        total_samples += spec.shape[0]

    # convert the list of batch * samples * features to samples * features 
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    spec_arr = np.concatenate(spec_arr, axis=0)

    if not raw_spectogram:
        predictions = np.concatenate(predictions_arr, axis=0)
    else:
        predictions = spec_arr

    # razor off any extra datapoints 
    if samples > len(predictions):
        samples = len(predictions)
    else:
        predictions = predictions[:samples]
        ground_truth_labels = ground_truth_labels[:samples]

    print(predictions.shape)

    # Ensure predictions are in the correct shape (num_samples, features) before windowing
    predictions = apply_windowing(predictions, window_size, stride=stride, flatten_predictions=True)
    ground_truth_labels = apply_windowing(ground_truth_labels.reshape(-1, 1), window_size, stride=stride, flatten_predictions=False)

    ground_truth_labels = ground_truth_labels.squeeze()
    
    # Fit the UMAP reducer       
    reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')

    embedding_outputs = reducer.fit_transform(predictions)
    hdbscan_labels = generate_hdbscan_labels(embedding_outputs)

    np.savez(f"save_dir", embedding_outputs=embedding_outputs, hdbscan_labels=hdbscan_labels, ground_truth_labels=ground_truth_labels, s=spec_arr)
    
    # Assuming 'glasbey' is a predefined object with a method 'extend_palette'
    cmap = glasbey.extend_palette(["#000000"], palette_size=30)
    cmap = mcolors.ListedColormap(cmap)    

    ground_truth_labels = average_colors_per_sample(ground_truth_labels, cmap)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

    axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=hdbscan_labels, s=10, alpha=.1, cmap=cmap)
    axes[0].set_title("HDBSCAN")

    # Plot with the original color scheme
    axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=ground_truth_labels, s=10, alpha=.1, cmap=cmap)
    axes[1].set_title("Original Coloring")

    # Adjust title based on 'raw_spectogram' flag
    if raw_spectogram:
        plt.title(f'UMAP of Spectogram', fontsize=14)
    else:
        plt.title(f'UMAP Projection of (Layer: {layer_index}, Key: {dict_key})', fontsize=14)

    # Save the plot if 'save_dir' is specified, otherwise display it
    if save_dir:
        plt.savefig(save_dir, format='png')
    else:
        plt.show()

class ComputerClusterPerformance():
    def __init__(self, labels_path):

        # takes a list of paths to files that contain the labels 
        self.labels_paths = labels_path
            
    def syllable_to_phrase_labels(self, arr, silence=-1):
        new_arr = np.array(arr, dtype=int)
        current_syllable = None
        start_of_phrase_index = None
        first_non_silence_label = None  # To track the first non-silence syllable

        for i, value in enumerate(new_arr):
            if value != silence and value != current_syllable:
                if start_of_phrase_index is not None:
                    new_arr[start_of_phrase_index:i] = current_syllable
                current_syllable = value
                start_of_phrase_index = i
                
                if first_non_silence_label is None:  # Found the first non-silence label
                    first_non_silence_label = value

        if start_of_phrase_index is not None:
            new_arr[start_of_phrase_index:] = current_syllable

        # Replace the initial silence with the first non-silence syllable label
        if new_arr[0] == silence and first_non_silence_label is not None:
            for i in range(len(new_arr)):
                if new_arr[i] != silence:
                    break
                new_arr[i] = first_non_silence_label

        return new_arr

    def reduce_phrases(self, arr, remove_silence=True):
        current_element = arr[0]
        reduced_list = [] 

        for i, value in enumerate(arr):
            if value != current_element:
                reduced_list.append(current_element)
                current_element = value 

            # append last phrase
            if i == len(arr) - 1:
                reduced_list.append(current_element)

        if remove_silence == True:
            reduced_list = [value for value in reduced_list if value != 0]

        return np.array(reduced_list)

    def majority_vote(self, data, window_size=1):
        """
        Apply majority vote on the input data with a specified window size.

        Parameters:
        - data: list or array-like
          The input data to apply majority vote on.
        - window_size: int, default=3
          The size of the window to apply majority vote. Must be an odd number.

        Returns:
        - output: ndarray
          The array with majority vote applied.
        """
        # Function to find the majority element in a window
        def find_majority(window):
            count = Counter(window)
            majority = max(count.values())
            for num, freq in count.items():
                if freq == majority:
                    return num
            return window[len(window) // 2]  # Return the middle element if no majority found

        # Ensure the input data is in list form
        if isinstance(data, str):
            data = [int(x) for x in data.split(',') if x.strip().isdigit()]

        # Initialize the output array with a padding at the beginning
        output = [data[0]] * (window_size // 2)  # Pad with the first element

        # Apply the majority vote on each window
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            output.append(find_majority(window))

        # Pad the output array at the end to match the input array size
        output.extend([data[-1]] * (window_size // 2))

        return np.array(output)

    def compute_vmeasure_score(self):
        homogeneity_scores = []
        completeness_scores = []
        v_measure_scores = []

        for path_index, path in enumerate(self.labels_paths):
            f = np.load(path)
            hdbscan_labels = f['hdbscan_labels']
            ground_truth_labels = f['ground_truth_labels']

            # Remove points marked as noise
            remove_noise_index = np.where(hdbscan_labels == -1)[0]
            hdbscan_labels = np.delete(hdbscan_labels, remove_noise_index)
            ground_truth_labels = np.delete(ground_truth_labels, remove_noise_index)

            # Convert to phrase labels
            hdbscan_labels = self.majority_vote(hdbscan_labels)
            ground_truth_labels = self.syllable_to_phrase_labels(arr=ground_truth_labels, silence=0)

            # Compute scores
            homogeneity = homogeneity_score(ground_truth_labels, hdbscan_labels)
            completeness = completeness_score(ground_truth_labels, hdbscan_labels)
            v_measure = v_measure_score(ground_truth_labels, hdbscan_labels)

            # Append scores
            homogeneity_scores.append(homogeneity)
            completeness_scores.append(completeness)
            v_measure_scores.append(v_measure)

        # Calculate average and standard error
        metrics = {
            'Homogeneity': (np.mean(homogeneity_scores), np.std(homogeneity_scores, ddof=1) / np.sqrt(len(homogeneity_scores))),
            'Completeness': (np.mean(completeness_scores), np.std(completeness_scores, ddof=1) / np.sqrt(len(completeness_scores))),
            'V-measure': (np.mean(v_measure_scores), np.std(v_measure_scores, ddof=1) / np.sqrt(len(v_measure_scores)))
        }

        return metrics 
                
    def compute_adjusted_rand_index(self):
        adjusted_rand_indices = []

        for path_index, path in enumerate(self.labels_paths):
            f = np.load(path)
            hdbscan_labels = f['hdbscan_labels']
            ground_truth_labels = f['ground_truth_labels']

            # Remove points marked as noise
            remove_noise_index = np.where(hdbscan_labels == -1)[0]
            hdbscan_labels = np.delete(hdbscan_labels, remove_noise_index)
            ground_truth_labels = np.delete(ground_truth_labels, remove_noise_index)

            # Convert to phrase labels
            hdbscan_labels = self.majority_vote(hdbscan_labels)
            ground_truth_labels = self.syllable_to_phrase_labels(arr=ground_truth_labels, silence=0)

            # Compute Adjusted Rand Index
            ari = adjusted_rand_score(ground_truth_labels, hdbscan_labels)

            # Append score
            adjusted_rand_indices.append(ari)

        # Calculate average and standard error
        metrics = {
            'Adjusted Rand Index': (np.mean(adjusted_rand_indices), np.std(adjusted_rand_indices, ddof=1) / np.sqrt(len(adjusted_rand_indices)))
        }

        return metrics

    def compute_hopkins_statistic(self, X):
        """
        Compute the Hopkins statistic for the dataset X.
        
        Parameters:
        - X: ndarray of shape (n_samples, n_features)
          The input data to compute the Hopkins statistic.

        Returns:
        - hopkins_stat: float
          The Hopkins statistic value.
        """

        print(X.shape)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        m = int(0.1 * n_samples)  # Sample size, typically 10% of the dataset

        # Randomly sample m points from the dataset
        random_indices = np.random.choice(n_samples, m, replace=False)
        X_m = X[random_indices]

        # Generate m random points within the feature space
        X_random = np.random.uniform(np.min(X, axis=0), np.max(sX, axis=0), (m, n_features))

        # Nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)

        # Compute distances from random points to the nearest neighbors in the dataset
        u_distances, _ = nbrs.kneighbors(X_random, n_neighbors=1)
        u_distances = u_distances.sum()

        # Compute distances from sampled points to the nearest neighbors in the dataset
        w_distances, _ = nbrs.kneighbors(X_m, n_neighbors=2)
        w_distances = w_distances[:, 1].sum()  # Exclude the point itself

        # Compute the Hopkins statistic
        hopkins_stat = u_distances / (u_distances + w_distances)

        return hopkins_stat

    def compute_hopkins_statistic_from_file(self, file_path):
        """
        Compute the Hopkins statistic for the embedding data stored in the given file.

        Parameters:
        - file_path: str
          Path to the .npz file containing the embedding data.

        Returns:
        - hopkins_stat: float
          The Hopkins statistic value.
        """
        data = np.load(file_path)
        embedding_outputs = data['embedding_outputs']
        return self.compute_hopkins_statistic(embedding_outputs)

def plot_umap_projection(model, device, data_dirs, samples=100, category_colors_file=None, layer_index=None, dict_key=None, 
                         context=1000, save_name=None, raw_spectogram=False, remove_non_vocalization=True, plot_comparison=False,
                         save_dict_for_analysis=False, truncate_to_smallest_group=False):
    all_predictions = []
    all_ground_truth_labels = []
    all_specs = []
    all_vocalizations = []
    dataloader_indices = []
    sample_ids = []
    data_dir_mapping = []  # New list to store data directory for each sample

    # Reset Figure
    plt.figure(figsize=(8, 6))

    num_groups = len(data_dirs)

    if plot_comparison and num_groups > 1:
        plot_comparison = True
    else:
        plot_comparison = False

    sample_id_counter = 0  # Initialize a counter for unique sample IDs

    for dataloader_idx, data_dir in enumerate(data_dirs):
        data_loader = load_data(data_dir=data_dir, context=context)
        data_loader_iter = iter(data_loader)
        predictions_arr = []
        ground_truth_labels_arr = []
        spec_arr = []
        vocalization_arr = []

        # to allow sci notation 
        samples = int(samples)
        total_samples = 0

        while total_samples < samples:
            try:
                # Retrieve the next batch
                data, ground_truth_label, vocalization = next(data_loader_iter)

                num_classes = ground_truth_label.shape[-1]
                original_data_length = data.shape[1]
                original_label_length = ground_truth_label.shape[1]

                # Pad data to the nearest context size in the time dimension
                total_time = data.shape[1]  # Adjusted for batch first
                pad_size_time = (context - (total_time % context)) % context  # Adjusted padding calculation for time dimension
                data = F.pad(data, (0, 0, 0, pad_size_time), 'constant', 0)  # Adjusted padding for batch first in time dimension

                # Calculate the number of context windows in the song
                num_times = data.shape[1] // context  # Adjustment needed here as we're padding time

                batch, time_bins, freq = data.shape  # Adjusted for batch first

                # Reshape data to fit into multiple context-sized batches
                data = data.reshape(batch * num_times, context, freq)  # Adjusted reshape to exclude padded frequency since we're padding time

                # Pad ground truth labels to match data padding in time dimension
                total_length_labels = ground_truth_label.shape[1]  # Adjusted for batch first
                pad_size_labels_time = (context - (total_length_labels % context)) % context  # Adjusted padding calculation for time dimension
                ground_truth_label = F.pad(ground_truth_label, (0, 0, 0, pad_size_labels_time), 'constant', 0)  # Adjusted padding for batch first in time dimension
                vocalization = F.pad(vocalization, (0, pad_size_labels_time), 'constant', 0)

                ground_truth_label = ground_truth_label.reshape(batch * num_times, context, num_classes)  # Adjusted for batch first
                vocalization = vocalization.reshape(batch * num_times, context)

            except StopIteration:
                print(f"Dataloader {dataloader_idx}: samples collected {len(ground_truth_labels_arr) * context}")
                break

            if raw_spectogram == False:
                data = data.unsqueeze(1)
                data = data.permute(0, 1, 3, 2)

                _, layers = model.inference_forward(data.to(device))

                layer_output_dict = layers[layer_index]
                output = layer_output_dict.get(dict_key, None)

                if output is None:
                    print(f"Invalid key: {dict_key}. Skipping this batch.")
                    continue

                batches, time_bins, features = output.shape
                predictions = output.reshape(batches, time_bins, features)
                predictions = predictions.flatten(0, 1)
                predictions = predictions[:original_data_length]

                predictions_arr.append(predictions.detach().cpu().numpy())
            else:
                data = data.unsqueeze(1)
                data = data.permute(0, 1, 3, 2)

            data = data.squeeze(1)
            spec = data
            spec = spec.permute(0, 2, 1)

            spec = spec.flatten(0, 1)
            spec = spec[:original_data_length]  # Remove padding from spec

            ground_truth_label = ground_truth_label.flatten(0, 1)
            vocalization = vocalization.flatten(0, 1)

            ground_truth_label = ground_truth_label[:original_label_length]  # Remove padding from labels

            ground_truth_label = torch.argmax(ground_truth_label, dim=-1)

            spec_arr.append(spec.cpu().numpy())
            ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
            vocalization_arr.append(vocalization.cpu().numpy())

            # Add sample IDs and data directory for this batch
            batch_sample_ids = [sample_id_counter] * vocalization.shape[0]
            sample_ids.extend(batch_sample_ids)
            data_dir_mapping.extend([data_dir] * vocalization.shape[0])
            sample_id_counter += 1

            total_samples += spec.shape[0]

        # Convert the list of batch * samples * features to samples * features 
        ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
        spec_arr = np.concatenate(spec_arr, axis=0)
        vocalization_arr = np.concatenate(vocalization_arr, axis=0)
        
        if not raw_spectogram:
            predictions = np.concatenate(predictions_arr, axis=0)
        else:
            predictions = spec_arr

        # Filter for vocalization before any processing or visualization
        if remove_non_vocalization:
            vocalization_indices = np.where(vocalization_arr == 1)[0]
            predictions = predictions[vocalization_indices]
            ground_truth_labels = ground_truth_labels[vocalization_indices]
            spec_arr = spec_arr[vocalization_indices]
            sample_ids = [sample_ids[i] for i in vocalization_indices]
            data_dir_mapping = [data_dir_mapping[i] for i in vocalization_indices]

        all_predictions.append(predictions)
        all_ground_truth_labels.append(ground_truth_labels)
        all_specs.append(spec_arr)
        all_vocalizations.append(vocalization_arr)
        dataloader_indices.extend([dataloader_idx] * len(predictions))

    # Truncate to smallest group if requested
    if truncate_to_smallest_group:
        min_length = min(len(pred) for pred in all_predictions)
        
        # Find the last complete sample in each group
        truncated_lengths = []
        for i in range(len(all_predictions)):
            sample_ids_group = sample_ids[sum(len(p) for p in all_predictions[:i]):sum(len(p) for p in all_predictions[:i+1])]
            
            # Find the last complete sample that fits within min_length
            last_complete_sample = None
            for id in sorted(set(sample_ids_group), reverse=True):
                if sample_ids_group.count(id) <= min_length:
                    last_complete_sample = id
                    break
            
            if last_complete_sample is None:
                # If no complete sample fits, use the first sample
                last_complete_sample = sample_ids_group[0]
            
            truncated_length = sum(1 for id in sample_ids_group if id <= last_complete_sample)
            truncated_lengths.append(truncated_length)
        
        # Use the smallest truncated length that includes complete samples
        min_truncated_length = min(truncated_lengths)
        
        # Truncate all data to this length
        all_predictions = [pred[:min_truncated_length] for pred in all_predictions]
        all_ground_truth_labels = [labels[:min_truncated_length] for labels in all_ground_truth_labels]
        all_specs = [spec[:min_truncated_length] for spec in all_specs]
        all_vocalizations = [voc[:min_truncated_length] for voc in all_vocalizations]
        
        # Adjust other lists
        total_length = min_truncated_length * len(data_dirs)
        dataloader_indices = dataloader_indices[:total_length]
        sample_ids = sample_ids[:total_length]
        data_dir_mapping = data_dir_mapping[:total_length]

    # Combine all data
    combined_predictions = np.concatenate(all_predictions, axis=0)
    combined_ground_truth_labels = np.concatenate(all_ground_truth_labels, axis=0)
    combined_specs = np.concatenate(all_specs, axis=0)
    dataloader_indices = np.array(dataloader_indices)

    print(f"Shape of combined array for UMAP: {combined_predictions.shape}")

    # Fit the UMAP reducer
    reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')
    reducer_cluster = umap.UMAP(n_neighbors=200, min_dist=0, n_components=6, metric='cosine')

    embedding_outputs = reducer.fit_transform(combined_predictions)
    embedding_outputs_cluster = reducer_cluster.fit_transform(combined_predictions)

    hdbscan_labels = generate_hdbscan_labels(embedding_outputs_cluster, min_samples=1, min_cluster_size=int(combined_predictions.shape[0]/200))

    # Create colormaps
    cmap_dataloaders = plt.cm.get_cmap('tab10')
    dataloader_colors = cmap_dataloaders(np.linspace(0, 1, len(data_dirs)))

    # add the color black as silences
    cmap_ground_truth = glasbey.extend_palette(["#000000"], palette_size=30)
    cmap_ground_truth = mcolors.ListedColormap(cmap_ground_truth)

    # Compute unique labels and their corresponding colors for ground truth labels
    unique_ground_truth_labels = np.unique(combined_ground_truth_labels)
    ground_truth_label_colors = {label: cmap_ground_truth.colors[label % len(cmap_ground_truth.colors)] for label in unique_ground_truth_labels}

    # Create a colormap for HDBSCAN labels
    cmap_hdbscan = glasbey.extend_palette(["#FFFFFF"], palette_size=30)
    cmap_hdbscan = mcolors.ListedColormap(cmap_hdbscan)
    hdbscan_colors = np.array([cmap_hdbscan.colors[label % len(cmap_hdbscan.colors)] for label in hdbscan_labels])

    # Plot 1: Comparison plot (if applicable)
    if plot_comparison:
        fig, ax = plt.subplots(figsize=(16, 16), edgecolor='black', linewidth=2)
        colors = ['red', 'blue']  # Set colors to red and blue
        for idx, (color, data_dir) in enumerate(zip(colors, data_dirs)):
            mask = dataloader_indices == idx
            scatter = ax.scatter(embedding_outputs[mask, 0], embedding_outputs[mask, 1], 
                                 c=color, s=70, alpha=0.1, label=data_dir)
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlabel('UMAP 1', fontsize=48)
        ax.set_ylabel('UMAP 2', fontsize=48)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)
        ax.set_title("UMAP Projection Comparison", fontsize=48)
        ax.legend(fontsize=24, markerscale=2, loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.tight_layout()
        plt.savefig(save_name + "_comparison.png", bbox_inches='tight')
        plt.close()

    # Plot 2: Ground Truth Labels
    fig_ground_truth, ax_ground_truth = plt.subplots(figsize=(16, 16), edgecolor='black', linewidth=2)
    scatter_ground_truth = ax_ground_truth.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], 
                                                   c=combined_ground_truth_labels, s=70, alpha=0.1, cmap=cmap_ground_truth)
    ax_ground_truth.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_ground_truth.set_xlabel('UMAP 1', fontsize=48)
    ax_ground_truth.set_ylabel('UMAP 2', fontsize=48)
    for spine in ax_ground_truth.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    ax_ground_truth.set_title("Ground Truth Labels", fontsize=48)
    plt.tight_layout()
    plt.savefig(save_name + "_ground_truth.png")
    plt.close()

    # Plot 3: HDBSCAN Labels
    fig_hdbscan, ax_hdbscan = plt.subplots(figsize=(16, 16), edgecolor='black', linewidth=2)
    scatter_hdbscan = ax_hdbscan.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], 
                                         c=hdbscan_labels, s=70, alpha=0.1, cmap=cmap_hdbscan)
    ax_hdbscan.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_hdbscan.set_xlabel('UMAP 1', fontsize=48)
    ax_hdbscan.set_ylabel('UMAP 2', fontsize=48)
    for spine in ax_hdbscan.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    ax_hdbscan.set_title("HDBSCAN Discovered Labels", fontsize=48)
    plt.tight_layout()
    plt.savefig(save_name + "_hdbscan.png")
    plt.close()

    # Save the data for further analysis
    if save_dict_for_analysis:
        np.savez(f"files/labels_{save_name}", 
                 embedding_outputs=embedding_outputs,
                 hdbscan_labels=hdbscan_labels,
                 ground_truth_labels=combined_ground_truth_labels,
                 specs=combined_specs,
                 hdbscan_colors=hdbscan_colors,
                 ground_truth_colors=cmap_ground_truth.colors,
                 dataloader_indices=dataloader_indices,
                 dataloader_colors=dataloader_colors,
                 sample_ids=np.array(sample_ids),
                 data_dir_mapping=np.array(data_dir_mapping))

    print(f"Plots saved as {save_name}_comparison.png, {save_name}_ground_truth.png, and {save_name}_hdbscan.png")
    if save_dict_for_analysis:
        print(f"Data saved as files/labels_{save_name}.npz")
