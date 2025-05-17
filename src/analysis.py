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
from scipy.signal import fftconvolve      



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


def load_data( data_dir, context=1000):
   dataset = SongDataSet_Image(data_dir, num_classes=50, infinite_loader = False, pitch_shift= False, segment_length = None)
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


def generate_ghmm_labels(array):
    from hmmlearn import hmm

    print(array.shape)

    # normalize the array
    array = (array - np.mean(array)) / np.std(array)

    # define GHMM w/ 2 hidden states
    model = hmm.GaussianHMM(n_components=20, covariance_type='full')
    # train (EM)
    model.fit(array)

    # predict hidden states
    states = model.predict(array)

    print("means:", model.means_)
    print("covariances:", model.covars_)

    print(states.shape)
    return states

def _gaussian_kernel(radius: int, peak: float, sigma_divisor: float = 3.0):
    """
    build a symmetric 1-D gaussian kernel with height = `peak`.

    radius :   how many neighbours on each side (e.g. 1000)
    peak   :   kernel[0] after scaling (e.g. 0.05 or 0.1)
    """
    x     = np.arange(-radius, radius + 1)
    sigma = radius / sigma_divisor
    k     = np.exp(-0.5 * (x / sigma) ** 2)
    k    *= peak / k[radius]          # scale so centre == peak
    return k.astype(np.float32)       # save RAM

def smooth_predictions(pred, *, radius=1000, peak=0.05, include_self=True):
    """
    pred : (T, F) ndarray      (T ≈ 1e6, F ≈ 196)
    returns smoothed + residual-added predictions
    """
    kernel = _gaussian_kernel(radius, peak)
    if not include_self:
        kernel[radius] = 0.0          # don't double-count self

    # fft-based convolution along time axis for every feature at once
    if fftconvolve is not None:
        smoothed = fftconvolve(pred, kernel[:, None], mode='same', axes=0)
    else:                             # fallback: plain np.convolve (slower)
        smoothed = np.empty_like(pred)
        for f in range(pred.shape[1]):
            smoothed[:, f] = np.convolve(pred[:, f], kernel, mode='same')

    return pred + smoothed

def plot_umap_projection(model, device, data_dirs, category_colors_file="test_llb16", samples=1e6, file_path='category_colors.pkl',
                    layer_index=None, dict_key=None,
                    context=1000, save_name=None, raw_spectogram=False, save_dict_for_analysis=True,
                    remove_non_vocalization=True, min_cluster_size=500, state_finding_algorithm="HDBSCAN", psuedo_labels=True):
    """
    parameters:
    - data_dirs: list of data directories to analyze
    """
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = []
    vocalization_arr = []
    file_indices_arr = []
    dataset_indices_arr = []  # initialize this array
    file_map = {}
    current_file_index = 0

    # reset figure
    plt.figure(figsize=(8, 6))

    samples = int(samples)
    total_samples = 0

    # calculate samples per dataset
    samples_per_dataset = samples // len(data_dirs)
    print(f"samples per dataset: {samples_per_dataset}")

    # iterate through each dataset
    for dataset_idx, data_dir in enumerate(data_dirs):
        data_loader = load_data(data_dir=data_dir, context=context)
        data_loader_iter = iter(data_loader)
        dataset_samples = 0


        while dataset_samples < samples_per_dataset:
            try:
                # retrieve the next batch
                data, ground_truth_label, vocalization, file_path = next(data_loader_iter)

                # temporary fix for corrupted data
                if data.shape[1] < 10:
                    continue

                num_classes = ground_truth_label.shape[-1]
                original_data_length = data.shape[1]
                original_label_length = ground_truth_label.shape[1]


                # pad data to the nearest context size in the time dimension
                total_time = data.shape[1]
                pad_size_time = (context - (total_time % context)) % context
                data = F.pad(data, (0, 0, 0, pad_size_time), 'constant', 0)


                # calculate the number of context windows in the song
                num_times = data.shape[1] // context


                batch, time_bins, freq = data.shape


                # reshape data to fit into multiple context-sized batches
                data = data.reshape(batch * num_times, context, freq)


                # pad ground truth labels to match data padding in time dimension
                total_length_labels = ground_truth_label.shape[1]
                pad_size_labels_time = (context - (total_length_labels % context)) % context
                ground_truth_label = F.pad(ground_truth_label, (0, 0, 0, pad_size_labels_time), 'constant', 0)
                vocalization = F.pad(vocalization, (0, pad_size_labels_time), 'constant', 0)


                ground_truth_label = ground_truth_label.reshape(batch * num_times, context, num_classes)
                vocalization = vocalization.reshape(batch * num_times, context)


                # store file information
                if file_path not in file_map:
                    file_map[current_file_index] = file_path
                else:
                    continue


                if not raw_spectogram:
                    data = data.unsqueeze(1)
                    data = data.permute(0, 1, 3, 2)


                    _, layers = model.inference_forward(data.to(device))


                    layer_output_dict = layers[layer_index]
                    output = layer_output_dict.get(dict_key, None)


                    if output is None:
                        print(f"invalid key: {dict_key}. skipping this batch.")
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
                vocalization = vocalization[:original_data_length]


                ground_truth_label = torch.argmax(ground_truth_label, dim=-1)


                # create file_indices first
                file_indices = np.array(ground_truth_label.cpu().numpy())
                file_indices[:] = current_file_index


                # now create dataset_indices using the shape of file_indices
                dataset_indices = np.full_like(file_indices, dataset_idx)


                spec_arr.append(spec.cpu().numpy())
                ground_truth_labels_arr.append(ground_truth_label.cpu().numpy())
                vocalization_arr.append(vocalization.cpu().numpy())
                file_indices_arr.append(file_indices)
                dataset_indices_arr.append(dataset_indices)


                total_samples += spec.shape[0]
                current_file_index += 1


                if np.sum(np.concatenate(vocalization_arr, axis=0)) > samples:
                    break


                dataset_samples += spec.shape[0]


            except StopIteration:
                print(f"dataset {data_dir} exhausted after {dataset_samples} samples")
                break

    # convert the list of batch * samples * features to samples * features
    ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
    spec_arr = np.concatenate(spec_arr, axis=0)
    file_indices = np.concatenate(file_indices_arr, axis=0)
    dataset_indices = np.concatenate(dataset_indices_arr, axis=0)
    vocalization_arr = np.concatenate(vocalization_arr, axis=0)
    original_spec_arr = spec_arr

    if not raw_spectogram:
        predictions = np.concatenate(predictions_arr, axis=0)
    else:
        predictions = spec_arr

    # filter for vocalization before any processing or visualization
    if remove_non_vocalization:
        print(f"vocalization arr shape {vocalization_arr.shape}")
        print(f"predictions arr shape {predictions.shape}")
        print(f"ground truth labels arr shape {ground_truth_labels.shape}")
        print(f"spec arr shape {spec_arr.shape}")

        # Check for NaN or Inf values
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: NaN or Inf values found in predictions. Removing problematic samples...")
            # Remove samples with NaN or Inf values
            valid_indices = ~(np.isnan(predictions).any(axis=1) | np.isinf(predictions).any(axis=1))
            predictions = predictions[valid_indices]
            ground_truth_labels = ground_truth_labels[valid_indices]
            spec_arr = spec_arr[valid_indices]
            file_indices = file_indices[valid_indices]
            dataset_indices = dataset_indices[valid_indices]
            vocalization_arr = vocalization_arr[valid_indices]

        vocalization_indices = np.where(vocalization_arr == 1)[0]
        
        # Make sure vocalization_indices is not empty
        if len(vocalization_indices) == 0:
            print("Error: No vocalization indices found. Cannot proceed with UMAP.")
            return
            
        predictions = predictions[vocalization_indices]
        ground_truth_labels = ground_truth_labels[vocalization_indices]
        spec_arr = spec_arr[vocalization_indices]
        file_indices = file_indices[vocalization_indices]
        dataset_indices = dataset_indices[vocalization_indices]
    # fit the umap reducer with more conservative parameters
    print("initializing umap reducer...")
    # Try with more conservative parameters
    reducer = umap.UMAP(
        n_neighbors=200,  
        min_dist=0.1,  
        n_components=2,
        metric='cosine', 
        random_state=42 
    )
    print("umap reducer initialized.")

    # Convert to float32 to reduce memory usage
    predictions_float32 = predictions.astype(np.float32)

    # Check for and handle any remaining NaN values
    if np.isnan(predictions_float32).any():
        print("Warning: NaN values found. Replacing with zeros.")
        predictions_float32 = np.nan_to_num(predictions_float32)

    embedding_outputs = reducer.fit_transform(predictions_float32)
    print("umap fitting complete. shape of embedding outputs:", embedding_outputs.shape)

    if state_finding_algorithm == "HDBSCAN":
        print("generating hdbscan labels...")
        try:
            hdbscan_labels = generate_hdbscan_labels(embedding_outputs, min_samples=1, min_cluster_size=5000)
            print("hdbscan labels generated. unique labels found:", np.unique(hdbscan_labels))
        except Exception as e:
            print(f"HDBSCAN error: {e}")
            return

    # get unique labels and create color palettes
    unique_clusters = np.unique(hdbscan_labels)
    unique_ground_truth_labels = np.unique(ground_truth_labels)

    def create_color_palette(n_colors):
        """create a color palette with the specified number of colors"""
        colors = glasbey.create_palette(palette_size=n_colors)
        
        def hex_to_rgb(hex_str):
            # remove '#' if present
            hex_str = hex_str.lstrip('#')
            # convert hex to rgb
            return tuple(int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))
        
        # convert hex colors to rgb tuples
        rgb_colors = [hex_to_rgb(color) for color in colors]
        return rgb_colors


    # create color palettes
    n_ground_truth_labels = len(unique_ground_truth_labels)
    n_hdbscan_clusters = len(unique_clusters)
    ground_truth_colors = create_color_palette(n_ground_truth_labels)
    hdbscan_colors = create_color_palette(n_hdbscan_clusters)

    # make the first color in the hdbscan palette black
    hdbscan_colors[0] = (0, 0, 0)


    # create experiment-specific directory for images
    experiment_dir = os.path.join("imgs", "umap_plots", save_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)


    # plot hdbscan labels
    fig1, ax1 = plt.subplots(figsize=(16, 16), facecolor='white')
    ax1.set_facecolor('white')
    ax1.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1],
                c=hdbscan_labels, s=10, alpha=0.1,
                cmap=mcolors.ListedColormap(hdbscan_colors))
    ax1.set_title("hdbscan discovered labels", fontsize=48)
    ax1.set_xlabel("umap dimension 1", fontsize=48)
    ax1.set_ylabel("umap dimension 2", fontsize=48)
    ax1.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "hdbscan_labels.png"),
                facecolor=fig1.get_facecolor(), edgecolor='none')
    plt.close(fig1)


    # plot ground truth labels
    fig2, ax2 = plt.subplots(figsize=(16, 16), facecolor='white')
    ax2.set_facecolor('white')
    ax2.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1],
                c=ground_truth_labels, s=10, alpha=0.1,
                cmap=mcolors.ListedColormap(ground_truth_colors))
    ax2.set_title("ground truth labels", fontsize=48)
    ax2.set_xlabel("umap dimension 1", fontsize=48)
    ax2.set_ylabel("umap dimension 2", fontsize=48)
    ax2.tick_params(axis='both', which='both', bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "ground_truth_labels.png"),
                facecolor=fig2.get_facecolor(), edgecolor='none')
    plt.close(fig2)

    # ───────────────────────────── second-pass: balanced core  ─────────────────────────────
    # build a core with equal samples per pseudo-label cluster (skip noise = −1)
    pseudo_mask            = hdbscan_labels != -1
    clusters, counts       = np.unique(hdbscan_labels[pseudo_mask], return_counts=True)
    if len(clusters) >= 2:                                  # ensure at least 2 clusters to balance
        quota              = counts.min()
        rng                = np.random.default_rng(0)
        core_idx           = np.hstack([
            rng.choice(np.where(hdbscan_labels == c)[0], size=quota, replace=False)
            for c in clusters
        ])
        # Instead of UMAPing the predictions again, UMAP the original UMAP embedding (embedding_outputs)
        X_core             = embedding_outputs[core_idx]

        reducer_pass_2 = umap.UMAP(
            n_neighbors=200,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",  # UMAP on UMAP, so use euclidean
        ).fit(X_core)

        embedding_outputs_pass_2 = reducer_pass_2.transform(embedding_outputs)
        print("balanced umap complete. shape:", embedding_outputs_pass_2.shape)

        # run HDBSCAN again on the balanced embedding
        print("generating hdbscan labels (pass-2)…")
        try:
            hdbscan_labels_2 = generate_hdbscan_labels(
                embedding_outputs_pass_2,
                min_samples=1,
                min_cluster_size=5000
            )
            print("hdbscan-2 labels generated. unique:", np.unique(hdbscan_labels_2))
        except Exception as e:
            print(f"HDBSCAN-2 error: {e}")
            hdbscan_labels_2 = np.full(len(embedding_outputs_pass_2), -1)

        # colour map for pass-2 (size may differ)
        unique_clusters_2  = np.unique(hdbscan_labels_2)
        hdbscan_colors_2   = create_color_palette(len(unique_clusters_2))
        hdbscan_colors_2[0] = (0, 0, 0)                       # make noise black

        # plots for second pass
        fig3, ax3 = plt.subplots(figsize=(16, 16), facecolor='white')
        ax3.set_facecolor('white')
        ax3.scatter(embedding_outputs_pass_2[:, 0], embedding_outputs_pass_2[:, 1],
                    c=hdbscan_labels_2, s=10, alpha=0.1,
                    cmap=mcolors.ListedColormap(hdbscan_colors_2))
        ax3.set_title("hdbscan labels (balanced umap)", fontsize=48)
        ax3.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "hdbscan_labels_bal.png"),
                    facecolor=fig3.get_facecolor(), edgecolor='none')
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(16, 16), facecolor='white')
        ax4.set_facecolor('white')
        ax4.scatter(embedding_outputs_pass_2[:, 0], embedding_outputs_pass_2[:, 1],
                    c=ground_truth_labels, s=10, alpha=0.1,
                    cmap=mcolors.ListedColormap(ground_truth_colors))
        ax4.set_title("ground truth (balanced umap)", fontsize=48)
        ax4.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "ground_truth_bal.png"),
                    facecolor=fig4.get_facecolor(), edgecolor='none')
        plt.close(fig4)
    else:
        print("balancing skipped: need ≥2 clusters after first HDBSCAN.")
        embedding_outputs_pass_2 = None
        hdbscan_labels_2        = None

    # save the data
    np.savez(
        f"files/{save_name}",
        embedding_outputs=embedding_outputs,
        embedding_outputs_pass_2=embedding_outputs_pass_2,
        hdbscan_labels=hdbscan_labels,
        hdbscan_labels_2=hdbscan_labels_2,
        ground_truth_labels=ground_truth_labels,
        predictions=predictions,
        s=spec_arr,
        hdbscan_colors=hdbscan_colors,
        hdbscan_colors_2=hdbscan_colors_2 if 'hdbscan_colors_2' in locals() else None,
        ground_truth_colors=ground_truth_colors,
        original_spectogram=original_spec_arr,
        vocalization=vocalization_arr,
        file_indices=file_indices,
        dataset_indices=dataset_indices,
        file_map=file_map
    )

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
           left_dist = (idx - left_idx) if left_idx >= 0 else np.inf
           right_dist = (right_idx - idx) if right_idx < len(labels) else np.inf

           # Assign based on nearest non-noise label
           if left_dist == np.inf and right_dist == np.inf:
               # No non-noise neighbors found, remain -1
               continue
           elif left_dist <= right_dist:
               labels[idx] = labels[left_idx]
           else:
               labels[idx] = labels[right_idx]

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





