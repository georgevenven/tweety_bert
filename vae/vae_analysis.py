import sys
import os

sys.path.append("src")
os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors
import re 
from collections import Counter
import umap
from data_class import SongDataSet_Image
from torch.utils.data import DataLoader
import glasbey
from sklearn.metrics.cluster import completeness_score
import seaborn as sns
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

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

def load_data( data_dir, context=1000, psuedo_labels_generated=True):
    dataset = SongDataSet_Image(data_dir, num_classes=196)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    return loader 

def generate_hdbscan_labels(array, min_samples=5, min_cluster_size=1000):
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

    print(np.unique(labels))

    return labels

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

def plot_umap_projection(model, device, data_dir="test_llb16",
                         remove_silences=False, samples=100, file_path='category_colors.pkl', 
                         layer_index=None, dict_key=None, time_bins_per_umap_point=100, 
                         context=1000, save_dir=None, raw_spectogram=False, save_dict_for_analysis=False, compute_svm=False, color_scheme="Syllable"):
    predictions_arr = []
    ground_truth_labels_arr = []
    spec_arr = [] 

    # Reset Figure
    plt.figure(figsize=(8, 6))

    # to allow sci notation 
    samples = int(samples)
    total_samples = 0

    data_loader = load_data(data_dir=data_dir, context=context, psuedo_labels_generated=True)
    data_loader_iter = iter(data_loader)

    while total_samples < samples:
        try:
            # Retrieve the next batch
            data, ground_truth_label = next(data_loader_iter)

            # if smaller than context window, go to next song
            if data.shape[1] < context:
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

            # output is the mu vector from latent space 
            _, output, _ = model.forward(data.to(device))

            print(output.shape)
            predictions_arr.append(output.detach().cpu().numpy())

        # remove channel dimension 
        data = data.squeeze(1)
        spec = data

        # set the features (freq axis to be the last dimension)
        spec = spec.permute(0, 2, 1)
        # combine batches and timebins
        spec = spec.flatten(0, 1)

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
    print(ground_truth_labels.shape)

    # Fit the UMAP reducer       
    reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')

    embedding_outputs = reducer.fit_transform(predictions)
    hdbscan_labels = generate_hdbscan_labels(embedding_outputs)


    # ground_truth_labels = syllable_to_phrase_labels(arr=ground_truth_labels,silence=0)
    np.savez("files/labels", embedding_outputs=embedding_outputs, hdbscan_labels=hdbscan_labels, ground_truth_labels=ground_truth_labels)

    cmap = glasbey.extend_palette(["#000000"], palette_size=30)
    cmap = mcolors.ListedColormap(cmap)    

    ground_truth_labels = average_colors_per_sample(ground_truth_labels, cmap)


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create a figure and a 1x2 grid of subplots

    axes[0].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=hdbscan_labels, s=10, alpha=.1, cmap=cmap)
    axes[0].set_title("HDBSCAN")

    # Plot with the original color scheme
    axes[1].scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], c=ground_truth_labels, s=10, alpha=.1, cmap=cmap)
    axes[1].set_title("Original Coloring")

    if raw_spectogram:
        plt.title(f'UMAP of Spectogram', fontsize=14)
    else:
        plt.title(f'UMAP Projection of (Layer: {layer_index}, Key: {dict_key})', fontsize=14)

    # Save the plot if save_dir is specified
    if save_dir:
        plt.savefig(save_dir, format='png')
    else:
        plt.show()

########################################################3
from vae import VariationalAutoencoder
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# File paths
config_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/VAE-Bungie-Rawer-Specs/config.json"
weight_path = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/experiments/VAE-Bungie-Rawer-Specs/saved_weights/model_step_2500.pth"

# Load config file
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
    # Initialize your model (ensure this matches the architecture expected by the weights)
    # model = VariationalAutoencoder(latent_dims=config["latent_dims"], variational_beta=config["variational_beta"], input_height=config["input_height"], input_width=config["input_width"])  # Add any required arguments for your model initialization here
    model = VariationalAutoencoder(latent_dims=config["latent_dims"], variational_beta=config["variational_beta"], input_height=config["input_height"], input_width=config["input_width"])  # Add any required arguments for your model initialization here
    model = model.to(device)

# Load weights
model.load_state_dict(torch.load(weight_path, map_location=device))

plot_umap_projection(
model=model, 
device=device, 
data_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/dev_train",
remove_silences=False,  # Using new config parameter``
samples=2e7, ## Excessive to collect all the songs in test set 
file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
layer_index="Mu Latent Vars", 
dict_key="VAE", 
time_bins_per_umap_point=1, 
context=100,  # Using new config parameter98
raw_spectogram=False,
save_dict_for_analysis = True,
save_dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/vae_umap_100_dev.png",
compute_svm= False,
color_scheme = "Label"
)