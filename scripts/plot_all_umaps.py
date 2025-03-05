#!/usr/bin/env python3
import torch
import os
import sys
import json
from tqdm import tqdm
from torch import no_grad
import gc

# Add src directory to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append("src")

from utils import load_model, load_config
from data_class import determine_number_unique_classes
from analysis import plot_umap_projection
from utils import get_device

device = get_device()
print(f"Using device: {device}")

def plot_umap_for_layers(model, folder, config, category_colors_file, umap_data_dirs, umap_samples=1e6, min_cluster_size=500, skip_existing=True):
    """
    This function plots UMAP projections for each target layer and saves them.
    No FER analysis is performed here.
    """
    # Define the specific layers we want to analyze.
    target_layers = [
        ('attention_output', 'Attention output'),
        ('intermediate_residual_stream', 'Intermediate residual'),
        ('feed_forward_output_gelu', 'FF output after GELU'),
        ('feed_forward_output', 'Final FF output')
    ]

    # Filter model layers to only include our target layers
    layer_output_pairs = [
        (layer_id, layer_num, nn_dim) 
        for layer_id, layer_num, nn_dim in model.get_layer_output_pairs()
        if any(target[0] in layer_id for target in target_layers)
    ]

    for layer_id, layer_num, nn_dim in tqdm(layer_output_pairs, desc=f"Plotting UMAP for Layers in {folder}", leave=False):
        # Simple name for saving
        save_name = f"{folder}_layer{layer_num}_{layer_id}"

        print(f"Plotting UMAP for layer: {layer_id}, number: {layer_num}")

        try:
            with no_grad():
                plot_umap_projection(
                    model=model,
                    device=device,
                    data_dirs=umap_data_dirs,
                    samples=umap_samples,
                    category_colors_file=category_colors_file,
                    layer_index=layer_num,
                    dict_key=layer_id,
                    context=1000,
                    raw_spectogram=False,
                    save_dict_for_analysis=True,
                    save_name=save_name,
                    min_cluster_size=min_cluster_size
                )
        except Exception as e:
            print(f"Error processing layer {layer_id}, number {layer_num}: {str(e)}")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def execute_umap_plotting(experiment_configs, category_colors_file, umap_samples=1e6, min_cluster_size=5000, skip_existing=True):
    """
    Executes UMAP plotting across multiple experiments.
    """
    for exp_config in experiment_configs:
        base_path = os.path.join(project_root, exp_config['experiment_path'])

        if os.path.exists(base_path):
            try:
                model = load_model(base_path).to(device)
                config = load_config(os.path.join(base_path, 'config.json'))

                # Extract dataset name from train_dir for folder naming
                train_dir = os.path.join(project_root, exp_config['train_dir'])
                dataset_name = os.path.basename(train_dir).split('_')[0]

                # Use the actual data directories from the config
                umap_data_dirs = [train_dir]  # Add test_dir too if needed: [train_dir, os.path.join(project_root, exp_config['test_dir'])]

                plot_umap_for_layers(
                    model=model,
                    folder=dataset_name,
                    config=config,
                    category_colors_file=category_colors_file,
                    umap_data_dirs=umap_data_dirs,
                    umap_samples=umap_samples,
                    min_cluster_size=min_cluster_size,
                    skip_existing=skip_existing
                )
            except Exception as e:
                print(f"Error loading experiment '{base_path}': {str(e)}")
        else:
            print(f"Experiment path not found: {base_path}")


# Example usage
experiment_configs = [
    {
        "experiment_path": "experiments/TweetyBERT_Paper_Yarden_Model",
        "train_dir": "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb3_train",
        "test_dir": "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb3_test"
    },
        {
        "experiment_path": "experiments/TweetyBERT_Paper_Yarden_Model",
        "train_dir": "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb11_train",
        "test_dir": "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb11_test"
    },
        {
        "experiment_path": "experiments/TweetyBERT_Paper_Yarden_Model",
        "train_dir": "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb16_train",
        "test_dir": "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/llb16_test"
    },
]

category_colors_file = "files/category_colors_llb3.pkl"
umap_samples = 1e6
min_cluster_size = 5000

execute_umap_plotting(
    experiment_configs=experiment_configs,
    category_colors_file=category_colors_file,
    umap_samples=umap_samples,
    min_cluster_size=min_cluster_size,
    skip_existing=True
)
