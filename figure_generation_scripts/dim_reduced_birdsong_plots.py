#!/usr/bin/env python3

import torch
import os
import sys
import argparse

figure_generation_dir = os.path.dirname(__file__)
project_root = os.path.dirname(figure_generation_dir)
os.chdir(project_root)

print(project_root)

sys.path.append("src")

from utils import load_model
from analysis import plot_umap_projection

# THIS SHOULD ALWAYS BE CPU SO YOU CAN FIT SUPER LONG SONGS IN MODEL UNLESS YOU HAVE A100 or BETTER!
device = torch.device("cpu")

def main(experiment_folder, data_dirs, category_colors_file, save_name, samples, layer_index, dict_key, context, raw_spectogram, save_dict_for_analysis, min_cluster_size):
    model = load_model(experiment_folder)
    model = model.to(device)

    plot_umap_projection(
        model=model, 
        device=device, 
        data_dirs=data_dirs,
        samples=samples, 
        category_colors_file=category_colors_file, 
        layer_index=layer_index, 
        dict_key=dict_key, 
        context=context, 
        raw_spectogram=raw_spectogram,
        save_dict_for_analysis=save_dict_for_analysis,
        save_name=save_name,
        min_cluster_size=min_cluster_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dimension-reduced birdsong plots.")
    parser.add_argument('--experiment_folder', type=str, default="experiments/default_experiment", help='Path to the experiment folder.')
    parser.add_argument('--data_dirs', nargs='+', default=["train_dir"], help='List of directories containing the data.')
    parser.add_argument('--category_colors_file', type=str, default="files/category_colors_llb3.pkl", help='Path to the category colors file.')
    parser.add_argument('--save_name', type=str, default="default_experiment_hdbscan", help='Name to save the output.')
    parser.add_argument('--samples', type=float, default=1e6, help='Number of samples to use.')
    parser.add_argument('--layer_index', type=int, default=-2, help='Layer index to use for UMAP projection.')
    parser.add_argument('--dict_key', type=str, default="attention_output", help='Dictionary key to use for UMAP projection.')
    parser.add_argument('--context', type=int, default=1000, help='Context size for the model.')
    parser.add_argument('--raw_spectogram', type=bool, default=False, help='Whether to use raw spectogram.')
    parser.add_argument('--save_dict_for_analysis', type=bool, default=True, help='Whether to save dictionary for analysis.')
    parser.add_argument('--min_cluster_size', type=int, default=500, help='Minimum cluster size for HDBSCAN.')

    args = parser.parse_args()
    main(
        args.experiment_folder, 
        args.data_dirs,
        args.category_colors_file, 
        args.save_name,
        args.samples,
        args.layer_index,
        args.dict_key,
        args.context,
        args.raw_spectogram,
        args.save_dict_for_analysis,
        args.min_cluster_size
    )
