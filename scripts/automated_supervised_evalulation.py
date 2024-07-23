import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import re
from torch.utils.data import DataLoader
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

sys.path.append("src")

from linear_probe import LinearProbeModel, LinearProbeTrainer,ModelEvaluator
from utils import load_model, detailed_count_parameters, load_config
from data_class import SongDataSet_Image, CollateFunction
from src.analysis import plot_umap_projection
from data_class import SongDataSet_Image

def load_config_from_path(path):
    with open(path, 'r') as f:
        config = json.load(f)  # Load and parse the JSON file
    return config 

# Change to relative paths for train and test directories
train_dir = "files/llb3_train"
test_dir = "files/llb3_test"
# Print changes
train_dir = os.path.join(project_root, "files/llb3_train")
test_dir = os.path.join(project_root, "files/llb3_test")


train_dataset = SongDataSet_Image(train_dir, num_classes=21, psuedo_labels_generated=False)
test_dataset = SongDataSet_Image(test_dir, num_classes=21, psuedo_labels_generated=False)

collate_fn = CollateFunction(segment_length=1000)  # Adjust the segment length if needed

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


def probe_eval(model, train_loader, test_loader, results_path, folder, config):
    folder_path = os.path.join(results_path, folder)
    os.makedirs(folder_path, exist_ok=True)

    layer_output_pairs = model.get_layer_output_pairs()

    all_results = {}

    for layer_id, layer_num, nn_dim in tqdm(layer_output_pairs, desc=f"Probing Layers in {folder}", leave=False, unit="layer"):
        print(f"Evaluating for layer: {layer_id}, number: {layer_num}")

        # Instantiate the classifier for the current layer
        classifier_model = LinearProbeModel(num_classes=21, model_type="neural_net",
                                            model=model, freeze_layers=True,
                                            layer_num=layer_num, layer_id=layer_id,
                                            classifier_dims=nn_dim)
        classifier_model = classifier_model.to(device)

        # Create a trainer and perform training
        trainer = LinearProbeTrainer(classifier_model, train_loader, test_loader, device, lr=1e-3, plotting=False, batches_per_eval=100, desired_total_batches=3e4, patience=4, use_tqdm=True)

        trainer.train()

        # Evaluate the model
        evaluator = ModelEvaluator(classifier_model, test_loader, num_classes=21, device=device, use_tqdm=False)
        class_frame_error_rates, total_frame_error_rate = evaluator.validate_model_multiple_passes(num_passes=1, max_batches=1e4)

        all_results[f"{layer_id}_{layer_num}"] = {
            "total_error_rate": total_frame_error_rate,
            "class_error_rates": class_frame_error_rates
        }      

        umap_path = os.path.join(folder_path, f"UMAP of layer_num: {layer_num} sub_layer: {layer_id}.png") 
        category_colors_path = os.path.join(project_root, "files/category_colors_llb3.pkl") 
        try:
            plot_umap_projection(
                model=model, 
                device=device, 
                data_dir=test_dir, 
                subsample_factor=config['subsample'],  # Using new config parameter
                remove_silences=False,  # Using new config parameter
                samples=100, 
                file_path=category_colors_path, 
                layer_index=layer_num, 
                dict_key=layer_id, 
                time_bins_per_umap_point=1, 
                context=1000,  # Using new config parameter
                raw_spectogram=False,
                save_dict_for_analysis=False,
                save_dir=umap_path
            )
        except Exception as e:
            error_file_path = os.path.join(folder_path, "UMAP_Error_Log.txt")
            with open(error_file_path, "w") as file:
                file.write(f"UMAP plot for layer_num: {layer_num}, sub_layer: {layer_id} could not be created.\n")
                file.write(f"Error: {e}")

    return all_results


def get_highest_numbered_file(files):
    weights_files = [f for f in files if re.match(r'model_step_\d+\.pth', f)]
    if not weights_files:
        return None
    highest_num = max([int(re.search(r'\d+', f).group()) for f in weights_files])
    return f'model_step_{highest_num}.pth'

import json

def execute_eval_of_experiments(base_path, results_path, train_loader, test_loader):
    base_path = os.path.join(project_root, base_path)  # Combine with project_root
    results_path = os.path.join(project_root, results_path)  # Combine with project_root
    experiment_folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder)) and folder != 'archive']
    
    all_experiments_results = {}  # Dictionary to store results from all experiments

    for folder in tqdm(experiment_folders, desc="Evaluating Experiments", unit="experiment"):
        folder_path = os.path.join(base_path, folder)
        config_path = os.path.join(folder_path, 'config.json')  # Correctly form the path

        weights_folder = os.path.join(folder_path, 'saved_weights')

        if os.path.exists(weights_folder):
            files = os.listdir(weights_folder)
            weights_file = get_highest_numbered_file(files)
                
            if weights_file and config_path:
                weight_path = os.path.join(weights_folder, weights_file)
                model = load_model(config_path, weight_path)
                config = load_config(config_path)
                experiment_results = probe_eval(model, train_loader, test_loader, results_path, folder, config)
                all_experiments_results[folder] = experiment_results
            else: 
                print(f"Files for eval of experiment '{folder}' not found")

    combined_results_file = os.path.join(results_path, 'combined_probe_results.json')
    with open(combined_results_file, 'w') as f:
        json.dump(all_experiments_results, f)

    return all_experiments_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

experiment_paths = "experiments"
results_path = "results"

all_results = execute_eval_of_experiments(experiment_paths, results_path, train_loader, test_loader)
