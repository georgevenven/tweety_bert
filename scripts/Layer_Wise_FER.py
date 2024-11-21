"""
Layer-wise Frame Error Rate Analysis for TweetyBERT

This script performs layer-wise probing of a TweetyBERT model to analyze how well
each layer captures phrase information. It trains linear probes on the outputs of
each layer and evaluates their performance on phrase classification.
"""

import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src directory to path for imports
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append("src")

from linear_probe import LinearProbeModel, LinearProbeTrainer, ModelEvaluator
from utils import load_model, load_config
from data_class import SongDataSet_Image, CollateFunction, determine_number_unique_classes


# Set batch size to 96 ... because no gradients stored for tweetybert thus more space for linear probe + frozen TweetyBERT
def setup_data_loaders(train_dir, test_dir, num_classes, batch_size=96, segment_length=1000):
    """
    Creates DataLoaders for training, testing, and evaluation.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory
        num_classes (int): Number of unique classes
        batch_size (int): Batch size for training/testing
        segment_length (int): Length of audio segments
    
    Returns:
        tuple: (train_loader, test_loader, eval_loader)
    """
    train_dataset = SongDataSet_Image(train_dir, num_classes=num_classes, infinite_loader=False, segment_length=1000, phrase_labels=True)
    test_dataset = SongDataSet_Image(test_dir, num_classes=num_classes, infinite_loader=False, segment_length=1000, phrase_labels=True)
    eval_dataset = SongDataSet_Image(test_dir, num_classes=num_classes, infinite_loader=False, segment_length=1000, phrase_labels=True)

    collate_fn = CollateFunction(segment_length=segment_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader, eval_loader

def probe_eval(model, train_loader, test_loader, eval_loader, results_path, folder, config, num_classes, train_dir, skip_existing=True):
    # Create base folder for layer-wise analysis
    folder_path = os.path.join(results_path, 'layer_wise_analysis')
    os.makedirs(folder_path, exist_ok=True)
    
    experiment_path = os.path.join(folder_path, folder)
    os.makedirs(experiment_path, exist_ok=True)

    # Define the specific layers we want to analyze
    target_layers = [
        ('attention_output', 'Attention output'),
        ('intermediate_residual_stream', 'Intermediate residual'),
        ('feed_forward_output_gelu', 'FF output after GELU'),
        ('feed_forward_output', 'Final FF output')
    ]
    
    all_results = {}

    # Filter layer_output_pairs to only include our target layers
    layer_output_pairs = [
        (layer_id, layer_num, nn_dim) 
        for layer_id, layer_num, nn_dim in model.get_layer_output_pairs()
        if any(target[0] in layer_id for target in target_layers)
    ]

    for layer_id, layer_num, nn_dim in tqdm(layer_output_pairs, desc=f"Probing Key Transformer Layers in {folder}", leave=False):
        # Create specific folder for this layer
        layer_results_path = os.path.join(experiment_path, f'layer_{layer_num}_{layer_id}')
        
        # Check if this layer has already been processed
        results_file = os.path.join(layer_results_path, 'results.json')
        if skip_existing and os.path.exists(results_file):
            print(f"Skipping already processed layer: {layer_id}, number: {layer_num}")
            
            # Load existing results
            try:
                with open(results_file, 'r') as f:
                    layer_results = json.load(f)
                all_results[f'layer_{layer_num}_{layer_id}'] = layer_results
                continue
            except Exception as e:
                print(f"Error loading existing results for layer {layer_id}, number: {layer_num}. Will reprocess. Error: {str(e)}")
        
        print(f"Evaluating transformer layer: {layer_id}, number: {layer_num}")
        os.makedirs(layer_results_path, exist_ok=True)

        # Initialize probe model for this layer
        classifier_model = LinearProbeModel(
            num_classes=num_classes, 
            model_type="neural_net", 
            model=model, 
            freeze_layers=True, 
            layer_num=layer_num, 
            layer_id=layer_id, 
            TweetyBERT_readout_dims=nn_dim, 
            classifier_type="linear_probe"
        )
        classifier_model = classifier_model.to(device)
        
        # Train the probe
        trainer = LinearProbeTrainer(
            model=classifier_model, 
            train_loader=train_loader, 
            test_loader=test_loader, 
            device=device, 
            lr=6e-4, 
            plotting=False, 
            batches_per_eval=25, 
            desired_total_batches=1000, 
            patience=6
        )
        weight_differences = trainer.train()

        # Evaluate the probe
        evaluator = ModelEvaluator(
            model=classifier_model, 
            test_loader=eval_loader, 
            num_classes=num_classes,
            device=device,
            filter_unseen_classes=True,
            train_dir=train_dir
        )

        class_frame_error_rates, total_frame_error_rate, errors_per_class, correct_per_class = evaluator.evalulate_model(
            num_passes=1,
            context=1000
        )

        # Save results for this layer
        evaluator.save_results(
            class_frame_error_rates, 
            total_frame_error_rate, 
            layer_results_path,
            errors_per_class,
            correct_per_class
        )

        all_results[f'layer_{layer_num}_{layer_id}'] = {
            'class_frame_error_rates': class_frame_error_rates,
            'total_frame_error_rate': total_frame_error_rate,
            'errors_per_class': errors_per_class,
            'correct_per_class': correct_per_class
        }

        # After evaluation, clean up memory
        del classifier_model, trainer, evaluator
        torch.cuda.empty_cache()

    # Save combined results at the experiment level
    combined_results_file = os.path.join(experiment_path, 'combined_layer_results.json')
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f)

    return all_results

def execute_eval_of_experiments(experiment_configs, results_path):
    """
    Executes evaluation across multiple experiments.
    
    Args:
        experiment_configs (list): List of experiment configurations
        results_path (str): Directory to save results
    
    Returns:
        dict: Combined results from all experiments
    """
    all_experiments_results = {}

    for exp_config in experiment_configs:
        base_path = os.path.join(project_root, exp_config['experiment_path'])
        train_dir = os.path.join(project_root, exp_config['train_dir'])
        test_dir = os.path.join(project_root, exp_config['test_dir'])

        # Extract dataset name (llb3, llb11, or llb16) from train_dir
        dataset_name = os.path.basename(train_dir).split('_')[0]

        # Setup data and model
        num_classes = determine_number_unique_classes(train_dir)
        train_loader, test_loader, eval_loader = setup_data_loaders(train_dir, test_dir, num_classes)

        if os.path.exists(base_path):
            try:
                model = load_model(base_path).to(device)
                config = load_config(os.path.join(base_path, 'config.json'))
                experiment_results = probe_eval(
                    model, train_loader, test_loader, eval_loader, 
                    results_path, dataset_name, 
                    config, num_classes, train_dir
                )
                all_experiments_results[dataset_name] = experiment_results
            except Exception as e:
                print(f"Error loading experiment '{base_path}': {str(e)}")
        else:
            print(f"Experiment path not found: {base_path}")

    # Save combined results
    combined_results_file = os.path.join(results_path, 'combined_probe_results.json')
    with open(combined_results_file, 'w') as f:
        json.dump(all_experiments_results, f)

    return all_experiments_results

# Setup device and run evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## Output Structure: 
# results/
# └── layer_wise_analysis/
#     ├── llb3/
#     │   ├── layer_0_Q/
#     │   ├── layer_1_K/
#     │   └── combined_layer_results.json
#     ├── llb11/
#     │   ├── layer_0_Q/
#     │   ├── layer_1_K/
#     │   └── combined_layer_results.json
#     └── llb16/
#         ├── layer_0_Q/
#         ├── layer_1_K/
#         └── combined_layer_results.json

# Configuration for experiments
experiment_configs = [
    {
        "experiment_path": "/media/george-vengrovski/George-SSD/llb_stuff/LLB_Model_For_Paper",
        "train_dir": "/media/george-vengrovski/George-SSD/llb_stuff/llb3_train",
        "test_dir": "/media/george-vengrovski/George-SSD/llb_stuff/llb3_test"
    },
    {
        "experiment_path": "/media/george-vengrovski/George-SSD/llb_stuff/LLB_Model_For_Paper",
        "train_dir": "/media/george-vengrovski/George-SSD/llb_stuff/llb11_train",
        "test_dir": "/media/george-vengrovski/George-SSD/llb_stuff/llb11_test"
    },
    {
        "experiment_path": "/media/george-vengrovski/George-SSD/llb_stuff/LLB_Model_For_Paper",
        "train_dir": "/media/george-vengrovski/George-SSD/llb_stuff/llb16_train",
        "test_dir": "/media/george-vengrovski/George-SSD/llb_stuff/llb16_test"
    }
]

results_path = "results"
all_results = execute_eval_of_experiments(experiment_configs, results_path)