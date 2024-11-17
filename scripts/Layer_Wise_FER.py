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
from data_class import SongDataSet_Image, CollateFunction, deterimine_number_unique_classes

def load_config_from_path(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config 

def setup_data_loaders(train_dir, test_dir, num_classes, batch_size=48, segment_length=1000):
    print("Setting up data loaders...")
    train_dataset = SongDataSet_Image(train_dir, num_classes=num_classes, infinite_loader=False)
    test_dataset = SongDataSet_Image(test_dir, num_classes=num_classes, infinite_loader=False)
    eval_dataset = SongDataSet_Image(test_dir, num_classes=num_classes, infinite_loader=False, segment_length=None)

    collate_fn = CollateFunction(segment_length=segment_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader, eval_loader

def probe_eval(model, train_loader, test_loader, eval_loader, results_path, folder, config, num_classes, train_dir):
    folder_path = os.path.join(results_path, folder)
    os.makedirs(folder_path, exist_ok=True)

    layer_output_pairs = model.get_layer_output_pairs()
    all_results = {}

    for layer_id, layer_num, nn_dim in tqdm(layer_output_pairs, desc=f"Probing Layers in {folder}", leave=False, unit="layer"):
        print(f"Evaluating for layer: {layer_id}, number: {layer_num}")

        classifier_model = LinearProbeModel(num_classes=num_classes, model_type="neural_net", model=model, freeze_layers=True, layer_num=layer_num, layer_id=layer_id, TweetyBERT_readout_dims=nn_dim, classifier_type="linear_probe")
        classifier_model = classifier_model.to(device)
        
        trainer = LinearProbeTrainer(model=classifier_model, train_loader=train_loader, test_loader=test_loader, device=device, lr=1e-4, plotting=False, batches_per_eval=100, desired_total_batches=1000, patience=4)
        print("Training classifier model...")
        weight_differences = trainer.train()

        evaluator = ModelEvaluator(model=classifier_model, 
                                   test_loader=eval_loader, 
                                   num_classes=num_classes,
                                   device=device,
                                   filter_unseen_classes=True,
                                   train_dir=train_dir)

        print("Evaluating model...")
        class_frame_error_rates, total_frame_error_rate = evaluator.evalulate_model(num_passes=1, context=1000)

        layer_results_path = os.path.join(folder_path, f'layer_{layer_num}_{layer_id}')
        print(f"Saving results to {layer_results_path}...")
        evaluator.save_results(class_frame_error_rates, total_frame_error_rate, layer_results_path)

        all_results[f'layer_{layer_num}_{layer_id}'] = {
            'class_frame_error_rates': class_frame_error_rates,
            'total_frame_error_rate': total_frame_error_rate
        }

    return all_results

def get_highest_numbered_file(files):
    weights_files = [f for f in files if re.match(r'model_step_\d+\.pth', f)]
    if not weights_files:
        return None
    highest_num = max([int(re.search(r'\d+', f).group()) for f in weights_files])
    return f'model_step_{highest_num}.pth'

def execute_eval_of_experiments(experiment_configs, results_path):
    all_experiments_results = {}

    for exp_config in experiment_configs:
        base_path = os.path.join(project_root, exp_config['experiment_path'])
        train_dir = os.path.join(project_root, exp_config['train_dir'])
        test_dir = os.path.join(project_root, exp_config['test_dir'])

        print(f"Determining number of unique classes in {train_dir}...")
        num_classes = deterimine_number_unique_classes(train_dir)
        train_loader, test_loader, eval_loader = setup_data_loaders(train_dir, test_dir, num_classes)

        if os.path.exists(base_path):
            print(f"Loading model from {base_path}...")
            try:
                model = load_model(base_path)
                config = load_config(os.path.join(base_path, 'config.json'))
                experiment_results = probe_eval(model, train_loader, test_loader, eval_loader, 
                                             results_path, os.path.basename(base_path), 
                                             config, num_classes, train_dir)
                all_experiments_results[os.path.basename(base_path)] = experiment_results
            except Exception as e:
                print(f"Error loading experiment '{base_path}': {str(e)}")
        else:
            print(f"Experiment path not found: {base_path}")

    combined_results_file = os.path.join(results_path, 'combined_probe_results.json')
    print(f"Saving combined results to {combined_results_file}...")
    with open(combined_results_file, 'w') as f:
        json.dump(all_experiments_results, f)

    return all_experiments_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Specify the three sets of models and their corresponding data directories
experiment_configs = [
    {
        "experiment_path": "/media/george-vengrovski/Extreme SSD/LLB3_NONORM_NOTHRESH",
        "train_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb3_no_norm_no_thresh_train",
        "test_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb3_no_norm_no_thresh_test"
    },
    {
        "experiment_path": "/media/george-vengrovski/Extreme SSD/LLB11_NONORM_NOTHRESH",
        "train_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb11_no_norm_no_thresh_train",
        "test_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb11_no_norm_no_thresh_test"
    },
    {
        "experiment_path": "experiments/LLB16_Whisperseg_NoNormNoThresholding",
        "train_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb16_no_threshold_no_norm_train",
        "test_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb16_no_threshold_no_norm_test"
    }
]
results_path = "results"

all_results = execute_eval_of_experiments(experiment_configs, results_path)