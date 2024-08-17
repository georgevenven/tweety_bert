import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from experiment_manager import ExperimentRunner

#Initialize experiment runner
experiment_runner = ExperimentRunner(device="cuda")

configurations = [
        {
        "experiment_name": "LLB3_Whisperseg",
        "continue_training": False,
        "train_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb3_train",
        "test_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb3_test",
        "batch_size": 42,
        "d_transformer": 196,   
        "nhead_transformer": 4,
        "num_freq_bins": 196,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 4,
        "m": 250,
        "p": 0.01,
        "alpha": 1,
        "pos_enc_type": "relative",
        "pitch_shift": True,
        "learning_rate": 3e-4,
        "max_steps": 5e4,
        "eval_interval": 500,
        "save_interval": 500,
        "context": 1000,
        "weight_decay": 0.0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        },
        {
        "experiment_name": "LLB11_Whisperseg",
        "continue_training": False,
        "train_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb11_train",
        "test_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb11_test",
        "batch_size": 42,
        "d_transformer": 196,   
        "nhead_transformer": 4,
        "num_freq_bins": 196,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 4,
        "m": 250,
        "p": 0.01,
        "alpha": 1,
        "pos_enc_type": "relative",
        "pitch_shift": True,
        "learning_rate": 3e-4,
        "max_steps": 5e4,
        "eval_interval": 500,
        "save_interval": 500,
        "context": 1000,
        "weight_decay": 0.0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        },
        {
        "experiment_name": "LLB16_Whisperseg",
        "continue_training": False,
        "train_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb16_train",
        "test_dir": "/media/george-vengrovski/Extreme SSD/yarden_data/llb16_test",
        "batch_size": 42,
        "d_transformer": 196,   
        "nhead_transformer": 4,
        "num_freq_bins": 196,
        "dropout": 0.2,
        "dim_feedforward": 768,
        "transformer_layers": 4,
        "m": 250,
        "p": 0.01,
        "alpha": 1,
        "pos_enc_type": "relative",
        "pitch_shift": True,
        "learning_rate": 3e-4,
        "max_steps": 5e4,
        "eval_interval": 500,
        "save_interval": 500,
        "context": 1000,
        "weight_decay": 0.0,
        "early_stopping": True,
        "patience": 8,
        "trailing_avg_window": 200,
        "num_ground_truth_labels": 50
        },

]


for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
