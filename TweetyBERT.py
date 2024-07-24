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
        "experiment_name": "5288_WhisperSeg_Preprocessing_Padded",
        "continue_training": False,
        "train_dir": "/media/george-vengrovski/disk1/5288_train_padded",
        "test_dir": "/media/george-vengrovski/disk1/5288_test_padded",
        "batch_size": 12,
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
        "learning_rate": 5e-4,
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
     
#         {
#         "experiment_name": "Budige_Smoothing_Test",
#         "continue_training": True,
#         # "train_dir": "/media/george-vengrovski/disk1/multispecies_data_set_train",
#         # "test_dir": "/media/george-vengrovski/disk1/budgie_test_specs",
#         # "max_steps": 5e10,
#         # "eval_interval": 1,
#         # "patience": 800,
#         # "m": 50
#         },
]


for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
