import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
import json
import sys
sys.path.append("src")
os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

from data_class import CollateFunction, SongDataSet_Image
from utils import load_model
from linear_probe import LinearProbeModel, LinearProbeTrainer, ModelEvaluator
import datetime

from TweetyNET import frame_error_rate, TweetyNET_Dataset, TweetyNetTrainer, CollateFunction, TweetyNet, ModelEvaluator

train_dir = '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/yarden_train'
test_dir = '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/yarden_test'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cv_parent_dir = "/media/george-vengrovski/disk1/supervised_eval_dataset"

# List all directories within 'cv_parent_dir'
all_dirs = [d for d in os.listdir(cv_parent_dir) if os.path.isdir(os.path.join(cv_parent_dir, d))]

# Filter out train and validation directories and sort them
train_dirs = sorted([d for d in all_dirs if "train" in d])
val_dirs = sorted([d for d in all_dirs if "val" in d])

# Pair each train directory with the corresponding val directory based on the naming pattern
cv_pairs = []
for train_dir in train_dirs:
    # Split the directory name and extract the size and iteration
    parts = train_dir.split('_')
    size = parts[-2]  # Second to last item is the size
    iteration = parts[-1]  # Last item is the iteration
    prefix = parts[0]
    # Construct the matching validation directory name using the correct prefix
    val_dir = f"{prefix}_val_{size}_{iteration}"

    if val_dir in val_dirs:
        cv_pairs.append((os.path.join(cv_parent_dir, train_dir), os.path.join(cv_parent_dir, val_dir)))
    else:
        print(f"No matching validation directory found for {train_dir}")

# Train and evaluate models for each cross-validation split
for idx, (train_dir, val_dir) in enumerate(cv_pairs):
    print(f"training on {train_dir} and evaluating on {val_dir}")

    train_dataset = TweetyNET_Dataset(train_dir, num_classes=30)
    test_dataset = TweetyNET_Dataset(val_dir, num_classes=30)

    collate_fn = CollateFunction(segment_length=370)  # Adjust the segment length if needed

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


    model = TweetyNet(num_classes=30, input_shape=(1, 513, 370))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2, weight_decay=0.0)

    use_gpu = True
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize the TweetyNetTrainer
    trainer = TweetyNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        optimizer=optimizer,
        desired_total_steps=1e4,  # Set your desired total steps
        patience=4  # Set your patience for early stopping
    )

    # Start the training process
    trainer.train()
    # Initialize the ModelEvaluator with the test_loader and the trained model
    evaluator = ModelEvaluator(test_loader=test_loader, model=model, device=device)

    # Validate the model. This method should return the class-wise and total frame error rates
    class_frame_error_rates, total_frame_error_rate = evaluator.validate_model_multiple_passes()

    # Use the name of the current cross-validation directory for the results folder
    results_folder_name = os.path.basename(val_dir)
    results_dir = os.path.join("results/llb3_tweety_net_linear_probe", results_folder_name)  # Modified to save into the relative path /results/{cv_dirs}
    os.makedirs(results_dir, exist_ok=True)

    # Save the results to a file for later inspection
    evaluator.save_results(class_frame_error_rates, total_frame_error_rate, results_dir)