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

# Load TweetyBERT model
weights_path = "experiments/Yarden_FreqTruncated/saved_weights/model_step_17000.pth"
config_path = "experiments/Yarden_FreqTruncated/config.json"
tweety_bert_model = load_model(config_path, weights_path)

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

    train_dataset = SongDataSet_Image(train_dir, num_classes=50)
    val_dataset = SongDataSet_Image(val_dir, num_classes=50)
    collate_fn = CollateFunction(segment_length=1000)  # Adjust the segment length if needed
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, collate_fn=collate_fn)

    # Initialize and train classifier model, the num classes is a hack and needs to be fixed later on by removing one hot encodings 
    classifier_model = LinearProbeModel(num_classes=21, model_type="neural_net", model=tweety_bert_model,
                                        freeze_layers=True, layer_num=-2, layer_id="attention_output", classifier_dims=196)

    classifier_model = classifier_model.to(device)
    trainer = LinearProbeTrainer(model=classifier_model, train_loader=train_loader, test_loader=val_loader,
                                 device=device, lr=1e-4, plotting=False, batches_per_eval=50, desired_total_batches=1e4, patience=4)
    trainer.train()

    eval_dataset = SongDataSet_Image(val_dir, num_classes=50, infinite_loader=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Evaluate the trained model
    evaluator = ModelEvaluator(model=classifier_model, test_loader=eval_loader, num_classes=50,
                               device='cuda:0', filter_unseen_classes=True, train_dir=train_dir)
    class_frame_error_rates, total_frame_error_rate = evaluator.validate_model_multiple_passes(num_passes=1, max_batches=1250)

    # Use the name of the current cross-validation directory for the results folder
    results_folder_name = os.path.basename(val_dir)

    # Save the evaluation results
    results_dir = os.path.join("results/llb3_demo_linear_probe", results_folder_name)  # Modified to save into the relative path /results/{cv_dirs}
    os.makedirs(results_dir, exist_ok=True)
    evaluator.save_results(class_frame_error_rates, total_frame_error_rate, results_dir)