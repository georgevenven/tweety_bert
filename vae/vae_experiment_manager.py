import sys
import os

sys.path.append("src")
os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

import torch
from data_class import SongDataSet_Image, CollateFunction
from torch.utils.data import DataLoader
from analysis import plot_umap_projection
from utils import detailed_count_parameters
import json
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from vae import VariationalAutoencoder
from vae_trainer import ModelTrainer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ExperimentRunner:
    def __init__(self, device, base_save_dir='experiments'):
        self.device = device
        self.base_save_dir = base_save_dir
        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)

    def archive_existing_experiments(self, experiment_name):
        archive_dir = os.path.join(self.base_save_dir, 'archive')
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)

        # Check if the experiment with the same name already exists
        source = os.path.join(self.base_save_dir, experiment_name)
        if os.path.exists(source):
            base_destination = os.path.join(archive_dir, experiment_name)
            destination = base_destination
            # Check for duplicates and create a unique name for the archive
            copy_number = 1
            while os.path.exists(destination):
                # Append a copy number to the experiment name
                destination = f"{base_destination}_copy{copy_number}"
                copy_number += 1
            
            # Move the current folder to the archive directory with the unique name
            shutil.move(source, destination)
    
    def run_experiment(self, config, i):
        experiment_name = config.get('experiment_name', f"experiment_{i}")
        self.archive_existing_experiments(experiment_name)
        
        # Create a directory for this experiment based on experiment_name
        experiment_dir = os.path.join(self.base_save_dir, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        
        # Save the config as a metadata file
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        # Data Loading
        collate_fn = CollateFunction(segment_length=config['input_width'])
        train_dataset = SongDataSet_Image(config['train_dir'], num_classes=1)
        test_dataset = SongDataSet_Image(config['test_dir'], num_classes=1)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=16)
        
        model = VariationalAutoencoder(config["latent_dims"], config["variational_beta"], input_height=config["input_height"], input_width=config["input_width"]).to(self.device)

        detailed_count_parameters(model)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        saved_weights_dir = os.path.join(experiment_dir, 'saved_weights')

        if not os.path.exists(saved_weights_dir):
            os.makedirs(saved_weights_dir)

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Initialize trainer
        trainer = ModelTrainer(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            self.device,  
            weights_save_dir=saved_weights_dir,  # pass the directory for saved weights
            experiment_dir=experiment_dir, 
            max_steps=config['max_steps'], 
            eval_interval=config['eval_interval'], 
            save_interval=config['save_interval'], 
            overfit_on_batch=False, 
            early_stopping=config['early_stopping'],
            patience=config['patience'],
            trailing_avg_window=config['trailing_avg_window'],
        )        

        # Train the model
        trainer.train()
        
        # Plot the results
        trainer.plot_results(save_plot=True, config=config)


#Initialize experiment runner
experiment_runner = ExperimentRunner(device="cuda")

# Define configurations
configurations = [
        {"experiment_name": "VAE-Bungie-Rawer-Specs", "latent_dims": 32, "variational_beta": 1, "input_width": 100,"input_height": 513,  "train_dir": "files/dev_train", "test_dir": "files/dev_test", "batch_size": 48, "learning_rate": 3e-4, "max_steps": 25e3, "eval_interval": 250, "save_interval": 1000, "remove_silences": False, "early_stopping": True, "patience": 4, "trailing_avg_window":200}
]

for i, config in enumerate(configurations):
    experiment_runner.run_experiment(config, i)
