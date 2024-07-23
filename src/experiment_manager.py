import os
import torch
from data_class import SongDataSet_Image, CollateFunction
from model import TweetyBERT
from trainer import ModelTrainer
from utils import detailed_count_parameters
import json
import shutil
from itertools import cycle
from torch.utils.data import DataLoader

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
    
    def find_latest_weights(self, experiment_dir):
        saved_weights_dir = os.path.join(experiment_dir, 'saved_weights')
        weight_files = [os.path.join(saved_weights_dir, f) for f in os.listdir(saved_weights_dir) if f.endswith('.pth')]
        if not weight_files:
            return None
        # Return the latest weights file based on file modification time
        return max(weight_files, key=os.path.getmtime)
    
    def load_existing_statistics(self, experiment_dir):
        training_stats = {}
        last_step = 0

        with open(os.path.join(experiment_dir, "training_statistics.json"), 'r') as json_file:
            stats = json.load(json_file)
            last_step = max(last_step, stats['step'])
            for key, value in stats.items():
                training_stats[key] = value  # Assign value directly without wrapping in a list

        return training_stats, last_step

    def run_experiment(self, config, i):
        experiment_name = config.get('experiment_name', f"experiment_{i}")

        # Determine the directory for this experiment
        experiment_dir = os.path.join(self.base_save_dir, experiment_name)
        
        training_stats, last_step = None, 0  # Initialize

        if config.get('continue_training', False):
            assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist for continuing training."
            
            # Load the existing config
            with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
                loaded_config = json.load(f)

            # Check for potential conflicts in config that would affect model loading
            critical_keys = ['d_transformer', 'nhead_transformer', 'num_freq_bins', 'num_ground_truth_labels', 'transformer_layers']
            conflict_keys = [key for key in critical_keys if key in config and config[key] != loaded_config[key]]

            if conflict_keys:
                print("WARNING: Conflicting config values that affect model structure:", conflict_keys)
                print("Experiment stopped to prevent model loading issues. Please ensure compatibility or start a new experiment.")
                return
            else:
                loaded_config.update(config)  # Update loaded config with new values, prioritizing passed config
                config = loaded_config

            # Load existing training statistics to resume training
            training_stats, last_step = self.load_existing_statistics(experiment_dir)

        else:
            # Archive existing experiment if needed and create a new directory
            self.archive_existing_experiments(experiment_name)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            
        
        # Save the config as a metadata file
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        # Data Loading
        collate_fn = CollateFunction(segment_length=config['context'])
        train_dataset = SongDataSet_Image(config['train_dir'], num_classes=config['num_ground_truth_labels'])
        test_dataset = SongDataSet_Image(config['test_dir'], num_classes=config['num_ground_truth_labels'])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=16)
        
        # Initialize model
        model = TweetyBERT(
            d_transformer=config['d_transformer'], 
            nhead_transformer=config['nhead_transformer'],
            num_freq_bins=config['num_freq_bins'],
            num_labels=config['num_ground_truth_labels'],
            dropout=config['dropout'],
            dim_feedforward=config['dim_feedforward'],
            transformer_layers=config['transformer_layers'],
            m=config['m'],
            p=config['p'],
            alpha=config['alpha'],
            pos_enc_type=config['pos_enc_type'],
            length = config['context']
        ).to(self.device)

        # Load latest weights if continuing training
        if config.get('continue_training', False):
            weights_path = self.find_latest_weights(experiment_dir)
            if weights_path is not None:
                print("weights to continue training has been succesfully found!")
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
            else:
                print("No weights found to continue training.")
                return

        detailed_count_parameters(model)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
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
            trailing_avg_window=config['trailing_avg_window']
            )        

        # Initialize trainer with potentially passed training_stats and last_step
        trainer = ModelTrainer(
            model, 
            train_loader, 
            test_loader, 
            optimizer, 
            self.device,  
            weights_save_dir=saved_weights_dir,  
            experiment_dir=experiment_dir, 
            max_steps=config['max_steps'], 
            eval_interval=config['eval_interval'], 
            save_interval=config['save_interval'], 
            overfit_on_batch=False, 
            early_stopping=config['early_stopping'],
            patience=config['patience'],
            trailing_avg_window=config['trailing_avg_window']
        )        

        # # Train the model with passed training_stats and last_step
        trainer.train(continue_training=config.get('continue_training', False), training_stats=training_stats, last_step=last_step)
        
        # Plot the results
        trainer.plot_results(save_plot=True, config=config)