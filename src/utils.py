import torch 
import json
from model import TweetyBERT
import os

def load_weights(dir, model):
    """
    Load the saved weights into the model.

    Args:
    dir (str): The directory where the model weights are saved.
    model (torch.nn.Module): The model into which weights are to be loaded.

    Raises:
    FileNotFoundError: If the weights file is not found.
    """
    try:
        model.load_state_dict(torch.load(dir))
    except FileNotFoundError:
        raise FileNotFoundError(f"Weight file not found at {dir}")

def detailed_count_parameters(model, print_layer_params=False):
    """
    Print details of layers with the number of trainable parameters in the model.

    Args:
    model (torch.nn.Module): The model whose parameters are to be counted.
    print_layer_params (bool): If True, prints parameters for each layer.
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
        if print_layer_params:
            print(f"Layer: {name} | Parameters: {param:,} | Shape: {list(parameter.shape)}")
    print(f"Total Trainable Parameters: {total_params:,}")

def load_config(config_path):
    """
    Load the configuration file.

    Args:
    config_path (str): The path to the configuration JSON file.

    Returns:
    dict: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

def load_model(experiment_folder):
    """
    Initialize and load the model with the given configuration and weights from the experiment folder.

    Args:
    experiment_folder (str): The path to the experiment folder containing the config and weights.

    Returns:
    torch.nn.Module: The initialized model.
    """
    config_path = os.path.join(experiment_folder, 'config.json')
    weight_folder = os.path.join(experiment_folder, 'saved_weights')

    # Load configuration
    config = load_config(config_path)

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
        length=config['context']
    )

    # Find the weight file with the highest step number
    weight_files = [f for f in os.listdir(weight_folder) if f.startswith('model_step_') and f.endswith('.pth')]
    if weight_files:
        latest_weight_file = max(weight_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        weight_path = os.path.join(weight_folder, latest_weight_file)
        load_weights(dir=weight_path, model=model)
    else:
        print("Model loaded with randomly initiated weights.")

    return model
