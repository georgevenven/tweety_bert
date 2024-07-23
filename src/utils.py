import torch 
import json
from model import TweetyBERT

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

def load_model(config_path, weight_path=None):
    """
    Initialize and load the model with the given configuration and weights.

    Args:
    config_path (str): The path to the model configuration file.
    weight_path (str, optional): The path to the model weights. If not provided, initializes with random weights.

    Returns:
    torch.nn.Module: The initialized model.
    """
    config = load_config(config_path)

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

    if weight_path:
        load_weights(dir=weight_path, model=model)
    else:
        print("Model loaded with randomly initiated weights.")

    return model
