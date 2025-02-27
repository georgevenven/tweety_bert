import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Assuming 'src' is a directory containing your modules
sys.path.append("src")

from data_class import SongDataSet_Image, determine_number_unique_classes, CollateFunction
from utils import load_model
from linear_probe import LinearProbeModel, LinearProbeTrainer, ModelEvaluator

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base directories for datasets
datasets = ["llb3", "llb11", "llb16"]
base_data_dir = "/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset"

# Configurations to iterate over
configurations = [
    {
        "name": "TweetyBERT_linear_probe",
        "freeze_layers": True,
        "model_type": "neural_net",
        "layer_id": "attention_output",
        "learning_rate": 1e-2
    },
    {
        "name": "TweetyBERT_finetuned",
        "freeze_layers": False,
        "model_type": "neural_net",
        "layer_id": "embedding",  # will automatically use the last layer
        "learning_rate": 3e-4
    },
    {
        "name": "TweetyBERT_untrained",
        "freeze_layers": True,
        "model_type": "neural_net",
        "layer_id": "attention_output",
        "learning_rate": 1e-2
    },
    {
        "name": "linear_probe_raw_spectrogram",
        "freeze_layers": True,
        "model_type": "raw",
        "layer_id": None,  # nn not used here
        "learning_rate": 1e-2
    }
]

# Loop over each dataset
for dataset in tqdm(datasets, desc="Processing datasets"):
    print(f"Processing dataset: {dataset}")
    train_dir = os.path.join(base_data_dir, f"{dataset}_train")
    test_dir = os.path.join(base_data_dir, f"{dataset}_test")

    # Determine the number of unique classes in the training set
    num_classes = determine_number_unique_classes(train_dir)
    print(f"Number of classes in {dataset}: {num_classes}")

    # Create datasets and data loaders
    train_dataset = SongDataSet_Image(train_dir, num_classes=num_classes, infinite_loader=False, phrase_labels=True, pitch_shift=False)
    test_dataset = SongDataSet_Image(test_dir, num_classes=num_classes, infinite_loader=False, phrase_labels=True, pitch_shift=False)

    collate_fn = CollateFunction(segment_length=1000)
    train_loader = DataLoader(train_dataset, batch_size=42, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=42, shuffle=True, collate_fn=collate_fn)

    # Loop over each configuration
    for config in tqdm(configurations, desc=f"Running configs for {dataset}"):
        print(f"Running configuration: {config['name']}")

        # Path to experiment folder
        experiment_folder = "experiments/TweetyBERT_Paper_Yarden_Model"

        if config["name"] == "TweetyBERT_untrained":  
            # Load the TweetyBERT model once to use in configurations that require it
            tweety_bert_model = load_model(experiment_folder, random_init=True).to(device)
        else:
            tweety_bert_model = load_model(experiment_folder, random_init=False).to(device)

        # Initialize the classifier model using config parameters
        classifier_model = LinearProbeModel(
            num_classes=num_classes,
            model_type=config["model_type"],
            model=tweety_bert_model, 
            freeze_layers=config["freeze_layers"],
            layer_num=-2,
            layer_id=config["layer_id"],
            TweetyBERT_readout_dims=196,
            classifier_type="linear_probe"
        )

        classifier_model = classifier_model.to(device)

        # Set up the trainer
        trainer = LinearProbeTrainer(
            model=classifier_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=config["learning_rate"],
            plotting=False,
            batches_per_eval=25,
            desired_total_batches=5000,
            patience=6
        )

        # Train the model
        print("Starting training...")
        weight_differences = trainer.train()
        print("Training completed.")

        # Prepare evaluation dataset and loader
        eval_dataset = SongDataSet_Image(
            test_dir,
            num_classes=num_classes,
            infinite_loader=False,
            segment_length=1000,
            phrase_labels=True,
            pitch_shift=False
        )
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

        # Initialize the evaluator
        evaluator = ModelEvaluator(
            model=classifier_model,
            test_loader=eval_loader,
            num_classes=num_classes,
            device=device,
            filter_unseen_classes=True,
            train_dir=train_dir
        )

        # Evaluate the model
        print("Starting evaluation...")
        class_frame_error_rates, total_frame_error_rate, errors_per_class, correct_per_class = evaluator.evalulate_model(
            num_passes=1,
            context=1000
        )
        print("Evaluation completed.")

        # Save the results
        results_dir = f"results/{config['name']}_{dataset}"
        os.makedirs(results_dir, exist_ok=True)
        evaluator.save_results(
            class_frame_error_rates, 
            total_frame_error_rate, 
            results_dir,
            errors_per_class,
            correct_per_class
        )
        print(f"Results saved to {results_dir}")

print("All configurations have been processed.")
