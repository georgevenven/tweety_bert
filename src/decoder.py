import os
import json
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_class import SongDataSet_Image, CollateFunction
from utils import load_model
from linear_probe import LinearProbeModel, LinearProbeTrainer
import argparse

class TweetyBertClassifier:
    def __init__(self, model_dir, linear_decoder_dir, context_length=1000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tweety_bert_model = self.load_tweety_bert(model_dir)
        self.linear_decoder_dir = linear_decoder_dir
        self.train_dir = os.path.join(linear_decoder_dir, "train")
        self.test_dir = os.path.join(linear_decoder_dir, "test")
        self.num_classes = None
        self.model_dir = model_dir
        self.data_file = None
        self.context_length = context_length

    def load_tweety_bert(self, model_dir):
        return load_model(model_dir)

    def prepare_data(self, data_file, test_train_split=0.8):
        self.data_file = data_file
        data = np.load(data_file)
        vocalization = data['vocalization']
        labels = data['hdbscan_labels']
        specs = data['s']

        specs = np.pad(specs, ((0, 0), (20, 0)), 'constant', constant_values=0)

        vocalization = labels  # temporary lazy solution

        # Replace noise labels (-1) with the closest non-noise label
        for i in range(len(labels)):
            if labels[i] == -1:
                left = right = i
                while left >= 0 or right < len(labels):
                    if left >= 0 and labels[left] != -1:
                        labels[i] = labels[left]
                        break
                    if right < len(labels) and labels[right] != -1:
                        labels[i] = labels[right]
                        break
                    left -= 1
                    right += 1

        self.num_classes = len(np.unique(labels))
        print(f"Number of classes: {self.num_classes}")

        list_of_data = [
            (labels[i:i+self.context_length], specs[i:i+self.context_length], vocalization[i:i+self.context_length])
            for i in range(0, len(labels), self.context_length)
        ]
        np.random.shuffle(list_of_data)

        split_index = int(len(list_of_data) * test_train_split)
        train_data = list_of_data[:split_index]
        test_data = list_of_data[split_index:]

        self._save_data(train_data, self.train_dir)
        self._save_data(test_data, self.test_dir)

    def _save_data(self, data, directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        for i, (labels, specs, vocalization) in enumerate(data):
            np.savez(os.path.join(directory, f"{i}.npz"), 
                     labels=labels, s=specs.T, vocalization=vocalization)

    def create_dataloaders(self, batch_size=42):
        collate_fn = CollateFunction(segment_length=self.context_length)

        train_dataset = SongDataSet_Image(self.train_dir, num_classes=self.num_classes, infinite_loader=False, segment_length=self.context_length)
        test_dataset = SongDataSet_Image(self.test_dir, num_classes=self.num_classes, infinite_loader=False, segment_length=self.context_length)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def create_classifier(self):
        if self.num_classes is None:
            raise ValueError("Number of classes is not set. Run prepare_data first.")
        
        self.classifier_model = LinearProbeModel(
            num_classes=self.num_classes, 
            model_type="neural_net", 
            model=self.tweety_bert_model, 
            freeze_layers=True, 
            layer_num=-1, 
            layer_id="attention_output", 
            TweetyBERT_readout_dims=196,
            classifier_type="decoder"
        ).to(self.device)

    def train_classifier(self, lr=1e-4, batches_per_eval=10, desired_total_batches=200, patience=4, generate_loss_plot=False):
        trainer = LinearProbeTrainer(
            model=self.classifier_model, 
            train_loader=self.train_loader, 
            test_loader=self.test_loader, 
            device=self.device, 
            lr=lr, 
            plotting=generate_loss_plot,
            batches_per_eval=batches_per_eval, 
            desired_total_batches=desired_total_batches, 
            patience=patience
        )
        trainer.train()

    def save_decoder_state(self):
        save_dir = os.path.join(self.linear_decoder_dir, "decoder_state")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save decoder weights
        torch.save(self.classifier_model.state_dict(), os.path.join(save_dir, "decoder_weights.pth"))

        # Save configuration
        config = {
            "num_classes": self.num_classes,
            "model_dir": self.model_dir,
            "data_file": self.data_file,
        }
        with open(os.path.join(save_dir, "decoder_config.json"), "w") as f:
            json.dump(config, f)

        shutil.copy2(self.data_file, os.path.join(save_dir, "original_data.npz"))

        print(f"Decoder state saved in {save_dir}")

    def generate_specs(self, num_specs=100):
        spec_generator = SpecGenerator(self.classifier_model, self.test_dir, self.context_length, num_classes=self.num_classes)
        spec_generator.generate_specs(num_specs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TweetyBert Classifier")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--bird_name", type=str, required=True, help="Name of the bird")
    parser.add_argument("--generate_loss_plot", action="store_true", default=False, help="Generate loss plot during training")

    args = parser.parse_args()

    experiment_name = args.experiment_name
    bird_name = args.bird_name
    model_dir = f"../experiments/{experiment_name}"
    linear_decoder_dir = f"../experiments_{bird_name}_linear_decoder"
    data_file = f"../files/{bird_name}.npz"

    classifier = TweetyBertClassifier(model_dir=model_dir, linear_decoder_dir=linear_decoder_dir)
    classifier.prepare_data(data_file)
    classifier.create_dataloaders()
    classifier.create_classifier()
    classifier.train_classifier(generate_loss_plot=args.generate_loss_plot)
    classifier.save_decoder_state()
    classifier.generate_specs()
