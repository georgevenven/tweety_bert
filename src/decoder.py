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
import matplotlib.pyplot as plt
from scipy.stats import mode
from tqdm import tqdm

def majority_vote(arr, window_size):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2)
        result[i] = mode(arr[start:end])[0]
    return result

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

        # Delete existing decoder directory if it exists
        if os.path.exists(linear_decoder_dir):
            shutil.rmtree(linear_decoder_dir)
        os.makedirs(linear_decoder_dir)

    @classmethod
    def load_decoder_state(cls, linear_decoder_dir):
        save_dir = os.path.join(linear_decoder_dir, "decoder_state")
        
        # Load configuration
        with open(os.path.join(save_dir, "decoder_config.json"), "r") as f:
            config = json.load(f)

        # Create instance
        instance = cls(config["model_dir"], linear_decoder_dir)
        
        # Set attributes
        instance.num_classes = config["num_classes"]
        instance.data_file = config["data_file"]

        # Create and load classifier
        instance.create_classifier()
        instance.classifier_model.load_state_dict(torch.load(os.path.join(save_dir, "decoder_weights.pth")))

        print(f"Decoder state loaded from {save_dir}")
        return instance

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


class SpecGenerator:
    def __init__(self, model, song_dir, context_length, num_classes=None):
        self.model = model
        self.song_dir = song_dir
        self.context_length = context_length
        self.spec_height = 196
        self.device = next(model.parameters()).device
        self.num_classes = num_classes if num_classes is not None else model.num_classes

    def z_score_spectrogram(self, spec):
        spec_mean = np.mean(spec)
        spec_std = np.std(spec)
        return (spec - spec_mean) / spec_std

    def process_spec_chunk(self, spec_chunk, label_chunk, output_dir, file, i):
        spec_tensor = torch.Tensor(spec_chunk).to(self.device).unsqueeze(0).unsqueeze(0)
        logits = self.model.forward(spec_tensor)
        
        if self.num_classes <= 2:
            # For binary classification or single class detection
            predicted_probs = torch.sigmoid(logits.squeeze())
            predicted_classes = (predicted_probs > 0.5).long().cpu().detach().numpy()
        else:
            # For multi-class classification
            predicted_classes = torch.argmax(logits, dim=2).cpu().detach().numpy().flatten()
        
        self.plot_and_save(spec_chunk, predicted_classes, label_chunk, logits, output_dir, file, i)

    def plot_and_save(self, spec_chunk, predicted_classes, label_chunk, logits, output_dir, file, i):
        plt.figure(figsize=(40, 15))  # Increased width significantly
        plt.subplots_adjust(hspace=0.5, left=0.05, right=0.98, top=0.95, bottom=0.05)

        # Plot spectrogram
        ax1 = plt.subplot2grid((15, 1), (0, 0), rowspan=8)
        im1 = ax1.imshow(spec_chunk, aspect='auto', origin='lower', extent=[0, spec_chunk.shape[1], 0, spec_chunk.shape[0]])
        ax1.set_title('Spectrogram', fontsize=24)
        ax1.tick_params(axis='both', which='major', labelsize=18)

        # Plot ground truth labels
        ax2 = plt.subplot2grid((15, 1), (8, 0), rowspan=1, sharex=ax1)
        ax2.imshow([label_chunk], aspect='auto', origin='lower', cmap='viridis', extent=[0, len(label_chunk), 0, 1])
        ax2.set_title('Ground Truth Labels', fontsize=24)
        ax2.set_yticks([])

        # Plot predicted classes
        ax3 = plt.subplot2grid((15, 1), (9, 0), rowspan=1, sharex=ax1)
        ax3.imshow([predicted_classes], aspect='auto', origin='lower', cmap='viridis', extent=[0, len(predicted_classes), 0, 1])
        ax3.set_title('Predicted Classes', fontsize=24)
        ax3.set_yticks([])

        # Calculate and plot majority vote predictions
        window_size = 100
        majority_vote_predictions = majority_vote(predicted_classes, window_size)
        ax4 = plt.subplot2grid((15, 1), (10, 0), rowspan=1, sharex=ax1)
        ax4.imshow([majority_vote_predictions], aspect='auto', origin='lower', cmap='viridis', extent=[0, len(majority_vote_predictions), 0, 1])
        ax4.set_title(f'Majority Vote Predictions (Window Size: {window_size})', fontsize=24)
        ax4.set_yticks([])

        # Plot class probability
        ax5 = plt.subplot2grid((15, 1), (11, 0), rowspan=3, sharex=ax1)
        probs = torch.sigmoid(logits[0]).cpu().detach().numpy()
        
        ax5.plot(probs, color='blue', label='Class Probability')
        ax5.set_title('Class Probability', fontsize=24)
        ax5.set_xlabel('Timebins', fontsize=24)
        ax5.set_ylabel('Probability', fontsize=24)
        ax5.set_ylim(0, 1)
        ax5.tick_params(axis='both', which='major', labelsize=18)

        # Ensure all subplots are aligned
        plt.xlim(0, spec_chunk.shape[1])

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{file}_chunk_{i}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_specs(self, num_specs):
        output_dir = "imgs/decoder_specs"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        processed_specs = 0
        files = os.listdir(self.song_dir)
        
        with tqdm(total=num_specs, desc="Generating spectrograms") as pbar:
            for file in files:
                if processed_specs >= num_specs:
                    break

                data = np.load(os.path.join(self.song_dir, file))
                spec = data['s']
                labels = data['labels']

                spec_length = spec.shape[1]
                spec_height = spec.shape[0]

                if spec_height > self.spec_height:
                    spec = spec[20:216, :]
                
                spec = self.z_score_spectrogram(spec)

                if spec_length > self.context_length:
                    num_splits = (spec_length // self.context_length) + 1
                    for i in range(num_splits):
                        if processed_specs >= num_specs:
                            break

                        if i == num_splits - 1:
                            spec_chunk = spec[:, i*self.context_length:]
                            label_chunk = labels[i*self.context_length:]
                        else:
                            spec_chunk = spec[:, i*self.context_length:(i+1)*self.context_length]
                            label_chunk = labels[i*self.context_length:(i+1)*self.context_length]

                        self.process_spec_chunk(spec_chunk, label_chunk, output_dir, file, i)
                        processed_specs += 1
                        pbar.update(1)
                else:
                    self.process_spec_chunk(spec, labels, output_dir, file, "")
                    processed_specs += 1
                    pbar.update(1)

        print(f"Generated {processed_specs} spectrograms.")

    def plot_song_statistics(self, entropy, num_songs, phrase_duration, output_dir, file):
        plt.figure(figsize=(20, 15))  # Adjusted figure size for better visibility

        # Plot total song entropy
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(entropy, 'o-', label='Total Song Entropy')
        ax1.plot(np.convolve(entropy, np.ones(10)/10, mode='valid'), 'r-', label='Smoothed Entropy')
        ax1.set_title('Total Song Entropy', fontsize=16)
        ax1.set_ylabel('Entropy', fontsize=14)
        ax1.legend()

        # Plot total number of songs sung
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(num_songs, 'o-', label='Total Number of Songs Sung')
        ax2.plot(np.convolve(num_songs, np.ones(10)/10, mode='valid'), 'r-', label='Smoothed Number of Songs')
        ax2.set_title('Total Number of Songs Sung', fontsize=16)
        ax2.set_ylabel('Number of Songs', fontsize=14)
        ax2.legend()

        # Plot average phrase duration
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(phrase_duration, 'o-', label='Average Phrase Duration')
        ax3.plot(np.convolve(phrase_duration, np.ones(10)/10, mode='valid'), 'r-', label='Smoothed Phrase Duration')
        ax3.set_title('Average Phrase Duration', fontsize=16)
        ax3.set_ylabel('Phrase Duration', fontsize=14)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{file}_song_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TweetyBert Classifier")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--bird_name", type=str, required=True, help="Name of the bird")
    parser.add_argument("--generate_loss_plot", action="store_true", default=False, help="Generate loss plot during training")

    args = parser.parse_args()

    experiment_name = args.experiment_name
    bird_name = args.bird_name
    model_dir = f"experiments/{experiment_name}"
    linear_decoder_dir = f"experiments/{bird_name}_linear_decoder"
    data_file = f"files/{bird_name}.npz"

    classifier = TweetyBertClassifier(model_dir=model_dir, linear_decoder_dir=linear_decoder_dir)
    classifier.prepare_data(data_file)
    classifier.create_dataloaders()
    classifier.create_classifier()
    classifier.train_classifier(generate_loss_plot=args.generate_loss_plot)
    classifier.save_decoder_state()
    classifier.generate_specs()