import numpy as np
import os
import shutil
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_class import SongDataSet_Image, CollateFunction
from utils import load_model
from linear_probe import LinearProbeModel, LinearProbeTrainer
from tqdm import tqdm
import json

class TweetyBertClassifier:
    def __init__(self, config_path, weights_path, linear_decoder_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tweety_bert_model = self.load_tweety_bert(config_path, weights_path)
        self.linear_decoder_dir = linear_decoder_dir
        self.train_dir = os.path.join(linear_decoder_dir, "train")
        self.test_dir = os.path.join(linear_decoder_dir, "test")
        self.num_classes = None
        self.config_path = config_path
        self.weights_path = weights_path
        self.data_file = None

    def load_tweety_bert(self, config_path, weights_path):
        return load_model(config_path, weights_path)

    def prepare_data(self, data_file, test_train_split=0.8, length=1000):
        self.data_file = data_file
        data = np.load(data_file)
        vocalization = data['vocalization']
        labels = data['hdbscan_labels']
        specs = data['original_spectogram']

        specs = np.pad(specs, ((0, 0), (20, 0)), 'constant', constant_values=0)
        
        temp = np.full_like(vocalization, -1)
        indexes = np.where(vocalization == 1.0)[0]
        temp[indexes] = labels
        labels = temp + 1

        self.num_classes = len(np.unique(labels))
        print(f"Number of classes: {self.num_classes}")

        list_of_data = [
            (labels[i:i+length], specs[i:i+length], vocalization[i:i+length])
            for i in range(0, len(labels), length)
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

    def create_dataloaders(self, batch_size=48, segment_length=1000):
        collate_fn = CollateFunction(segment_length=segment_length)

        train_dataset = SongDataSet_Image(self.train_dir, num_classes=self.num_classes, infinite_loader=False)
        test_dataset = SongDataSet_Image(self.test_dir, num_classes=self.num_classes, infinite_loader=False)

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
            layer_num=-2, 
            layer_id="attention_output", 
            classifier_dims=196
        ).to(self.device)

    def train_classifier(self, lr=1e-5, batches_per_eval=100, desired_total_batches=4000, patience=8, generate_loss_plot=False):
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

    def generate_specs(self, num_specs=100, context_length=1000):
        spec_generator = SpecGenerator(self.classifier_model, self.test_dir, context_length)
        spec_generator.generate_specs(num_specs)

    def save_decoder_state(self):
        save_dir = os.path.join(self.linear_decoder_dir, "decoder_state")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save decoder weights
        torch.save(self.classifier_model.state_dict(), os.path.join(save_dir, "decoder_weights.pth"))

        # Save configuration
        config = {
            "num_classes": self.num_classes,
            "tweety_bert_config": self.config_path,
            "tweety_bert_weights": self.weights_path,
            "data_file": self.data_file,
        }
        with open(os.path.join(save_dir, "decoder_config.json"), "w") as f:
            json.dump(config, f)

        # Copy the original data file
        shutil.copy2(self.data_file, os.path.join(save_dir, "original_data.npz"))

        print(f"Decoder state saved in {save_dir}")

    @classmethod
    def load_decoder_state(cls, linear_decoder_dir):
        save_dir = os.path.join(linear_decoder_dir, "decoder_state")
        
        # Load configuration
        with open(os.path.join(save_dir, "decoder_config.json"), "r") as f:
            config = json.load(f)

        # Create instance
        instance = cls(config["tweety_bert_config"], config["tweety_bert_weights"], linear_decoder_dir)
        
        # Set attributes
        instance.num_classes = config["num_classes"]
        instance.data_file = config["data_file"]

        # Create and load classifier
        instance.create_classifier()
        instance.classifier_model.load_state_dict(torch.load(os.path.join(save_dir, "decoder_weights.pth")))

        print(f"Decoder state loaded from {save_dir}")
        return instance

class SpecGenerator:
    def __init__(self, model, song_dir, context_length=1000):
        self.model = model
        self.song_dir = song_dir
        self.context_length = context_length
        self.spec_height = 196
        self.device = next(model.parameters()).device
        self.num_classes = model.num_classes

    def z_score_spectrogram(self, spec):
        spec_mean = np.mean(spec)
        spec_std = np.std(spec)
        return (spec - spec_mean) / spec_std

    def process_spec_chunk(self, spec_chunk, label_chunk, output_dir, file, i):
        spec_tensor = torch.Tensor(spec_chunk).to(self.device).unsqueeze(0).unsqueeze(0)
        logits = self.model.forward(spec_tensor)
        predicted_classes = torch.argmax(logits, dim=2).cpu().detach().numpy().flatten()
        self.plot_and_save(spec_chunk, predicted_classes, label_chunk, logits, output_dir, file, i)

    def plot_and_save(self, spec_chunk, predicted_classes, label_chunk, logits, output_dir, file, i):
        color_map = plt.get_cmap('viridis')
        class_colors = color_map(predicted_classes / self.num_classes)

        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(hspace=1.5, left=0, right=1, top=1, bottom=0)
        plt.subplot2grid((15, 1), (0, 0), rowspan=8)
        plt.imshow(spec_chunk, aspect='auto', origin='lower')
        plt.title('Spectrogram', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.subplot2grid((15, 1), (8, 0), rowspan=1)
        plt.imshow([predicted_classes], aspect='auto', origin='lower', cmap='viridis')
        plt.title('Predicted Classes', fontsize=24)
        plt.yticks([])
        plt.xticks([])
        plt.subplot2grid((15, 1), (9, 0), rowspan=1)
        plt.imshow([label_chunk], aspect='auto', origin='lower', cmap='viridis')
        plt.title('True Labels', fontsize=24)
        plt.yticks([])
        plt.xticks([])
        plt.subplot2grid((15, 1), (10, 0), rowspan=4)
        plt.imshow(logits[0].T.cpu().detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title('Logits Heatmap', fontsize=24)
        plt.xlabel('Timebins', fontsize=24)
        plt.ylabel('Logits', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(f"{output_dir}/{file}_chunk_{i}.png", bbox_inches='tight')
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

# Usage example:
if __name__ == "__main__":
    classifier = TweetyBertClassifier(
        config_path="experiments/PitchShiftTest/config.json",
        weights_path="experiments/PitchShiftTest/saved_weights/model_step_12500.pth",
        linear_decoder_dir="/media/george-vengrovski/disk1/linear_decoder_test"
    )

    classifier.prepare_data("/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_for_training_classifier.npz")
    classifier.create_dataloaders()
    classifier.create_classifier()
    classifier.train_classifier(generate_loss_plot=False)
    classifier.save_decoder_state()
    classifier.generate_specs()

    # loaded_classifier = TweetyBertClassifier.load_decoder_state("/media/george-vengrovski/disk1/linear_decoder_test")
    # loaded_classifier.generate_specs()