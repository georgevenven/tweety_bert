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
from spectogram_generator import WavtoSpec  # Add this import
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch import nn
from torch.nn import functional as F
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

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

    def train_classifier(self, lr=1e-5, batches_per_eval=100, desired_total_batches=500, patience=8, generate_loss_plot=False):
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


class TweetyBertInference:
    def __init__(self, classifier_path, spec_dst_folder):
        self.classifier = TweetyBertClassifier.load_decoder_state(classifier_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wav_to_spec = None
        self.spec_dst_folder = spec_dst_folder
        
        # Create the spectrogram destination folder if it doesn't exist
        os.makedirs(self.spec_dst_folder, exist_ok=True)
        
        # Generate color palette
        base_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        additional_colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, 8))
        colors = np.vstack((base_colors, additional_colors))
        colors = colors[np.random.permutation(len(colors))]
        colors = np.vstack(([0, 0, 0, 1], colors))  # Add black at the beginning for silence
        self.cmap = mcolors.ListedColormap(colors[:self.classifier.num_classes])

    def setup_wav_to_spec(self, folder, csv_file_dir=None):
        self.wav_to_spec = WavtoSpec(folder, self.spec_dst_folder, csv_file_dir)

    def inference_data_class(self, data, context_length=1000):
        recording_length = data[1].shape[1]

        spectogram = data[1][20:216]
        spec_mean = np.mean(spectogram)
        spec_std = np.std(spectogram)
        spectogram = (spectogram - spec_mean) / spec_std

        ground_truth_labels = np.array(data[0], dtype=int)
        vocalization = np.array(data[2], dtype=int)
        
        ground_truth_labels = torch.from_numpy(ground_truth_labels).long().squeeze(0)
        spectogram = torch.from_numpy(spectogram).float().permute(1, 0)
        ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.classifier.num_classes).float()
        vocalization = torch.from_numpy(vocalization).long()

        pad_amount = context_length - (recording_length % context_length)
        if recording_length < context_length:
            pad_amount = context_length - recording_length
         
        if recording_length > context_length and pad_amount != 0:
            pad_amount = context_length - (spectogram.shape[0] % context_length)
            spectogram = F.pad(spectogram, (0, 0, 0, pad_amount), 'constant', 0)
            ground_truth_labels = F.pad(ground_truth_labels, (0, 0, 0, pad_amount), 'constant', 0)
            vocalization = F.pad(vocalization, (0, pad_amount), 'constant', 0)

        spectogram = spectogram.reshape(spectogram.shape[0] // context_length, context_length, spectogram.shape[1])
        ground_truth_labels = ground_truth_labels.reshape(ground_truth_labels.shape[0] // context_length, context_length, ground_truth_labels.shape[1])
        vocalization = vocalization.reshape(vocalization.shape[0] // context_length, context_length)

        return spectogram, ground_truth_labels, vocalization

    def max_vote(self, predicted_labels):
        processed_labels = predicted_labels.copy()
        
        zero_indices = np.where(predicted_labels == 0)[0]
        non_zero_indices = np.where(predicted_labels != 0)[0]
        
        regions = []
        current_region = []
        
        for i in range(len(non_zero_indices)):
            if i == 0 or non_zero_indices[i] - non_zero_indices[i-1] == 1:
                current_region.append(non_zero_indices[i])
            else:
                if current_region:
                    regions.append(current_region)
                current_region = [non_zero_indices[i]]
        
        if current_region:
            regions.append(current_region)
        
        for region in regions:
            start = region[0]
            end = region[-1]
            
            if (start == 0 or predicted_labels[start-1] == 0) and \
               (end == len(predicted_labels)-1 or predicted_labels[end+1] == 0):
                
                region_labels = predicted_labels[start:end+1]
                max_vote_label = np.bincount(region_labels).argmax()
                
                processed_labels[start:end+1] = max_vote_label
        
        return processed_labels

    def convert_to_onset_offset(self, labels):
        sampling_rate = 44100 
        NFFT = 1024
        hop_length = NFFT // 2
        ms_per_timebin = (hop_length / sampling_rate) * 1000

        onsets = []
        offsets = []
        current_label = 0

        for i, label in enumerate(labels):
            if label != current_label:
                if current_label != 0:
                    offsets.append(i * ms_per_timebin)
                if label != 0:
                    onsets.append(i * ms_per_timebin)
                current_label = label

        if current_label != 0:
            offsets.append(len(labels) * ms_per_timebin)

        return list(zip(onsets, offsets))

    def visualize_spectrogram(self, spec, predicted_labels, file_name):
        plt.figure(figsize=(15, 10))
        
        # Plot spectrogram
        plt.subplot(2, 1, 1)
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title('Spectrogram', fontsize=24)
        plt.colorbar(format='%+2.0f dB')
        
        # Plot predicted labels
        plt.subplot(2, 1, 2)
        im = plt.imshow([predicted_labels], aspect='auto', origin='lower', cmap=self.cmap, 
                        vmin=0, vmax=self.classifier.num_classes-1)
        plt.title('Predicted Labels', fontsize=24)
        cbar = plt.colorbar(im, ticks=range(self.classifier.num_classes))
        cbar.set_label('Syllable Class')
        cbar.set_ticklabels(['Silence'] + [f'Class {i}' for i in range(1, self.classifier.num_classes)])
        
        plt.tight_layout()
        output_path = os.path.join(self.spec_dst_folder, f"{os.path.splitext(file_name)[0]}_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def process_file(self, file_path, visualize=False):
        spec, vocalization, labels = self.wav_to_spec.process_file(self.wav_to_spec, file_path=file_path)

        spectogram, _, _ = self.inference_data_class((labels, spec, vocalization), context_length=1000)
        spec_tensor = torch.Tensor(spectogram).to(self.device).unsqueeze(1)

        logits = self.classifier.classifier_model(spec_tensor.permute(0,1,3,2))
        logits = logits.reshape(logits.shape[0] * logits.shape[1], -1)

        predicted_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
        post_processed_labels = self.max_vote(predicted_labels)

        onsets_offsets = self.convert_to_onset_offset(post_processed_labels)
        
        song_present = len(onsets_offsets) > 0

        if visualize:
   
            self.visualize_spectrogram(spectogram.flatten(0,1).T, post_processed_labels, os.path.basename(file_path))

        return {
            "file_name": file_path,
            "song_present": song_present,
            "syllable_onsets/offsets": onsets_offsets
        }
        
    def process_folder(self, folder_path, visualize=False):
        results = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    file_path = os.path.join(root, file)
                    try:
                        result = self.process_file(file_path, visualize)
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return results

    def save_results(self, results, output_path):
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# Usage example:
if __name__ == "__main__":
    # classifier = TweetyBertClassifier(
    #     config_path="experiments/PitchShiftTest/config.json",
    #     weights_path="experiments/PitchShiftTest/saved_weights/model_step_12500.pth",
    #     linear_decoder_dir="/media/george-vengrovski/disk1/linear_decoder"
    # )

    # classifier.prepare_data("files/labels_for_training_classifier.npz")
    # classifier.create_dataloaders()
    # classifier.create_classifier()
    # classifier.train_classifier(generate_loss_plot=False)
    # classifier.save_decoder_state()
    # classifier.generate_specs()

    # loaded_classifier = TweetyBertClassifier.load_decoder_state("/media/george-vengrovski/disk1/linear_decoder_test")
    # loaded_classifier.generate_specs()

    classifier_path = "/media/george-vengrovski/disk1/linear_decoder"
    folder_path = "/media/george-vengrovski/Extreme SSD/20240726_All_Area_X_Lesions/USA5288"
    output_path = "/media/george-vengrovski/Extreme SSD/20240726_All_Area_X_Lesions/USA5288_specs/song_database.csv"
    spec_dst_folder = "/media/george-vengrovski/Extreme SSD/20240726_All_Area_X_Lesions/USA5288_specs"

    inference = TweetyBertInference(classifier_path, spec_dst_folder)
    inference.setup_wav_to_spec(folder_path)
    
    results = inference.process_folder(folder_path, visualize=True)
    inference.save_results(results, output_path)