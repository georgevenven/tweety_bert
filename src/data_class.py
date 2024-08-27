import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import random
import logging

def deterimine_number_unique_classes(dir):
    unique_labels = set()

    for file in os.listdir(dir):
        data = np.load(os.path.join(dir, file), allow_pickle=True)
        labels = np.unique(data['labels'])

        # add to set
        unique_labels.update(labels)

    return len(unique_labels)
    

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=40, infinite_loader=True, segment_length=1000, pitch_shift=False):
        self.file_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]
        self.num_classes = num_classes
        self.infinite_loader = infinite_loader
        self.segment_length = segment_length
        self.pitch_shift = pitch_shift

    def apply_pitch_shift(self, spectrogram):
        # Shift the pitch of the spectrogram according to a normal distribution
        shift_amount = random.randint(-50, 50)
        
        # Create an array of zeros with the same shape as the spectrogram
        shifted_spectrogram = np.zeros_like(spectrogram)
        
        if shift_amount > 0:
            shifted_spectrogram[shift_amount:] = spectrogram[:-shift_amount]
        elif shift_amount < 0:
            shifted_spectrogram[:shift_amount] = spectrogram[-shift_amount:]
        else:
            shifted_spectrogram = spectrogram
        
        return shifted_spectrogram

    def __getitem__(self, idx):
        if self.infinite_loader:
            original_idx = idx  # Store the original index for logging
            idx = random.randint(0, len(self.file_paths) - 1)
        file_path = self.file_paths[idx]

        try:
            # Load data and preprocess
            data = np.load(file_path, allow_pickle=True)
            file_name = os.path.basename(file_path)
            spectogram = data['s']
            
            spectogram = spectogram[20:216]
            # Calculate mean and standard deviation of the spectrogram
            spec_mean = np.mean(spectogram)
            spec_std = np.std(spectogram) + 1e-8  # Add epsilon to avoid division by zero
            # Z-score the spectrogram
            spectogram = (spectogram - spec_mean) / spec_std

            if self.pitch_shift:
                spectogram = self.apply_pitch_shift(spectogram)

            # Process labels
            ground_truth_labels = np.array(data['labels'], dtype=int)
            vocalization = np.array(data['vocalization'], dtype=int)
            
            ground_truth_labels = torch.from_numpy(ground_truth_labels).long().squeeze(0)
            spectogram = torch.from_numpy(spectogram).float().permute(1, 0)
            ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()
            vocalization = torch.from_numpy(vocalization).long()

            if self.segment_length is not None:
                # Truncate if larger than context window
                if spectogram.shape[0] > self.segment_length:
                    # Get random view of size segment
                    # Find range of valid starting pts (essentially these are the possible starting pts for the length to equal segment window)
                    starting_points_range = spectogram.shape[0] - self.segment_length        
                    start = torch.randint(0, starting_points_range, (1,)).item()  
                    end = start + self.segment_length     

                    spectogram = spectogram[start:end]
                    ground_truth_labels = ground_truth_labels[start:end]
                    vocalization = vocalization[start:end]

                # Pad with 0s if shorter
                if spectogram.shape[0] < self.segment_length:
                    pad_amount = self.segment_length - spectogram.shape[0]
                    spectogram = F.pad(spectogram, (0, 0, 0, pad_amount), 'constant', 0)
                    ground_truth_labels = F.pad(ground_truth_labels, (0, pad_amount), 'constant', 0)
                    vocalization = F.pad(vocalization, (0, pad_amount), 'constant', 0)


            return spectogram, ground_truth_labels, vocalization, file_name

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # Recursively call __getitem__ with a different index if in infinite loader mode
            if self.infinite_loader:
                return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
            else:
                raise e
    
    def __len__(self):
        if self.infinite_loader:
            # Return an arbitrarily large number to simulate an infinite dataset
            return int(1e12)
        else:
            return len(self.file_paths)

class CollateFunction:
    def __init__(self, segment_length=1000):
        pass
    def __call__(self, batch):
        # Unzip the batch (a list of (spectogram, ground_truth_labels, vocalization, file_path) tuples)
        spectograms, ground_truth_labels, vocalization, file_names = zip(*batch)

        # Stack tensors along a new dimension to match the BERT input size.
        spectograms = torch.stack(spectograms, dim=0)
        vocalization = torch.stack(vocalization, dim=0)
        ground_truth_labels = torch.stack(ground_truth_labels, dim=0)
        # Keep file_paths as a list
        # Final reshape for model
        spectograms = spectograms.unsqueeze(1).permute(0,1,3,2)
        return spectograms, ground_truth_labels, vocalization, file_names