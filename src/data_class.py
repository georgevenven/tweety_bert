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

def determine_number_unique_classes(dir):
    unique_labels = set()

    for file in os.listdir(dir):
        data = np.load(os.path.join(dir, file), allow_pickle=True)
        labels = np.unique(data['labels'])

        # add to set
        unique_labels.update(labels)

    return len(unique_labels)
    

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=40, infinite_loader=True, segment_length=1000, pitch_shift=False, min_length=100):
        self.file_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir)]
        self.num_classes = num_classes
        self.infinite_loader = infinite_loader
        self.segment_length = segment_length
        self.pitch_shift = pitch_shift
        self.min_length = min_length  # Add min_length parameter

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
            idx = random.randint(0, len(self.file_paths) - 1)
        file_path = self.file_paths[idx]

        try:
            # Load data and preprocess
            data = np.load(file_path, allow_pickle=True)
            file_name = os.path.basename(file_path)
            spectrogram = data['s']
            
            # Skip files if the spectrogram length is less than min_length
            if spectrogram.shape[0] < self.min_length:
                if self.infinite_loader:
                    return self.__getitem__(random.randint(0, len(self.file_paths) - 1))
                else:
                    raise ValueError(f"Spectrogram length {spectrogram.shape[0]} is less than min_length {self.min_length}")

            spectrogram = spectrogram[20:216]
            # Calculate mean and standard deviation of the spectrogram
            spec_mean = np.mean(spectrogram)
            spec_std = np.std(spectrogram) + 1e-8  # Add epsilon to avoid division by zero
            # Z-score the spectrogram
            spectrogram = (spectrogram - spec_mean) / spec_std

            if self.pitch_shift:
                spectrogram = self.apply_pitch_shift(spectrogram)

            # Process labels
            ground_truth_labels = np.array(data['labels'], dtype=int)
            vocalization = np.array(data['vocalization'], dtype=int)
            
            ground_truth_labels = torch.from_numpy(ground_truth_labels).long().squeeze(0)
            spectrogram = torch.from_numpy(spectrogram).float().permute(1, 0)
            ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()
            vocalization = torch.from_numpy(vocalization).long()

            # Shapes Here #
            # spectrogram: torch.Size([timebins, freqbins])
            # ground_truth_labels: torch.Size([timebins, classes])
            # vocalization: torch.Size([timebins])

            if self.segment_length is not None:
                # Truncate if larger than context window
                if spectrogram.shape[0] > self.segment_length:
                    starting_points_range = spectrogram.shape[0] - self.segment_length        
                    start = torch.randint(0, starting_points_range, (1,)).item()  
                    end = start + self.segment_length     

                    spectrogram = spectrogram[start:end]
                    ground_truth_labels = ground_truth_labels[start:end]
                    vocalization = vocalization[start:end]

                elif spectrogram.shape[0] < self.segment_length:
                    pad_amount = self.segment_length - spectrogram.shape[0]

                    # Pad spectrogram: [timebins, freqbins]
                    spectrogram = F.pad(spectrogram, (0, 0, 0, pad_amount), mode='constant', value=0)

                    # Pad ground_truth_labels: [timebins, classes]
                    ground_truth_labels = F.pad(ground_truth_labels, (0, 0, 0, pad_amount), mode='constant', value=0)
                    # Pad vocalization: [timebins]
                    vocalization = F.pad(vocalization, (0, pad_amount), mode='constant', value=0)
                    
            return spectrogram, ground_truth_labels, vocalization, file_name

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