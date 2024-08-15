import os
import sys
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Finding WhisperSeg
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
whisperseg_dir = os.path.join(parent_dir, 'WhisperSeg')
sys.path.append(whisperseg_dir)

from model import WhisperSegmenterFast

def simple_visualizer(audio, sr, prediction, output_path):
    # Create spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    
    # Plot spectrogram
    plt.figure(figsize=(50, 5))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    
    # Add vocalization blocks
    for start, end in zip(prediction['onset'], prediction['offset']):
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.title('Spectrogram with Vocalization Blocks')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

class WhisperSegProcessor:
    def __init__(self, root_dir: str, output_csv: str, save_spectrograms: bool = False, delete_existing_csv: bool = False):
        self.root_dir = root_dir
        self.output_csv = output_csv
        self.save_spectrograms = save_spectrograms
        self.delete_existing_csv = delete_existing_csv
        self.segmenter = WhisperSegmenterFast("nccratliri/whisperseg-canary-ct2", device="cpu")
        self.min_frequency = 0
        self.spec_time_step = 0.001
        self.min_segment_length = 0.005
        self.eps = 0.01
        self.num_trials = 3
        self.initialize_csv()
        self.processed_files = set()
        self.load_processed_files()

    def initialize_csv(self):
        if self.delete_existing_csv and os.path.exists(self.output_csv):
            os.remove(self.output_csv)
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["file_name", "onset/offset"])

    def load_processed_files(self):
        if os.path.exists(self.output_csv):
            with open(self.output_csv, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    self.processed_files.add(row[0])

    def process_wav_file(self, file_path: str) -> List[Tuple[float, float]]:
        # Load audio at native sampling rate
        audio, sr = librosa.load(file_path, sr=None)

        # Generate prediction
        prediction = self.segmenter.segment(
            audio,
            sr=sr,
            min_frequency=self.min_frequency,
            spec_time_step=self.spec_time_step,
            min_segment_length=self.min_segment_length,
            eps=self.eps,
            num_trials=self.num_trials
        )

        # Extract onset and offset times
        segments = [(onset, offset) for onset, offset, _ in zip(prediction['onset'], prediction['offset'], prediction['cluster'])]

        # Save spectrogram if enabled
        if self.save_spectrograms:
            spec_dir = os.path.join(current_dir, 'imgs')
            os.makedirs(spec_dir, exist_ok=True)
            spec_file = os.path.join(spec_dir, f"{os.path.basename(file_path)}.png")
            simple_visualizer(audio, sr, prediction, spec_file)

        return segments

    def process_directory(self):
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    file_name = os.path.basename(file_path)
                    if file_name not in self.processed_files:
                        segments = self.process_wav_file(file_path)
                        self.save_csv_database(file_name, segments)
                        print(f"Processed: {file_name}")
                    else:
                        print(f"Skipped (already processed): {file_name}")

    def save_csv_database(self, file_name: str, segments: List[Tuple[float, float]]):
        with open(self.output_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([file_name, segments])
        self.processed_files.add(file_name)

def main():
    root_dir = "/media/george-vengrovski/disk2/canary/sorted_1/USA5326"  
    output_csv = "files/segmentation_results.csv"
    save_spectrograms = True  # Set to True if you want to save spectrograms
    delete_existing_csv = False  # Set to True if you want to delete existing CSV and start fresh

    processor = WhisperSegProcessor(root_dir, output_csv, save_spectrograms, delete_existing_csv)
    processor.process_directory()

if __name__ == "__main__":
    main()