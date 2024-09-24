import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import multiprocessing
import soundfile as sf
import gc
import csv
import sys
import psutil
from tqdm import tqdm
import json
import argparse
from scipy.signal import ellip, filtfilt
import time


class WavtoSpec:
    def __init__(self, src_dir, dst_dir, song_detection_json_path=None, step_size=119, nfft=1024):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.song_detection_json_path = song_detection_json_path
        self.use_json = song_detection_json_path is not None
        self.step_size = step_size
        self.nfft = nfft

    def process_directory(self):
        audio_files = [os.path.join(root, file)
                       for root, dirs, files in os.walk(self.src_dir)
                       for file in files if file.lower().endswith('.wav')]

        skipped_files_count = 0

        for file_path in tqdm(audio_files, desc="Processing files"):
            result = self.convert_to_spectrogram(file_path, song_detection_json_path=self.song_detection_json_path, save_npz=True)
            if result is None:
                skipped_files_count += 1

        print(f"Total files processed: {len(audio_files)}")
        print(f"Total files skipped due to no vocalization data: {skipped_files_count}")

    @staticmethod
    def process_file(instance, file_path):
        return instance.convert_to_spectrogram(file_path, song_detection_json_path=None, save_npz=False)

    def convert_to_spectrogram(self, file_path, song_detection_json_path, min_length_ms=500, min_timebins=1000, save_npz=True):
        try:
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                data = wav_file.read(dtype='int16')
                if wav_file.channels > 1:
                    data = data[:, 0]

            length_in_ms = (len(data) / samplerate) * 1000
            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return None

            file_name = os.path.basename(file_path)

            if self.use_json or song_detection_json_path is not None:
                vocalization_data, phrase_labels = self.check_vocalization(file_name=file_name, data=data, samplerate=samplerate, song_detection_json_path=song_detection_json_path)
                if not vocalization_data:
                    print("file skipped due to no vocalization")
                    return None
            else:
                vocalization_data = [(0, len(data)/samplerate)]  # Assume entire file is vocalization
                phrase_labels = {}  # Empty dict if not using JSON

            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)  # Apply high-pass filter

            hop_length = self.step_size
            window = 'hann'  # Hamming window is computationally cheaper
            n_fft = self.nfft
            Sxx = librosa.stft(data.astype(float), n_fft=n_fft, hop_length=hop_length, window=window)
            Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)

            labels = np.zeros(Sxx.shape[1], dtype=int)
            for label, intervals in phrase_labels.items():
                for start_sec, end_sec in intervals:
                    start_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, start_sec)
                    end_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, end_sec)
                    labels[start_bin:end_bin] = int(label)

            if Sxx.shape[1] >= min_timebins:
                for i, (start_sec, end_sec) in enumerate(vocalization_data):
                    start_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, start_sec)
                    end_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, end_sec)

                    segment_Sxx_log = Sxx_log[:, start_bin:end_bin]
                    segment_labels = labels[start_bin:end_bin]
                    segment_vocalization = np.ones(end_bin - start_bin, dtype=int)  # All 1s since this is a vocalization segment

                    if save_npz:
                        spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                        segment_spec_file_path = os.path.join(self.dst_dir, f"{spec_filename}_segment_{i}.npz")
                        np.savez(segment_spec_file_path, 
                                            s=segment_Sxx_log, 
                                            vocalization=segment_vocalization, 
                                            labels=segment_labels)
                        print(f"Segment {i} spectrogram, vocalization data, and labels saved to {segment_spec_file_path}")

                return Sxx_log, vocalization_data, labels
            else:
                print(f"Spectrogram for {file_path} has less than {min_timebins} timebins and will not be saved.")
                return None

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def check_vocalization(self, file_name, data, samplerate, song_detection_json_path):
        if not self.use_json:
            return [(0, len(data)/samplerate)], {}  # Assume entire file is vocalization if not using JSON

        # Open JSON file
        if not os.path.exists(song_detection_json_path):
            print(f"JSON file {song_detection_json_path} does not exist.")
            return None, None

        with open(song_detection_json_path, 'r') as json_file:
            json_data = json.load(json_file)
            for entry in json_data:
                if entry['filename'] == file_name:
                    if not entry['song_present']:
                        return None, None
                    
                    onsets_offsets = [(seg['onset_ms'] / 1000, seg['offset_ms'] / 1000) for seg in entry['segments']]
                    
                    # Process phrase labels
                    phrase_labels = {}
                    if 'phrase_label onset/offsets' in row and row['phrase_label onset/offsets']:
                        try:
                            phrase_data = json.loads(row['phrase_label onset/offsets'].replace("'", '"'))
                            for label, intervals in phrase_data.items():
                                phrase_labels[label] = intervals
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON for {file_name}. Raw data: {row['phrase_label onset/offsets']}")
                    
                    return onsets_offsets, phrase_labels

        print(f"No matching entry found for {file_name} in {song_detection_json_path}.")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to spectrograms.")
    parser.add_argument('--src_dir', type=str, help='Source directory containing WAV files.')
    parser.add_argument('--dst_dir', type=str, help='Destination directory to save spectrograms.')
    parser.add_argument('--song_detection_json_path', type=str, default=None, help='Path to the JSON file with song detection data.')
    parser.add_argument('--step_size', type=int, default=119, help='Step size for the spectrogram.')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of FFT points for the spectrogram.')

    args = parser.parse_args()

    wav_to_spec = WavtoSpec(args.src_dir, args.dst_dir, args.song_detection_json_path, args.step_size, args.nfft)
    wav_to_spec.process_directory()

if __name__ == "__main__":
    main()
