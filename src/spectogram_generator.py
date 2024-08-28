import os
import numpy as np
from scipy.signal import spectrogram, windows, ellip, filtfilt
import matplotlib.pyplot as plt
import multiprocessing
import soundfile as sf
import gc
import csv
import sys
import psutil
from tqdm import tqdm
import json

class WavtoSpec:
    def __init__(self, src_dir, dst_dir, csv_file_dir=None):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.csv_file_dir = csv_file_dir
        self.use_csv = csv_file_dir is not None

    def process_directory(self):
        audio_files = [os.path.join(root, file)
                       for root, dirs, files in os.walk(self.src_dir)
                       for file in files if file.lower().endswith('.wav')]

        skipped_files_count = 0

        for file_path in tqdm(audio_files, desc="Processing files"):
            result = self.convert_to_spectrogram(file_path, csv_file_dir=self.csv_file_dir, save_npz=True)
            if result is None:
                skipped_files_count += 1

        print(f"Total files processed: {len(audio_files)}")
        print(f"Total files skipped due to no vocalization data: {skipped_files_count}")

    @staticmethod
    def process_file(instance, file_path):
        return instance.convert_to_spectrogram(file_path, csv_file_dir=None, save_npz=False)

    def convert_to_spectrogram(self, file_path, csv_file_dir, min_length_ms=1025, min_timebins=250, save_npz=True):
        try:
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                data = wav_file.read(dtype='int16')
                if wav_file.channels > 1:
                    data = data[:, 0]
            
            # Skip small files (less than 1 second)
            length_in_ms = (len(data) / samplerate) * 1000
            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return None
            
            file_name = os.path.basename(file_path)
            
            # Check if there is vocalization in the file and get phrase labels
            if self.use_csv or csv_file_dir is not None:
                vocalization_data, phrase_labels = self.check_vocalization(file_name=file_name, data=data, samplerate=samplerate, csv_file_dir=csv_file_dir)
                if not vocalization_data:
                    print("file skipped due to no vocalization")
                    return None
            else:
                vocalization_data = [(0, len(data)/samplerate)]  # Assume entire file is vocalization
                phrase_labels = {}  # Empty dict if not using CSV
            
            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)
            
            NFFT = 1024
            step_size = 119
            overlap_samples = NFFT - step_size
            window = windows.gaussian(NFFT, std=NFFT/8)
            
            f, t, Sxx = spectrogram(data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)
            Sxx_log = 10 * np.log10(Sxx + 1e-6)
            
            # Convert phrase labels to timebins
            labels = np.zeros(t.size, dtype=int)
            for label, intervals in phrase_labels.items():
                for start_sec, end_sec in intervals:
                    start_bin = np.searchsorted(t, start_sec)
                    end_bin = np.searchsorted(t, end_sec)
                    labels[start_bin:end_bin] = int(label)
            
            if t.size >= min_timebins:
                for i, (start_sec, end_sec) in enumerate(vocalization_data):
                    start_bin = np.searchsorted(t, start_sec)
                    end_bin = np.searchsorted(t, end_sec)
                    
                    segment_Sxx_log = Sxx_log[:, start_bin:end_bin]
                    segment_labels = labels[start_bin:end_bin]
                    segment_vocalization = np.ones(end_bin - start_bin, dtype=int)  # All 1s since this is a vocalization segment
                    
                    if save_npz:
                        spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                        segment_spec_file_path = os.path.join(self.dst_dir, f"{spec_filename}_segment_{i}.npz")
                        np.savez_compressed(segment_spec_file_path, 
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
        
        finally:
            plt.close('all')
            gc.collect()

    def check_vocalization(self, file_name, data, samplerate, csv_file_dir):
        if not self.use_csv:
            return [(0, len(data)/samplerate)], {}  # Assume entire file is vocalization if not using CSV

        # Open csv file
        csv_file_path = os.path.join(csv_file_dir)
        if not os.path.exists(csv_file_path):
            print(f"CSV file {csv_file_path} does not exist.")
            return None, None

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['file_name'] == file_name:
                    onset_offset_list = eval(row['onset/offset'])
                    onsets_offsets = [(onset, offset) for onset, offset in onset_offset_list]
                    
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

        print(f"No matching row found for {file_name} in {csv_file_path}.")
        return None, None

def main():
    src_dir = '/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5271'
    dst_dir = '/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5271_no_threshold_no_norm'
    csv_file_dir = '/home/george-vengrovski/Documents/tweety_bert/files/5271_Whisperseg.csv'  # Set to None if not using a CSV file

    wav_to_spec = WavtoSpec(src_dir, dst_dir, csv_file_dir)
    wav_to_spec.process_directory()
    
if __name__ == "__main__":
    main()
