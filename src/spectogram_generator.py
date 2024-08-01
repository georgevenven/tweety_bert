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

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = [pool.apply_async(self.process_file, args=(self, file_path, self.csv_file_dir)) 
                       for file_path in tqdm(audio_files, desc="Processing files")]
            for result in results:
                result.get()

    @staticmethod
    def process_file(instance, file_path, csv_file_dir):
        instance.convert_to_spectrogram(file_path, csv_file_dir)

    def convert_to_spectrogram(self, file_path, csv_file_dir, min_length_ms=1025, min_timebins=250):
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
                return

            folder_file_name = '/'.join(file_path.split('/')[-2:])
     
            # Check if there is vocalization in the file
            if self.use_csv:
                vocalization_data = self.check_vocalization(folder_file_name=folder_file_name, data=data, samplerate=samplerate, csv_file_dir=csv_file_dir)
                if vocalization_data is None:
                    print(f"No vocalization data found for {folder_file_name}. Skipping spectrogram generation.")
                    return
            else:
                vocalization_data = [(0, len(data))]  # Assume entire file is vocalization

            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)

            NFFT = 1024
            step_size = 119
            overlap_samples = NFFT - step_size
            window = windows.gaussian(NFFT, std=NFFT/8)

            f, t, Sxx = spectrogram(data, fs=samplerate, window=window, nperseg=NFFT, noverlap=overlap_samples)
            Sxx_log = 10 * np.log10(Sxx + 1e-6)
            Sxx_log_clipped = np.clip(Sxx_log, a_min=-2, a_max=None)
            Sxx_log_normalized = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))

            # Convert vocalization data to timebins
            vocalization_timebins = np.zeros(t.size, dtype=int)
            for start_sample, end_sample in vocalization_data:
                start_time = start_sample / samplerate
                end_time = end_sample / samplerate
                start_bin = np.searchsorted(t, start_time)
                end_bin = np.searchsorted(t, end_time)
                vocalization_timebins[start_bin:end_bin] = 1

            # Create labels array with all zeros
            labels = np.zeros(t.size, dtype=int)

            if t.size >= min_timebins:
                spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                spec_file_path = os.path.join(self.dst_dir, spec_filename + '.npz')
                np.savez_compressed(spec_file_path, s=Sxx_log_normalized, vocalization=vocalization_timebins, labels=labels)
                print(f"Spectrogram, vocalization data, and labels saved to {spec_file_path}")
            else:
                print(f"Spectrogram for {file_path} has less than {min_timebins} timebins and will not be saved.")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        finally:
            plt.close('all')
            gc.collect()

    def check_vocalization(self, folder_file_name, data, samplerate, csv_file_dir):
        if not self.use_csv:
            return [(0, len(data))]  # Assume entire file is vocalization if not using CSV

        # Open csv file
        csv_file_path = os.path.join(csv_file_dir)
        if not os.path.exists(csv_file_path):
            print(f"CSV file {csv_file_path} does not exist.")
            return None

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['file_name'] == folder_file_name:
                    onset_offset_list = eval(row['onset/offset'])
                    sample_list = [(int(onset * samplerate), int(offset * samplerate)) for onset, offset in onset_offset_list]
                    return sample_list

        print(f"No matching row found for {folder_file_name} in {csv_file_path}.")
        return None

    def get_segments_to_process(self, song_name, csv_file_dir, samplerate):
        if not self.use_csv:
            return None  # Not applicable when not using CSV

        segments_to_process = []
        csv_file_path = os.path.join(csv_file_dir, song_name + '.csv')
        if not os.path.exists(csv_file_path):
            print(f"CSV file {csv_file_path} does not exist.")
            return None

        with open(csv_file_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                start_ms = int(row['start_ms'])
                end_ms = int(row['end_ms'])
                start_sample = int(start_ms * samplerate / 1000)
                end_sample = int(end_ms * samplerate / 1000)
                segments_to_process.append((start_sample, end_sample))

        return segments_to_process


def main():
    src_dir = '/media/george-vengrovski/Extreme SSD/RHV_raw_recordings/USA5288'
    dst_dir = '/media/george-vengrovski/disk1/5288_specs'
    csv_file_dir = '/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/segmentation_results.csv'  # Set to None if not using a CSV file

    wav_to_spec = WavtoSpec(src_dir, dst_dir, csv_file_dir)
    wav_to_spec.process_directory()

if __name__ == "__main__":
    main()