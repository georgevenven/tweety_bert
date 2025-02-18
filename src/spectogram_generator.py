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
from multiprocessing import Pool, TimeoutError
import logging

class WavtoSpec:
    def __init__(self, src_dir, dst_dir, song_detection_json_path=None, step_size=119, nfft=1024, generate_random_files_number=None, single_threaded=False):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.song_detection_json_path = None if song_detection_json_path == "None" else song_detection_json_path
        self.use_json = self.song_detection_json_path is not None
        self.step_size = step_size
        self.nfft = nfft
        self.generate_random_files_number = generate_random_files_number
        self.single_threaded = single_threaded
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename='error_log.log',
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_directory(self):
        print("Starting process_directory")
        print(self.src_dir)

        audio_files = [os.path.join(root, file)
                    for root, dirs, files in os.walk(self.src_dir)
                    for file in files if file.lower().endswith('.wav')]

        
        # If more random files are requested than available, process all files
        if self.generate_random_files_number is not None and self.generate_random_files_number > len(audio_files):
            print(f"Requested {self.generate_random_files_number} random files, but only {len(audio_files)} available. Processing all files.")
            self.generate_random_files_number = None

        if self.generate_random_files_number is not None:
            print(f"Selecting {self.generate_random_files_number} random files")
            audio_files = np.random.choice(audio_files, self.generate_random_files_number, replace=False)
            print(f"{len(audio_files)} random files selected")

        if self.single_threaded:
            print("Running in single-threaded mode")
            pbar = tqdm(total=len(audio_files), desc="Processing files")
            failed_files = []
            skipped_files_count = 0

            for file_path in audio_files:
                try:
                    result = self.safe_process_file(self, file_path)
                    if result is None:
                        skipped_files_count += 1
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    failed_files.append(file_path)
                pbar.update(1)

            pbar.close()
        else:
            print(f"Running in multi-threaded mode using {multiprocessing.cpu_count()} processes")
            max_processes = multiprocessing.cpu_count()
            print(f"Using {max_processes} processes")

            manager = multiprocessing.Manager()
            skipped_files_count = manager.Value('i', 0)
            failed_files = manager.list()

            pbar = tqdm(total=len(audio_files), desc="Processing files")

            def update_progress_bar(_):
                pbar.update()

            def error_callback(e):
                logging.error(f"Error: {e}")
                skipped_files_count.value += 1
                pbar.update()

            def file_failed_callback(file_path):
                logging.error(f"File failed: {file_path}")
                failed_files.append(file_path)
                pbar.update()

            with Pool(processes=max_processes, maxtasksperchild=100) as pool:  # Adjust maxtasksperchild as needed
                for file_path in audio_files:
                    while True:
                        available_memory = psutil.virtual_memory().available
                        current_memory_usage = psutil.Process(os.getpid()).memory_info().rss
                        if available_memory > current_memory_usage:
                            break
                        print("Not enough memory to spawn new process, waiting...")
                        time.sleep(1)
                    
                    print(f"Processing file: {file_path}")
                    pool.apply_async(
                        WavtoSpec.safe_process_file,
                        args=(self, file_path),
                        callback=update_progress_bar,
                        error_callback=lambda e, fp=file_path: file_failed_callback(fp)
                    )

                pool.close()
                pool.join()

            pbar.close()

            print(f"Total files processed: {len(audio_files) - len(failed_files)}")
            print(f"Total files skipped due to errors or no vocalization data: {skipped_files_count.value}")

            # Retry failed files
            if failed_files:
                print(f"Retrying {len(failed_files)} failed files...")
                pbar = tqdm(total=len(failed_files), desc="Retrying failed files")
                with Pool(processes=max_processes, maxtasksperchild=100) as pool:  # Adjust maxtasksperchild as needed
                    for file_path in failed_files: 
                        print(f"Retrying file: {file_path}")
                        pool.apply_async(
                            WavtoSpec.safe_process_file,
                            args=(self, file_path),
                            callback=update_progress_bar,
                            error_callback=error_callback
                        )

                    pool.close()
                    pool.join()
                pbar.close()

    def has_vocalization(self, file_path):
        file_name = os.path.basename(file_path)
        vocalization_data, _ = self.check_vocalization(
            file_name=file_name,
            data=None,
            samplerate=None,
            song_detection_json_path=self.song_detection_json_path
        )

        if vocalization_data is None:
            return False
        return True

    @staticmethod
    def safe_process_file(instance, file_path):
        try:
            if instance.song_detection_json_path is not None:
                if instance.has_vocalization(file_path):
                    instance.multiprocess_process_file(file_path, instance.song_detection_json_path)
                else:
                    print(f"File {file_path} skipped due to no vocalization")
                    return None
            else:
                instance.multiprocess_process_file(file_path, instance.song_detection_json_path)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None
        return file_path

    def multiprocess_process_file(self, file_path, song_detection_json_path):
        return self.convert_to_spectrogram(
            file_path,
            song_detection_json_path=song_detection_json_path,
            save_npz=True
        )

    @staticmethod
    def process_file(instance, file_path):
        return instance.convert_to_spectrogram(file_path, song_detection_json_path=None, save_npz=False)
        
    def convert_to_spectrogram(self, file_path, song_detection_json_path, min_length_ms=500, min_timebins=500, save_npz=True):
        try:
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                data = wav_file.read(dtype='int16')
                if wav_file.channels > 1:
                    data = data[:, 0]

            length_in_ms = (len(data) / samplerate) * 1000
            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return None, None 

            file_name = os.path.basename(file_path)

            if song_detection_json_path is not None:
                vocalization_data, syllable_labels = self.check_vocalization(
                    file_name=file_name,
                    data=data,
                    samplerate=samplerate,
                    song_detection_json_path=song_detection_json_path
                )
                if not vocalization_data:
                    print(f"File {file_path} skipped due to no vocalization")
                    return None, None 
            else:
                vocalization_data = [(0, len(data)/samplerate)]  # Assume entire file is vocalization
                syllable_labels = {}  # Empty dict if not using JSON

            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)  # Apply high-pass filter

            hop_length = self.step_size
            window = 'hann'
            n_fft = self.nfft
            Sxx = librosa.stft(data.astype(float), n_fft=n_fft, hop_length=hop_length, window=window)
            Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)

            labels = np.zeros(Sxx.shape[1], dtype=int)

            for label, intervals in syllable_labels.items():
                for start_sec, end_sec in intervals:
                    start_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, start_sec)
                    end_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, end_sec)
                    labels[start_bin:end_bin] = int(label)

            if Sxx.shape[1] >= min_timebins:
                segments_saved = 0  # Track how many segments we actually save
                for i, (start_sec, end_sec) in enumerate(vocalization_data):
                    start_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, start_sec)
                    end_bin = np.searchsorted(np.arange(Sxx.shape[1]) * hop_length / samplerate, end_sec)

                    segment_Sxx_log = Sxx_log[:, start_bin:end_bin]
                    segment_labels = labels[start_bin:end_bin]
                    segment_vocalization = np.ones(end_bin - start_bin, dtype=int)

                    # Skip sub-segments if they are too small
                    if segment_Sxx_log.shape[1] < min_timebins:
                        logging.info(f"Skipping small sub-segment in {file_path} "
                                  f"(length={segment_Sxx_log.shape[1]} < min_timebins={min_timebins}).")
                        continue

                    if save_npz:
                        spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                        segment_spec_file_path = os.path.join(self.dst_dir, f"{spec_filename}_segment_{segments_saved}.npz")
                        np.savez(segment_spec_file_path, 
                                s=segment_Sxx_log, 
                                vocalization=segment_vocalization, 
                                labels=segment_labels)
                        logging.info(f"Segment {segments_saved} spectrogram saved to {segment_spec_file_path} "
                                  f"(length={segment_Sxx_log.shape[1]} timebins)")
                        segments_saved += 1

                if segments_saved == 0:
                    logging.warning(f"No segments from {file_path} met the minimum timebin requirement of {min_timebins}")
                else:
                    logging.info(f"Saved {segments_saved} segments from {file_path}")

                return Sxx_log, vocalization_data, labels
            else:
                logging.info(f"Spectrogram for {file_path} has less than {min_timebins} timebins and will not be saved.")
                return None, None, None

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None, None, None

    def check_vocalization(self, file_name, data, samplerate, song_detection_json_path):
        if not self.use_json:
            return [(0, len(data)/samplerate)], {}  # Assume entire file is vocalization if not using JSON

        # Open JSON file
        if not os.path.exists(song_detection_json_path):
            logging.error(f"JSON file {song_detection_json_path} does not exist.")
            return None, None

        try:
            with open(song_detection_json_path, 'r') as json_file:
                json_data = json.load(json_file)
                for entry in json_data:
                    if entry['filename'] == file_name:
                        if not entry['song_present']:
                            return None, None
                        
                        onsets_offsets = [(seg['onset_ms'] / 1000, seg['offset_ms'] / 1000) for seg in entry['segments']]

                        # Process syllable labels
                        syllable_labels = {}
                        if 'syllable_labels' in entry and entry['syllable_labels']:
                            syllable_labels = entry['syllable_labels']

                        return onsets_offsets, syllable_labels
        except Exception as e:
            logging.error(f"Error reading JSON file {song_detection_json_path}: {e}")
            return None, None

        logging.error(f"No matching entry found for {file_name} in {song_detection_json_path}.")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to spectrograms.")
    parser.add_argument('--src_dir', type=str, help='Source directory containing WAV files.')
    parser.add_argument('--dst_dir', type=str, help='Destination directory to save spectrograms.')
    parser.add_argument('--song_detection_json_path', type=str, default=None, help='Path to the JSON file with song detection data.')
    parser.add_argument('--step_size', type=int, default=119, help='Step size for the spectrogram.')
    parser.add_argument('--nfft', type=int, default=1024, help='Number of FFT points for the spectrogram.')
    parser.add_argument('--generate_random_files_number', type=int, default=None, help='Number of random files to process.')
    parser.add_argument('--single_threaded', action='store_true', help='Run in single-threaded mode')

    args = parser.parse_args()

    wav_to_spec = WavtoSpec(
        args.src_dir, 
        args.dst_dir, 
        args.song_detection_json_path, 
        args.step_size, 
        args.nfft, 
        args.generate_random_files_number,
        args.single_threaded
    )
    wav_to_spec.process_directory()

if __name__ == "__main__":
    main()



'''
Spectrogram Generation and Preprocessing
Raw audio recordings were first filtered to remove low-frequency noise components and emphasize relevant spectral features of canary songs. Specifically, a 5th-order elliptic high-pass filter (0.2 dB ripple, 40 dB stopband attenuation, 500 Hz cutoff frequency) was applied to the time-domain signal. This filter choice ensures that energy below 500 Hz—often dominated by low-frequency noise or cage sounds—is attenuated, improving subsequent feature extraction.

Following filtering, spectrograms were generated using a Short-Time Fourier Transform (STFT) with a Hann window, a 1024-point FFT, and a 119-sample hop length. The 119-sample hop length (at a 44.2 kHz sampling rate) yielded a temporal resolution sufficient to capture fine-grained temporal structure in canary syllables, while still maintaining computational efficiency. The resulting complex spectrogram was converted to a decibel scale via amplitude-to-dB conversion (referenced to the maximum amplitude), yielding a time-frequency representation well-suited for downstream analysis. 

Handling of Long and Short Recordings
Due to memory and computational constraints, recordings were segmented into manageable lengths. If a continuous WAV file exceeded a certain duration threshold (find exact ms threshold), it was split into multiple shorter files. Conversely, recordings shorter than find exact ms threshold were discarded to maintain consistent quality and uniformity in the dataset. This segmentation step ensured that no single file would overwhelm system memory during the STFT and embedding processes, thereby improving the robustness and scalability of the pipeline.

Vocalization-Based Filtering and JSON Metadata
To avoid processing large amounts of non-vocal segments (silence or environmental noise), a supervised "song detection" JSON file was employed (if available). For each WAV file, the system checked a corresponding JSON entry containing segment onsets, offsets, and syllable-level annotations. Only time intervals marked as containing vocalizations were retained, and non-vocal segments were omitted. If no vocalizations were present, the file was skipped altogether. This approach minimized unnecessary computation and focused the analysis on behaviorally relevant acoustic signals.

Segmenting Spectrograms and Removing Small Spectrograms
After applying the STFT, the resulting spectrogram was evaluated for minimal time dimension requirements. Segments of the time-frequency representation with fewer than 1000 time bins (~2.7 seconds, given the 119-sample hop length) were discarded to ensure sufficient contextual information for downstream analyses. This threshold prevented the inclusion of overly short, fragmented spectrograms that might hinder stable embedding and interpretation.

For segments meeting the minimum length criteria, the pipeline generated one or more smaller spectrogram excerpts corresponding to vocalization intervals. Each excerpt was saved as a compressed NumPy archive (.npz) for storage efficiency and faster downstream loading. These per-segment files contained three arrays: (1) the log-amplitude spectrogram, (2) a binary "vocalization" array indicating that the entire segment contains song, and (3) syllable-level label arrays if available from the JSON annotations. Segment-level saving further reduced memory overhead by distributing computations across multiple smaller files and preventing the accumulation of large, unwieldy in-memory arrays.

Labeling and Integration
When syllable-level labels were provided, the pipeline aligned these annotations with the spectrogram's time bins. Time bins were converted to seconds using the known sampling rate and hop length, allowing the assignment of correct syllable or silence labels to each time bin. This integration of annotations facilitated subsequent evaluations of model performance, including comparisons to ground truth labels and alignment with biologically meaningful vocal units (e.g., syllables, phrases).

Parallelization and Memory Management
To accelerate spectrogram generation and preprocessing, the code leveraged multiprocessing capabilities, distributing file processing tasks across all available CPU cores. Memory checks ensured that new processes were spawned only when sufficient system memory was available. In cases of transient memory pressure, the code waited before processing additional files, thereby maintaining stability and preventing system slowdowns or crashes.

If a file encountered errors or failures (e.g., due to corrupted input data), it was logged, and the pipeline attempted to reprocess it after completing the initial batch. This built-in resilience helped maintain data integrity and ensured that the pipeline could progress even in the face of irregularities in the input data.'''