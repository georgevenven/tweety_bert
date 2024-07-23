import os
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

class SpectrogramProcessor:
    def clear_directory(self, directory_path):
        """Deletes all files and subdirectories within a specified directory."""
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def __init__(self, data_root, train_dir, test_dir, train_prop=0.8, model=None, device=None):
        self.data_root = data_root
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_prop = train_prop
        self.kmeans = None

        # Existing initialization code...
        self.model = model
        self.device = device

        # Create directories if they don't exist
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)

        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

    def process_file(self, file):
        try:
            f = np.load(os.path.join(self.data_root, file), allow_pickle=True)
            spectrogram = f['s']
            spectrogram[np.isnan(spectrogram)] = 0

            if 'labels' in f:
                labels = f['labels']
            else:
                labels = None

            f_dict = {'s': spectrogram, 'labels': labels}
            segment_filename = f"{os.path.splitext(file)[0]}{os.path.splitext(file)[1]}"
            save_path = os.path.join(self.train_dir if np.random.uniform() < self.train_prop else self.test_dir, segment_filename)
            np.savez(save_path, **f_dict)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    def generate_train_test(self, file_min_size=1e3, file_limit_size=1e4):
        files = os.listdir(self.data_root)
        
        with Pool() as pool:
            list(tqdm(pool.imap(self.process_file, files), total=len(files), desc="Processing Files"))