import matplotlib.pyplot as plt
import os
import sys

# so relative paths can be used 
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

sys.path.append("src")

from spectogram_processor import SpectrogramProcessor

# Define a configuration class or use a dictionary
class Config:
    def __init__(self, data_root, train_dir, test_dir):
        self.data_root = data_root
        self.train_dir = train_dir
        self.test_dir = test_dir
        
configs = [
    Config(data_root="/media/george-vengrovski/disk1/combined_song_data_1", train_dir="/media/george-vengrovski/disk1/combined_song_data_1_train", test_dir="/media/george-vengrovski/disk1/combined_song_data_1_test")
]

# Iterate over the configurations and process
for config in configs:
    processor = SpectrogramProcessor(data_root=config.data_root, train_dir=config.train_dir, test_dir=config.test_dir)

    processor.clear_directory(config.train_dir)
    processor.clear_directory(config.test_dir)

    # if over 10k timebins, split the file 
    processor.generate_train_test()