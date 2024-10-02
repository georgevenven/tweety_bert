#!/usr/bin/env python3


import os
import sys
import torch
import shutil
import subprocess


# Set up paths
figure_generation_dir = os.path.dirname(__file__)
project_root = os.path.dirname(figure_generation_dir)
os.chdir(project_root)
print(f"Project root directory: {project_root}")


sys.path.append("src")
# Import necessary modules
from utils import load_model
from analysis import plot_umap_projection_two_datasets  # Use the modified function we created earlier


# Define variables
BIRD_NAME1 = "USA5510_Fall"
BIRD_NAME2 = "USA5510_Spring"
MODEL_NAME = "TweetyBERT_Pretrain_LLB_AreaX_FallSong"
WAV_FOLDER1 = "/media/rose/Extreme SSD/Compare_seasons_recordings/RC1_USA5510_Comp2"
WAV_FOLDER2 = "/media/rose/Extreme SSD/Compare_seasons_recordings/USA5510"
SONG_DETECTION_JSON_PATH = "/home/rose/Documents/tweety_net_song_detector/output/onset_offset_results.json"


# Generate spectrograms for UMAP
TEMP_DIR = "./temp"
UMAP_FILES1 = os.path.join(TEMP_DIR, "umap_files1")
UMAP_FILES2 = os.path.join(TEMP_DIR, "umap_files2")




if not os.path.exists(TEMP_DIR):
   os.makedirs(TEMP_DIR)
   print(f"Created temporary directory: {TEMP_DIR}")


if not os.path.exists(UMAP_FILES1):
   os.makedirs(UMAP_FILES1)
   print(f"Created UMAP files directory: {UMAP_FILES1}")


if not os.path.exists(UMAP_FILES2):
   os.makedirs(UMAP_FILES2)
   print(f"Created UMAP files directory: {UMAP_FILES2}")




# Generate spectrograms using the spectrogram_generator.py script
subprocess.run([
   'python', 'src/spectogram_generator.py',
   '--src_dir', WAV_FOLDER1,
   '--dst_dir', UMAP_FILES1,
   '--song_detection_json_path', SONG_DETECTION_JSON_PATH,
   '--generate_random_files_number', '2500'
])


# Generate spectrograms using the spectrogram_generator.py script
subprocess.run([
   'python', 'src/spectogram_generator.py',
   '--src_dir', WAV_FOLDER2,
   '--dst_dir', UMAP_FILES2,
   '--song_detection_json_path', SONG_DETECTION_JSON_PATH,
   '--generate_random_files_number', '2500'
])




# Define your model and device
device = torch.device("cpu")
experiment_folder = f"experiments/{MODEL_NAME}"
model = load_model(experiment_folder)
model = model.to(device)


# Use the generated spectrograms for UMAP
data_dir1 = UMAP_FILES1  # First dataset (generated spectrograms)
data_dir2 = UMAP_FILES2  # Replace with the path to your second dataset


# Call the plot_umap_projection_two_datasets function
plot_umap_projection_two_datasets(
   model=model,
   device=device,
   data_dir1=data_dir1,
   data_dir2=data_dir2,
   category_colors_file="test_llb16",
   samples=50000,
   file_path='files/category_colors_llb3.pkl',
   layer_index=-2,
   dict_key="attention_output",
   context=1000,
   save_name="USA5510_fall_spring",
   raw_spectogram=False,
   save_dict_for_analysis=False,
   remove_non_vocalization=True
)
# Delete UMAP files
shutil.rmtree(UMAP_FILES1)
shutil.rmtree(UMAP_FILES2)
print(f"Deleted UMAP files directories: {UMAP_FILES1, UMAP_FILES2}")
