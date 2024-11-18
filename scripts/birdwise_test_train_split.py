import os
import shutil
import random
import argparse
from tqdm import tqdm
import sys

script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

sys.path.append("src")

from spectogram_generator import WavtoSpec

def split_dataset(folder_path, train_ratio, train_folder_dest, test_folder_dest, move_files=False, generate_specs=False, song_detection_json=None):
    """
    Splits the npz files in the given folder into train and test sets based on the specified ratio
    and either copies or moves them to specified train and test destination folders based on the move_files flag.

    Parameters:
    folder_path (str): The path to the folder containing the dataset.
    train_ratio (float): The ratio of npz files to be included in the train set.
    train_folder_dest (str): The path to the destination train folder.
    test_folder_dest (str): The path to the destination test folder.
    move_files (bool): If True, files will be moved instead of copied. Defaults to False.
    generate_specs (bool): If True, generates spectrograms for files before splitting. Defaults to False.
    song_detection_json (str): Path to song detection JSON file for spectrogram generation.
    """
    # Generate spectrograms if requested
    if generate_specs:
        spec_generator = WavtoSpec(
            src_dir=folder_path,
            dst_dir=folder_path,  # Temporary storage in same directory
            song_detection_json_path=song_detection_json
        )
        spec_generator.process_directory()
        # Update folder_path to look for npz files instead of wav files
        file_extension = '.npz'
    else:
        file_extension = '.npz'

    # Create directories
    os.makedirs(train_folder_dest, exist_ok=True)
    os.makedirs(test_folder_dest, exist_ok=True)

    # List all relevant files
    all_files = [f for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f)) 
                 and f.endswith(file_extension)]

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate number of files for the train set
    train_size = int(len(all_files) * train_ratio)

    # Split files
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]

    # Either move or copy files to respective destination directories based on move_files flag
    for file in tqdm(train_files, desc="Processing train files"):
        src_file_path = os.path.join(folder_path, file)
        dest_file_path = os.path.join(train_folder_dest, file)
        if move_files:
            shutil.move(src_file_path, dest_file_path)
        else:
            shutil.copy2(src_file_path, dest_file_path)

    for file in tqdm(test_files, desc="Processing test files"):
        src_file_path = os.path.join(folder_path, file)
        dest_file_path = os.path.join(test_folder_dest, file)
        if move_files:
            shutil.move(src_file_path, dest_file_path)
        else:
            shutil.copy2(src_file_path, dest_file_path)

def split_from_lists(train_file_list, test_file_list, train_folder_dest, test_folder_dest, generate_specs=False, song_detection_json=None):
    os.makedirs(train_folder_dest, exist_ok=True)
    os.makedirs(test_folder_dest, exist_ok=True)

    def process_file_list(file_list_path):
        with open(file_list_path, 'r') as f:
            return [os.path.splitext(os.path.basename(line.strip()))[0] for line in f.readlines()]

    if generate_specs:
        # Create a temporary directory for all spectrograms
        temp_dir = os.path.join(os.path.dirname(train_folder_dest), 'temp_specs')
        os.makedirs(temp_dir, exist_ok=True)

        # Generate all spectrograms in temp directory
        spec_generator = WavtoSpec(
            src_dir='/media/george-vengrovski/George-SSD/llb_stuff/llb_birds',  # Base directory containing WAV files
            dst_dir=temp_dir,
            song_detection_json_path=song_detection_json
        )
        spec_generator.process_directory()

        # Get lists of file basenames (without extensions)
        train_files = process_file_list(train_file_list)
        test_files = process_file_list(test_file_list)

        # Move files to their respective directories
        for spec_file in os.listdir(temp_dir):
            if not spec_file.endswith('.npz'):
                continue
                
            # Extract the base name without segment suffix
            base_name = spec_file.split('_segment_')[0]
            
            if base_name in train_files:
                shutil.move(
                    os.path.join(temp_dir, spec_file),
                    os.path.join(train_folder_dest, spec_file)
                )
            elif base_name in test_files:
                shutil.move(
                    os.path.join(temp_dir, spec_file),
                    os.path.join(test_folder_dest, spec_file)
                )

        # Clean up temp directory
        shutil.rmtree(temp_dir)
    else:
        # If not generating specs, just copy existing spec files
        train_files = process_file_list(train_file_list)
        test_files = process_file_list(test_file_list)
        
        for npz_file in os.listdir(folder_path):
            if not npz_file.endswith('.npz'):
                continue
                
            base_name = npz_file.split('_segment_')[0]
            if base_name in train_files:
                shutil.copy2(
                    os.path.join(folder_path, npz_file),
                    os.path.join(train_folder_dest, npz_file)
                )
            elif base_name in test_files:
                shutil.copy2(
                    os.path.join(folder_path, npz_file),
                    os.path.join(test_folder_dest, npz_file)
                )

if __name__ == "__main__":
    # Configuration
    config = {
        # Common settings
        'train_folder_dest': '/media/george-vengrovski/George-SSD/llb_stuff/llb_train',
        'test_folder_dest': '/media/george-vengrovski/George-SSD/llb_stuff/llb_test',
        'generate_specs': True,
        'song_detection_json': '/media/george-vengrovski/George-SSD/llb_stuff/llb_birds/merged_output.json',
        
        # Choose mode: 'ratio' or 'lists'
        'mode': 'lists',  # or 'ratio'
        
        # Settings for ratio mode
        'folder_path': '/media/george-vengrovski/George-SSD/llb_stuff/llb_birds/yarden_data',
        'train_ratio': 0.8,
        'move_files': False,
        
        # Settings for lists mode
        'train_list': '/media/george-vengrovski/George-SSD/llb_stuff/LLB_Model_For_Paper/train_files.txt',
        'test_list': '/media/george-vengrovski/George-SSD/llb_stuff/LLB_Model_For_Paper/test_files.txt'
    }

    # Execute based on mode
    if config['mode'] == 'ratio':
        split_dataset(
            folder_path=config['folder_path'],
            train_ratio=config['train_ratio'],
            train_folder_dest=config['train_folder_dest'],
            test_folder_dest=config['test_folder_dest'],
            move_files=config['move_files'],
            generate_specs=config['generate_specs'],
            song_detection_json=config['song_detection_json']
        )
    else:  # lists mode
        split_from_lists(
            train_file_list=config['train_list'],
            test_file_list=config['test_list'],
            train_folder_dest=config['train_folder_dest'],
            test_folder_dest=config['test_folder_dest'],
            generate_specs=config['generate_specs'],
            song_detection_json=config['song_detection_json']
        )
