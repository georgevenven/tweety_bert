#!/usr/bin/env python3
"""
This script processes audio files from a specified directory, selecting files around given event dates
and balancing them by data size. It reads a JSON file containing song detection data to identify files
where a song is present. The script then categorizes files into multiple temporal subgroups before and after
each event date based on their creation dates, ensuring each subgroup has approximately equal data size.
It samples files to balance the total duration between these groups and copies them to designated output
directories for each event date.

Usage:
    python copy_files_from_wavdir_to_multiple_event_dirs.py <directory_path> <json_file> <output_path> [--num_samples <num_samples>] [--test_set_file <test_set_file>] [--songs_per_fold <songs_per_fold>]

Arguments:
    directory_path: Path to the main folder containing subdirectories with audio files.
    json_file: Path to the song detection JSON file.
    output_path: Directory where the selected files will be copied.
    --num_samples: Approximate number of samples per group for each event (default is 5).
    --test_set_file: Optional: Path to test set file for random fold creation.
    --songs_per_fold: Number of songs per fold when using test set (default is 100).
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np
import sys
from tqdm import tqdm

# Function to get the creation date of a file in a cross-platform way
def get_creation_date(path):
    stat = path.stat()
    if hasattr(stat, 'st_birthtime'):
        return stat.st_birthtime  # macOS
    elif os.name == 'nt':
        return stat.st_ctime  # Windows
    else:
        return stat.st_mtime  # Linux/Unix approximation using modification time

# Add this function to handle file paths
def find_file_in_directory(filename, search_dir):
    """Find a file in the directory tree and return its full path."""
    search_dir = Path(search_dir)
    for file_path in search_dir.rglob(filename):
        if file_path.is_file():
            return file_path
    return None

class InteractiveDateSelector:
    def __init__(self, time_data):
        self.time_data = sorted(time_data)  # Sort the dates
        self.selected_ranges = []
        self.current_group = 0
        self.info_text = None  # Add this to store the info text object
        
    def create_selection_plot(self):
        try:
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Convert datetime objects to matplotlib dates
            timestamps = matplotlib.dates.date2num(self.time_data)
            
            # Calculate weekly bins
            min_date = min(self.time_data)
            max_date = max(self.time_data)
            num_weeks = ((max_date - min_date).days // 7) + 1
            bins = [min_date + timedelta(weeks=x) for x in range(num_weeks + 1)]
            bins = matplotlib.dates.date2num(bins)
            
            # Create histogram with weekly bins
            plt.hist(timestamps, bins=bins, alpha=0.5, histtype='step', linewidth=2)
            
            # Format x-axis to show dates
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()
            
            # Add dynamic info text
            self.info_text = plt.figtext(0.02, 0.95, '', wrap=True, horizontalalignment='left', fontsize=10)
            
            self.span = SpanSelector(
                ax,
                self.on_select,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor=f'C{self.current_group}'),
                interactive=True,
                drag_from_anywhere=True,
                onmove_callback=self.on_move
            )
            
            plt.title('Select Date Ranges for Groups\nClick and drag to select ranges\nPress Enter when finished')
            plt.xlabel('Date')
            plt.ylabel('Number of Songs per Week')
            
            # Add instructions
            plt.figtext(0.02, 0.02, f'Currently selecting group {self.current_group + 1}\nPress Enter when done with all selections', 
                       wrap=True, horizontalalignment='left', fontsize=8)
            
            # Connect keyboard event
            fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            
            print("Plot window should now be visible. Please select date ranges.")
            plt.show(block=True)
            
            return self.selected_ranges
            
        except Exception as e:
            print(f"Error creating plot: {e}")
            return []

    def count_files_in_range(self, start_date, end_date):
        """Count files within the given date range."""
        count = 0
        for date in self.time_data:
            if start_date <= date <= end_date:
                count += 1
        return count

    def on_move(self, press, release):
        """Called while dragging the selector."""
        if press is None or release is None:
            return
        
        # Convert to datetime
        start_date = matplotlib.dates.num2date(press).replace(tzinfo=None)
        end_date = matplotlib.dates.num2date(release).replace(tzinfo=None)
        
        # Count files in range
        file_count = self.count_files_in_range(start_date, end_date)
        
        # Update info text
        self.info_text.set_text(
            f'Current selection:\n'
            f'From: {start_date.strftime("%Y-%m-%d")}\n'
            f'To: {end_date.strftime("%Y-%m-%d")}\n'
            f'Files in range: {file_count}'
        )
        plt.gcf().canvas.draw_idle()

    def on_select(self, xmin, xmax):
        # Convert matplotlib dates back to datetime
        start_date = matplotlib.dates.num2date(xmin).replace(tzinfo=None)
        end_date = matplotlib.dates.num2date(xmax).replace(tzinfo=None)
        
        # Count files in final selection
        file_count = self.count_files_in_range(start_date, end_date)
        
        self.selected_ranges.append((start_date, end_date))
        print(f"Selected range for group {self.current_group + 1}: "
              f"{start_date.date()} to {end_date.date()} "
              f"(Contains {file_count} files)")
        self.current_group += 1
        
        # Update span color
        new_span = SpanSelector(
            plt.gca(),
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor=f'C{self.current_group}'),
            interactive=True,
            drag_from_anywhere=True,
            onmove_callback=self.on_move
        )
        self.span.disconnect_events()
        self.span = new_span
        
        plt.figtext(0.02, 0.02, f'Currently selecting group {self.current_group + 1}\nPress Enter when done with all selections', 
                   wrap=True, horizontalalignment='left', fontsize=8)

    def on_key_press(self, event):
        if event.key == 'enter':
            plt.close()

def create_random_folds(file_list, songs_per_fold=100):
    """Split files into random folds of specified size, distributing remainder evenly."""
    random.shuffle(file_list)
    
    # Calculate number of complete folds
    num_files = len(file_list)
    num_complete_folds = num_files // songs_per_fold
    remaining_files = num_files % songs_per_fold
    
    # Create base folds
    folds = [file_list[i:i + songs_per_fold] 
             for i in range(0, num_complete_folds * songs_per_fold, songs_per_fold)]
    
    # If there are remaining files, distribute them evenly across folds
    if remaining_files > 0:
        remainder = file_list[num_complete_folds * songs_per_fold:]
        for i, file in enumerate(remainder):
            folds[i % len(folds)].append(file)
    
    return folds

# Modify argument parser to add optional test set parameter
parser = argparse.ArgumentParser(description='Select files and create groups either by time ranges or random folds.')
parser.add_argument('directory_path', type=str, help='Path to the main folder containing subdirectories')
parser.add_argument('json_file', type=str, help='Path to the song detection JSON file')
parser.add_argument('output_path', type=str, help='Directory where the selected files will be copied')
parser.add_argument('--num_samples', type=int, default=5, help='Approximate number of samples per group')
parser.add_argument('--test_set_file', type=str, help='Optional: Path to test set file for random fold creation')
parser.add_argument('--songs_per_fold', type=int, default=100, help='Number of songs per fold when using test set')
args = parser.parse_args()

# Load the song detection data
with open(args.json_file, 'r') as f:
    song_data = json.load(f)

# Prepare output directory
output_path = Path(args.output_path)
output_path.mkdir(parents=True, exist_ok=True)

if args.test_set_file:
    # Random fold creation mode
    print("Operating in random fold mode...")
    
    # Load test set files
    with open(args.test_set_file, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(test_files)} files from test set file")

    # First, create a map of available wav files
    base_dir = Path(args.directory_path)
    available_wav_files = {}
    for file_path in base_dir.rglob('*.wav'):
        if file_path.is_file():
            available_wav_files[file_path.name] = file_path

    # Filter song data for test files that actually exist in wav directory
    song_files = {}
    for entry in song_data:
        if entry.get('song_present', False):
            filename = Path(entry['filename']).name
            if any(test_file in entry['filename'] for test_file in test_files):
                if filename in available_wav_files:
                    song_files[entry['filename']] = entry
                else:
                    print(f"Skipping file not found in wav directory: {filename}")
    
    print(f"\nFound {len(song_files)} matching files in JSON data that exist in wav directory")

    # Create folds only with files that exist
    file_list = list(song_files.keys())
    if not file_list:
        print("No valid files found after filtering. Exiting...")
        sys.exit(1)

    folds = create_random_folds(file_list, args.songs_per_fold)
    print(f"\nCreated {len(folds)} folds")

    # Process each fold
    for fold_idx, fold_files in enumerate(folds, 1):
        fold_output_dir = output_path / f"group_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing fold {fold_idx}:")
        files_copied = 0
        
        # Copy files to fold directory
        for filename in fold_files:
            source_file = available_wav_files.get(Path(filename).name)
            if source_file:  # This check is redundant now but kept for safety
                shutil.copy(source_file, fold_output_dir / Path(filename).name)
                files_copied += 1
        
        print(f"Group {fold_idx} summary:")
        print(f"  - Files in fold: {len(fold_files)}")
        print(f"  - Files copied: {files_copied}")

else:
    # Original temporal selection mode
    print("Operating in temporal selection mode...")
    
    # Filter files where 'song_present' is true
    song_files = {entry['filename']: entry for entry in song_data if entry.get('song_present', False)}

    print("Creating time-based histogram data...")
    time_data = []
    base_dir = Path(args.directory_path)

    # Modified block: Recursive search for .wav files
    print("Indexing files in directory (recursive search)...")
    file_map = {}
    for file_path in base_dir.rglob('*.wav'):
        if file_path.is_file():
            file_map[file_path.name] = file_path

    # Print out all found .wav files
    print("Found the following .wav files:")
    for fname, fpath in file_map.items():
        print(f"Filename: {fname} -> Path: {fpath}")

    # Process song data with progress bar
    print("Processing song data...")
    for entry in tqdm(song_data):
        if entry.get('song_present', False):
            filename = Path(entry['filename']).name
            if filename in file_map:
                file_path = file_map[filename]
                creation_date = datetime.fromtimestamp(get_creation_date(file_path))
                time_data.append(creation_date)

    if not time_data:
        print("No valid files found! Check your directory path and JSON file.")
        sys.exit(1)

    print(f"Found {len(time_data)} valid files. Opening selection plot...")

    # Launch interactive selector
    selector = InteractiveDateSelector(time_data)
    print("Please select date ranges in the popup window. Press Enter when finished.")
    date_ranges = selector.create_selection_plot()

    if not date_ranges:
        print("No date ranges were selected. Exiting...")
        sys.exit(1)

    print(f"Selected {len(date_ranges)} date ranges. Processing files...")

    # Process files based on selected ranges
    for range_idx, (start_date, end_date) in enumerate(date_ranges):
        group_files = []
        
        # Iterate recursively through the directory
        for file in base_dir.rglob('*.wav'):
            if file.is_file() and file.name in song_files:
                file_creation_date = datetime.fromtimestamp(get_creation_date(file))
                if start_date <= file_creation_date <= end_date:
                    song_info = song_files[file.name]
                    if 'segments' in song_info and song_info['segments']:
                        duration = sum(segment['offset_ms'] - segment['onset_ms'] 
                                       for segment in song_info['segments'])
                        group_files.append((file, duration, file_creation_date))

        # Sample files from the group
        random.shuffle(group_files)
        selected_files = group_files[:args.num_samples]
        
        # Copy files to output directory
        group_output_dir = output_path / f"group_{range_idx + 1}"
        group_output_dir.mkdir(parents=True, exist_ok=True)
        
        for file, duration, _ in selected_files:
            shutil.copy(file, group_output_dir / file.name)
        
        print(f"Group {range_idx + 1}: Copied {len(selected_files)} files")
