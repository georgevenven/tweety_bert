#!/usr/bin/env python3
"""
This script processes audio files from a specified directory, selecting files around given event dates
and balancing them by data size. It reads a JSON file containing song detection data to identify files
where a song is present. The script then categorizes files into multiple temporal subgroups before and after
each event date based on their creation dates, ensuring each subgroup has approximately equal data size.
It samples files to balance the total duration between these groups and copies them to designated output
directories for each event date.

Usage:
    python copy_files_from_wavdir_to_multiple_event_dirs.py <directory_path> <json_file> <output_path> <event_dates> [--num_samples <num_samples>] [--num_groups <num_groups>]

Arguments:
    directory_path: Path to the main folder containing subdirectories with audio files.
    json_file: Path to the song detection JSON file.
    output_path: Directory where the selected files will be copied.
    event_dates: Event dates in YYYY-MM-DD format.
    --num_samples: Approximate number of samples per group for each event (default is 5).
    --num_groups: Number of groups to divide the files into (must be an even number, default is 2).
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Function to get the creation date of a file in a cross-platform way
def get_creation_date(path):
    stat = path.stat()
    if hasattr(stat, 'st_birthtime'):
        return stat.st_birthtime  # macOS
    elif os.name == 'nt':
        return stat.st_ctime  # Windows
    else:
        return stat.st_mtime  # Linux/Unix approximation using modification time

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Select files around event dates and balance by data size.')
parser.add_argument('directory_path', type=str, help='Path to the main folder containing subdirectories')
parser.add_argument('json_file', type=str, help='Path to the song detection JSON file')
parser.add_argument('output_path', type=str, help='Directory where the selected files will be copied')
parser.add_argument('event_dates', type=str, nargs='+', help='Event dates in YYYY-MM-DD format')
parser.add_argument('--num_samples', type=int, default=5, help='Approximate number of samples per group for each event')
parser.add_argument('--num_groups', type=int, default=2, help='Number of groups (even number)')
args = parser.parse_args()

# Validate num_groups
if args.num_groups % 2 != 0 or args.num_groups < 2:
    parser.error('num_groups must be an even number greater than or equal to 2.')

# Load the song detection data
with open(args.json_file, 'r') as f:
    song_data = json.load(f)

# Filter files where 'song_present' is true
song_files = {entry['filename']: entry for entry in song_data if entry.get('song_present', False)}

# Prepare output directory
output_path = Path(args.output_path)
output_path.mkdir(parents=True, exist_ok=True)

# Process each event date
for event_date in args.event_dates:
    event_dt = datetime.strptime(event_date, '%Y-%m-%d')

    # Separate pre-event and post-event files with their creation dates
    pre_event_files = []
    post_event_files = []

    # Iterate over each subdirectory in the main folder
    for subdirectory in Path(args.directory_path).iterdir():
        if subdirectory.is_dir():
            for file in subdirectory.rglob('*'):
                if file.is_file() and file.name in song_files:
                    song_info = song_files[file.name]
                    if 'segments' in song_info and song_info['segments']:
                        duration = sum(segment['offset_ms'] - segment['onset_ms'] for segment in song_info['segments'])
                        file_creation_date = datetime.fromtimestamp(get_creation_date(file))

                        if file_creation_date < event_dt:
                            pre_event_files.append((file, duration, file_creation_date))
                        else:
                            post_event_files.append((file, duration, file_creation_date))

    num_groups = args.num_groups
    num_pre_groups = num_groups // 2
    num_post_groups = num_groups // 2

    # Initialize group files
    group_files = [[] for _ in range(num_groups)]

    # Function to split files into equal-sized groups based on data size
    def split_into_equal_groups(files, num_groups):
        files_sorted = sorted(files, key=lambda x: x[2])  # Sort by creation date
        total_files = len(files_sorted)
        group_size = total_files // num_groups
        remainder = total_files % num_groups

        groups = []
        start_idx = 0
        for i in range(num_groups):
            extra = 1 if i < remainder else 0  # Distribute remainder among first groups
            end_idx = start_idx + group_size + extra
            group = files_sorted[start_idx:end_idx]
            groups.append(group)
            start_idx = end_idx
        return groups

    # Process pre-event files
    if pre_event_files:
        pre_event_groups = split_into_equal_groups(pre_event_files, num_pre_groups)
        for idx, group in enumerate(pre_event_groups):
            group_files[idx] = group
    else:
        print("No pre-event files found.")

    # Process post-event files
    if post_event_files:
        post_event_groups = split_into_equal_groups(post_event_files, num_post_groups)
        for idx, group in enumerate(post_event_groups):
            group_files[num_pre_groups + idx] = group
    else:
        print("No post-event files found.")

    # Sample files from each group
    selected_files_per_group = []
    for i, group in enumerate(group_files):
        files_in_group = group
        random.shuffle(files_in_group)
        selected_files = files_in_group[:args.num_samples]
        selected_files_per_group.append(selected_files)
        print(f"Group {i + 1} has {len(files_in_group)} files, selected {len(selected_files)} files.")

    # Prepare output directories and copy files
    for group_index, selected_files in enumerate(selected_files_per_group):
        group_output_dir = output_path / f"group_{group_index + 1}_event_{event_date}"
        if not selected_files:
            print(f"Group {group_index + 1} is empty. Skipping.")
            continue
        group_output_dir.mkdir(parents=True, exist_ok=True)

        for file, duration, _ in selected_files:
            shutil.copy(file, group_output_dir / file.name)

    # Print summary
    print(f"Copied files for event date {event_date}:")
    for i, group in enumerate(selected_files_per_group):
        print(f"Group {i + 1}: {len(group)} files")
