"""
This script processes audio files from a specified directory, selecting files around given event dates
and balancing them by duration. It reads a JSON file containing song detection data to identify files
where a song is present. The script then categorizes files into pre-event and post-event groups based
on their creation dates relative to the event dates. It samples files to balance the total duration
between these groups and copies them to designated output directories for each event date.

Usage:
    python copy_files_from_wavdir_to_multiple_event_dirs.py <directory_path> <json_file> <output_path> <event_dates> [--num_samples <num_samples>]

Arguments:
    directory_path: Path to the main folder containing subdirectories with audio files.
    json_file: Path to the song detection JSON file.
    output_path: Directory where the selected files will be copied.
    event_dates: Event dates in YYYY-MM-DD format.
    --num_samples: Approximate number of samples per group for each event (default is 5).
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import time

# Function to get the creation date of a directory in a cross-platform way
def get_creation_date(path):
    stat = path.stat()
    if hasattr(stat, 'st_birthtime'):
        return stat.st_birthtime  # macOS
    elif os.name == 'nt':
        return stat.st_ctime  # Windows
    else:
        return stat.st_mtime  # Linux/Unix approximation using modification time

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Select files around event dates and balance by duration.')
parser.add_argument('directory_path', type=str, help='Path to the main folder containing subdirectories')
parser.add_argument('json_file', type=str, help='Path to the song detection JSON file')
parser.add_argument('output_path', type=str, help='Directory where the selected files will be copied')
parser.add_argument('event_dates', type=str, nargs='+', help='Event dates in YYYY-MM-DD format')
parser.add_argument('--num_samples', type=int, default=5, help='Approximate number of samples per group for each event')
args = parser.parse_args()

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

    # Separate pre-event and post-event files
    pre_event_files = []
    post_event_files = []

    # Iterate over each subdirectory in the main folder
    for subdirectory in Path(args.directory_path).iterdir():
        if subdirectory.is_dir():
            subdirectory_creation_date = datetime.fromtimestamp(get_creation_date(subdirectory))

            # Classify each file within this subdirectory as pre-event or post-event
            for file in subdirectory.rglob('*'):
                if file.is_file() and file.name in song_files:
                    song_info = song_files[file.name]
                    if 'segments' in song_info and song_info['segments']:
                        duration = sum(segment['offset_ms'] - segment['onset_ms'] for segment in song_info['segments'])
                        
                        if subdirectory_creation_date < event_dt:
                            pre_event_files.append((file, duration))
                        else:
                            post_event_files.append((file, duration))

    # Modify the sample_balanced_files function to consider both duration and number of samples
    def sample_balanced_files(files, target_duration, num_samples):
        selected_files = []
        random.shuffle(files)
        total_duration = 0
        for file, duration in files[:num_samples]:  # Limit to num_samples
            selected_files.append(file)
            total_duration += duration
        return selected_files

    target_duration = min(sum(d for _, d in pre_event_files), sum(d for _, d in post_event_files))

    # Use num_samples parameter when sampling files
    pre_event_sample = sample_balanced_files(pre_event_files, target_duration, args.num_samples)
    post_event_sample = sample_balanced_files(post_event_files, target_duration, args.num_samples)

    # Prepare output directories for this event
    pre_event_output_dir = output_path / f"pre_event_{event_date}"
    post_event_output_dir = output_path / f"post_event_{event_date}"
    pre_event_output_dir.mkdir(parents=True, exist_ok=True)
    post_event_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy files to the target directories
    for file in pre_event_sample:
        shutil.copy(file, pre_event_output_dir / file.name)
    for file in post_event_sample:
        shutil.copy(file, post_event_output_dir / file.name)

    print(f"Copied {len(pre_event_sample)} pre-event and {len(post_event_sample)} post-event files for event date {event_date}")
