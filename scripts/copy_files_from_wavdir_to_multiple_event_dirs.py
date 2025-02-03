#!/usr/bin/env python3
"""
this script processes audio files from a specified directory, selecting files around given event dates
and balancing them by data size. it reads a json file containing song detection data to identify files
where a song is present. the script then categorizes files into multiple temporal subgroups before and after
each event date based on their creation dates, ensuring each subgroup has approximately equal data size.
it copies all files in each group to designated output directories for each event date.

usage:
    python copy_files_from_wavdir_to_multiple_event_dirs.py <directory_path> <json_file> <output_path> [--test_set_file <test_set_file>] [--songs_per_fold <songs_per_fold>]

arguments:
    directory_path: path to the main folder containing subdirectories with audio files.
    json_file: path to the song detection json file.
    output_path: directory where the selected files will be copied.
    --test_set_file: optional: path to test set file for random fold creation.
    --songs_per_fold: number of songs per fold when using test set (default is 100).
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.widgets
import numpy as np
import sys
from tqdm import tqdm
from bisect import bisect_left, bisect_right

# function to get the creation date of a file in a cross-platform way
def get_creation_date(path):
    stat = path.stat()
    if hasattr(stat, 'st_birthtime'):
        return stat.st_birthtime  # macos
    elif os.name == 'nt':
        return stat.st_ctime  # windows
    else:
        return stat.st_mtime  # linux/unix approximation using modification time

# helper: find a file in the directory tree and return its full path.
def find_file_in_directory(filename, search_dir):
    search_dir = Path(search_dir)
    for file_path in search_dir.rglob(filename):
        if file_path.is_file():
            return file_path
    return None

class InteractiveDateSelector:
    def __init__(self, time_data):
        self.time_data = sorted(time_data)
        self.selections = []  # list of dicts: {start, end, patch, label}
        self.info_text = None
        self.dashboard_text = None
        self.fig = None
        self.ax = None
        self.span = None

    def count_files_in_range(self, start_date, end_date):
        left = bisect_left(self.time_data, start_date)
        right = bisect_right(self.time_data, end_date)
        return right - left

    def update_dashboard(self):
        # sort selections so group numbering is temporal
        self.selections.sort(key=lambda s: s['start'])
        lines = []
        for i, sel in enumerate(self.selections, start=1):
            cnt = self.count_files_in_range(sel['start'], sel['end'])
            lines.append(f"group {i}: {sel['start'].strftime('%y-%m-%d')} to {sel['end'].strftime('%y-%m-%d')} - {cnt} songs")
            if sel.get('label'):
                sel['label'].set_text(f"grp {i}")
        dash_txt = "\n".join(lines)
        if self.dashboard_text:
            self.dashboard_text.set_text(dash_txt)
        else:
            self.dashboard_text = self.fig.text(0.75, 0.95, dash_txt,
                                                 va='top', fontsize=10,
                                                 bbox=dict(facecolor='white', alpha=0.5))
        self.fig.canvas.draw_idle()

    def on_move(self, xmin, xmax):
        start_date = matplotlib.dates.num2date(xmin).replace(tzinfo=None)
        end_date = matplotlib.dates.num2date(xmax).replace(tzinfo=None)
        cnt = self.count_files_in_range(start_date, end_date)
        msg = f"current: {start_date.strftime('%y-%m-%d')} to {end_date.strftime('%y-%m-%d')} - {cnt} songs"
        if self.info_text:
            self.info_text.set_text(msg)
        else:
            self.info_text = self.fig.text(0.02, 0.95, msg,
                                           fontsize=10,
                                           bbox=dict(facecolor='white', alpha=0.5))
        self.fig.canvas.draw_idle()

    def on_select(self, xmin, xmax):
        start_date = matplotlib.dates.num2date(xmin).replace(tzinfo=None)
        end_date = matplotlib.dates.num2date(xmax).replace(tzinfo=None)
        cnt = self.count_files_in_range(start_date, end_date)
        print(f"selected: {start_date.date()} to {end_date.date()} ({cnt} songs)")
        patch = self.ax.axvspan(xmin, xmax, color='c', alpha=0.3)
        x_center = (xmin + xmax) / 2
        label = self.ax.text(x_center, self.ax.get_ylim()[1]*0.95, "grp ?",
                             ha='center', va='top', fontsize=9, color='black',
                             bbox=dict(facecolor='white', alpha=0.7))
        self.selections.append({
            'start': start_date,
            'end': end_date,
            'patch': patch,
            'label': label
        })
        self.update_dashboard()

    def reset_selections(self):
        # remove all current selections and labels
        for sel in self.selections:
            sel['patch'].remove()
            if sel.get('label'):
                sel['label'].remove()
        self.selections = []
        # reinitialize the span selector to clear any lingering selection rectangle
        if self.span:
            self.span.disconnect_events()
        self.span = matplotlib.widgets.SpanSelector(
            self.ax, self.on_select, 'horizontal', useblit=True,
            interactive=True, drag_from_anywhere=True, onmove_callback=self.on_move
        )
        print("reset all selections")
        self.update_dashboard()

    def undo(self):
        if self.selections:
            sel = self.selections.pop()
            sel['patch'].remove()
            if sel.get('label'):
                sel['label'].remove()
            print("undo last selection")
            self.update_dashboard()
        else:
            print("nothing to undo")
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'enter':
            plt.close()
        elif event.key in ['backspace', 'delete']:
            self.undo()
        elif event.key == 'r':
            self.reset_selections()

    def create_selection_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 6))
        # create a line plot showing songs per day (with markers for fluctuation)
        timestamps = matplotlib.dates.date2num(self.time_data)
        min_date = min(self.time_data)
        max_date = max(self.time_data)
        # generate bins per day
        bins = matplotlib.dates.drange(min_date, max_date + timedelta(days=1), timedelta(days=1))
        counts, bin_edges = np.histogram(timestamps, bins=bins)
        x_vals = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.ax.plot(x_vals, counts, color='blue', lw=2, marker='o')
        # set sparse x-axis labels
        locator = matplotlib.dates.AutoDateLocator(maxticks=10)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
        self.fig.autofmt_xdate()
        self.ax.set_title('select date ranges\n(click-drag to select, enter to finish, backspace to undo, r to reset)')
        self.ax.set_xlabel('date')
        self.ax.set_ylabel('songs per day')
        self.span = matplotlib.widgets.SpanSelector(
            self.ax, self.on_select, 'horizontal', useblit=True,
            interactive=True, drag_from_anywhere=True, onmove_callback=self.on_move
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        print("plot up rn – select ranges, press enter when done, backspace to undo, r to reset")
        plt.show(block=True)
        self.selections.sort(key=lambda s: s['start'])
        return [(sel['start'], sel['end']) for sel in self.selections]

def create_random_folds(file_list, songs_per_fold=100):
    random.shuffle(file_list)
    num_files = len(file_list)
    num_complete_folds = num_files // songs_per_fold
    remaining_files = num_files % songs_per_fold
    folds = [file_list[i:i + songs_per_fold] for i in range(0, num_complete_folds * songs_per_fold, songs_per_fold)]
    if remaining_files > 0:
        remainder = file_list[num_complete_folds * songs_per_fold:]
        for i, file in enumerate(remainder):
            folds[i % len(folds)].append(file)
    return folds

# argument parser (removed --num_samples)
parser = argparse.ArgumentParser(description='select files and create groups either by time ranges or random folds.')
parser.add_argument('directory_path', type=str, help='path to the main folder containing subdirectories')
parser.add_argument('json_file', type=str, help='path to the song detection json file')
parser.add_argument('output_path', type=str, help='directory where the selected files will be copied')
parser.add_argument('--test_set_file', type=str, help='optional: path to test set file for random fold creation')
parser.add_argument('--songs_per_fold', type=int, default=100, help='number of songs per fold when using test set')
args = parser.parse_args()

# load the song detection data
with open(args.json_file, 'r') as f:
    song_data = json.load(f)

# prepare output directory
output_path = Path(args.output_path)
output_path.mkdir(parents=True, exist_ok=True)

if args.test_set_file:
    print("operating in random fold mode...")
    with open(args.test_set_file, 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    print(f"loaded {len(test_files)} files from test set file")
    base_dir = Path(args.directory_path)
    available_wav_files = {}
    for file_path in base_dir.rglob('*.wav'):
        if file_path.is_file():
            available_wav_files[file_path.name] = file_path
    song_files = {}
    for entry in song_data:
        if entry.get('song_present', False):
            filename = Path(entry['filename']).name
            if any(test_file in entry['filename'] for test_file in test_files):
                if filename in available_wav_files:
                    song_files[entry['filename']] = entry
                else:
                    print(f"skipping file not found in wav directory: {filename}")
    print(f"\nfound {len(song_files)} matching files in json data that exist in wav directory")
    file_list = list(song_files.keys())
    if not file_list:
        print("no valid files found after filtering. exiting...")
        sys.exit(1)
    folds = create_random_folds(file_list, args.songs_per_fold)
    print(f"\ncreated {len(folds)} folds")
    for fold_idx, fold_files in enumerate(folds, 1):
        fold_output_dir = output_path / f"group_{fold_idx}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nprocessing fold {fold_idx}:")
        files_copied = 0
        for filename in fold_files:
            source_file = available_wav_files.get(Path(filename).name)
            if source_file:
                shutil.copy(source_file, fold_output_dir / Path(filename).name)
                files_copied += 1
        print(f"group {fold_idx} summary:")
        print(f"  - files in fold: {len(fold_files)}")
        print(f"  - files copied: {files_copied}")
else:
    print("operating in temporal selection mode...")
    song_files = {entry['filename']: entry for entry in song_data if entry.get('song_present', False)}
    print("creating time-based line plot data...")
    time_data = []
    base_dir = Path(args.directory_path)
    print("indexing files in directory (recursive search)...")
    file_map = {}
    for file_path in base_dir.rglob('*.wav'):
        if file_path.is_file():
            file_map[file_path.name] = file_path
    print("found the following .wav files:")
    for fname, fpath in file_map.items():
        print(f"filename: {fname} -> path: {fpath}")
    print("processing song data...")
    for entry in tqdm(song_data):
        if entry.get('song_present', False):
            filename = Path(entry['filename']).name
            if filename in file_map:
                file_path = file_map[filename]
                creation_date = datetime.fromtimestamp(get_creation_date(file_path))
                time_data.append(creation_date)
    if not time_data:
        print("no valid files found! check your directory path and json file.")
        sys.exit(1)
    print(f"found {len(time_data)} valid files. opening selection plot...")
    selector = InteractiveDateSelector(time_data)
    print("please select date ranges in the popup window. press enter when finished.")
    date_ranges = selector.create_selection_plot()
    if not date_ranges:
        print("no date ranges were selected. exiting...")
        sys.exit(1)
    print(f"selected {len(date_ranges)} date ranges. processing files...")
    for range_idx, (start_date, end_date) in enumerate(date_ranges):
        group_files = []
        for file in base_dir.rglob('*.wav'):
            if file.is_file() and file.name in song_files:
                file_creation_date = datetime.fromtimestamp(get_creation_date(file))
                if start_date <= file_creation_date <= end_date:
                    song_info = song_files[file.name]
                    if 'segments' in song_info and song_info['segments']:
                        duration = sum(segment['offset_ms'] - segment['onset_ms']
                                       for segment in song_info['segments'])
                        group_files.append((file, duration, file_creation_date))
        # process all files in the group (sampling removed)
        selected_files = group_files
        group_output_dir = output_path / f"group_{range_idx + 1}"
        group_output_dir.mkdir(parents=True, exist_ok=True)
        for file, duration, _ in selected_files:
            shutil.copy(file, group_output_dir / file.name)
        print(f"group {range_idx + 1}: copied {len(selected_files)} files")
