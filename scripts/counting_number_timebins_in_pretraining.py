#!/usr/bin/env python3

import json

def tally_song_duration(json_path, txt_path):
    # read the list of filenames from the txt file
    with open(txt_path, 'r') as f:
        used_files = set(line.strip() for line in f if line.strip())

    # read the json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_song_ms = 0.0

    # iterate through each entry in the json
    for entry in data:
        filename = entry.get("filename", "")
        # check if the current file is in the list of used files
        if filename in used_files:
            # if there's song present, sum durations
            if entry.get("song_present", False):
                segments = entry.get("segments", [])
                for seg in segments:
                    onset_ms = seg.get("onset_ms", 0)
                    offset_ms = seg.get("offset_ms", 0)
                    total_song_ms += (offset_ms - onset_ms)

    # convert total duration to seconds (if desired)
    total_song_seconds = total_song_ms / 1000.0
    return total_song_seconds

if __name__ == "__main__":
    # Replace these paths with your actual file paths
    json_file = "/media/george-vengrovski/Desk SSD/TweetyBERT/contains_seasonality.json"
    txt_file = "/media/george-vengrovski/Desk SSD/TweetyBERT/models/Seasonality_Model_Final_TweetyBERT_Paper/train_files.txt"

    duration = tally_song_duration(json_file, txt_file)
    print(f"Total song duration (seconds): {duration}")
    # Convert to hours for easier interpretation
    hours = duration / 3600
    print(f"Total song duration (hours): {hours:.2f}")