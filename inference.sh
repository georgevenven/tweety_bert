#!/bin/bash

WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/home/george-vengrovski/Documents/projects/tweety_net_song_detector/output/onset_offset_results.json"
BIRD_NAME="test"

# call the inference script
python src/inference.py --bird_name "$BIRD_NAME" --wav_dir "$WAV_FOLDER" --song_detection_json "$SONG_DETECTION_JSON_PATH" --apply_post_processing True