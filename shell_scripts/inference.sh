#!/bin/bash

# Navigate up one directory
cd ..

WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json"
BIRD_NAME="llb3_for_paper"
APPLY_POST_PROCESSING="True"
WINDOW_SIZE=200
VISUALIZE="True"

# call the inference script
python src/inference.py \
    --bird_name "$BIRD_NAME" \
    --wav_dir "$WAV_FOLDER" \
    --song_detection_json "$SONG_DETECTION_JSON_PATH" \
    --apply_post_processing "$APPLY_POST_PROCESSING" \
    --window_size "$WINDOW_SIZE" \
    --visualize "$VISUALIZE"