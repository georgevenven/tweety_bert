#!/bin/bash

# Navigate up one directory
cd ..

# Variables for model and bird names
BIRD_NAME="USA5508"
MODEL_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"

# Specify the WAV folder and song detection JSON path
WAV_FOLDER="/media/george-vengrovski/Diana-SSD/GEORGE/FallvSpring2024_fallandspringcombined/USA5508_Combined"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/Diana-SSD/GEORGE/spring_fall_cohort_merged_output.json"

# Event dates (you can specify multiple dates)
EVENT_DATES=("2024-07-20")  # Example date

# Number of samples to select (optional)
NUM_SAMPLES=500

# Temporary directory paths
TEMP_DIR="./temp"
OUTPUT_PATH="$TEMP_DIR"

# Paths for spectrogram files
SPEC_FILES_PRE="$TEMP_DIR/spec_files_pre"
SPEC_FILES_POST="$TEMP_DIR/spec_files_post"

# Create temporary directories if they don't exist
mkdir -p "$TEMP_DIR" "$SPEC_FILES_PRE" "$SPEC_FILES_POST"
echo "Created temporary directories"

# Call the Python script to select and copy files
python scripts/copy_files_from_wavdir_to_multiple_event_dirs.py \
    "$WAV_FOLDER" \
    "$SONG_DETECTION_JSON_PATH" \
    "$OUTPUT_PATH" \
    "${EVENT_DATES[@]}" \
    --num_samples $NUM_SAMPLES

# Generate spectrograms for pre-event files
python src/spectogram_generator.py \
    --src_dir "$OUTPUT_PATH/pre_event_${EVENT_DATES[0]}" \
    --dst_dir "$SPEC_FILES_PRE" \
    --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# Generate spectrograms for post-event files
python src/spectogram_generator.py \
    --src_dir "$OUTPUT_PATH/post_event_${EVENT_DATES[0]}" \
    --dst_dir "$SPEC_FILES_POST" \
    --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# UMAP with multiple directories
python figure_generation_scripts/dim_reduced_birdsong_plots.py \
    --experiment_folder "experiments/$MODEL_NAME" \
    --data_dir "$SPEC_FILES_PRE" "$SPEC_FILES_POST" \
    --save_name "$BIRD_NAME" \
    --samples 1e6

# Train and save Decoder 
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

# Clean up all temporary files
rm -rf "$TEMP_DIR"
echo "Cleaned up all temporary files"
