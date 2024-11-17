#!/bin/bash

# Navigate up one directory
cd ..

# Variables for model and bird names
BIRD_NAME="USA5508_test"
MODEL_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"

# Specify the WAV folder and song detection JSON path
WAV_FOLDER="/media/george-vengrovski/Diana-SSD/GEORGE/FallvSpring2024_fallandspringcombined/USA5508_Combined"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/Diana-SSD/GEORGE/spring_fall_cohort_merged_output.json"

# Event dates (you can specify multiple dates)
EVENT_DATES=("2024-07-20")  # Example date

# Number of samples to select (optional)
NUM_SAMPLES=50

# Number of groups (must be an even number)
NUM_GROUPS=4  # Change this to the desired even number of groups

# Temporary directory paths
TEMP_DIR="./temp"
OUTPUT_PATH="$TEMP_DIR"

# Create temporary directories if they don't exist
mkdir -p "$TEMP_DIR"
echo "Created temporary directories"

# Call the Python script to select and copy files
python scripts/copy_files_from_wavdir_to_multiple_event_dirs.py \
    "$WAV_FOLDER" \
    "$SONG_DETECTION_JSON_PATH" \
    "$OUTPUT_PATH" \
    "${EVENT_DATES[@]}" \
    --num_samples $NUM_SAMPLES \
    --num_groups $NUM_GROUPS

# Initialize an array to store spectrogram directories
DATA_DIRS=()

# Generate spectrograms for each group
for (( i=1; i<=$NUM_GROUPS; i++ ))
do
    GROUP_SRC_DIR="$OUTPUT_PATH/group_${i}_event_${EVENT_DATES[0]}"
    GROUP_SPEC_DIR="$TEMP_DIR/spec_files_group_$i"
    mkdir -p "$GROUP_SPEC_DIR"
    echo "Processing group $i"
    python src/spectogram_generator.py \
        --src_dir "$GROUP_SRC_DIR" \
        --dst_dir "$GROUP_SPEC_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH"
    DATA_DIRS+=("$GROUP_SPEC_DIR")
done

# UMAP with multiple directories
python figure_generation_scripts/dim_reduced_birdsong_plots.py \
    --experiment_folder "experiments/$MODEL_NAME" \
    --data_dir "${DATA_DIRS[@]}" \
    --save_name "$BIRD_NAME" \
    --samples 1e5

# Train and save Decoder
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

# Clean up all temporary files
rm -rf "$TEMP_DIR"
echo "Cleaned up all temporary files"
