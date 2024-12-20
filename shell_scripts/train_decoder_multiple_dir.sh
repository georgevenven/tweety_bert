#!/bin/bash

# Navigate up one directory
cd ..

# Variables for model and bird names
BIRD_NAME="LLb3_test_with_modification_toscrtipt"
MODEL_NAME="LLB_Model_For_Paper"

# Specify the WAV folder and song detection JSON path
WAV_FOLDER="/media/george-vengrovski/George-SSD/llb_stuff/llb_birds/yarden_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json"

# Number of samples to select (optional)
NUM_SAMPLES=15

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
    --num_samples $NUM_SAMPLES

# Initialize an array to store spectrogram directories
DATA_DIRS=()

# Get the number of groups from the output directory structure
NUM_GROUPS=$(ls -d $OUTPUT_PATH/group_* | wc -l)

# Generate spectrograms for each group
for (( i=1; i<=$NUM_GROUPS; i++ ))
do
    GROUP_SRC_DIR="$OUTPUT_PATH/group_${i}"
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
    --samples 1e4

# Train and save Decoder
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

#Clean up all temporary files
rm -rf "$TEMP_DIR"
echo "Cleaned up all temporary files"
