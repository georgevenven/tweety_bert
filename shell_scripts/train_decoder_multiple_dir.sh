#!/bin/bash

# =====================
# Parameters
# =====================
BIRD_NAME="llb3"
MODEL_NAME="LLB_Model_For_Paper"
NUM_SAMPLES="1e5"  # Number of samples for UMAP

# Data paths
WAV_FOLDER="/media/george-vengrovski/disk1/canary/canary_recordings/USA5347"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/flash-drive/jsons/merged_output.json"

# Temporary directory paths
TEMP_DIR="./temp"
OUTPUT_PATH="$TEMP_DIR"

# Navigate up one directory
cd ..

# Create temporary directories if they don't exist
mkdir -p "$TEMP_DIR"
echo "Created temporary directories"

# Call the Python script to select and copy files
python scripts/copy_files_from_wavdir_to_multiple_event_dirs.py \
    "$WAV_FOLDER" \
    "$SONG_DETECTION_JSON_PATH" \
    "$OUTPUT_PATH"

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
    --samples $NUM_SAMPLES

# Train and save Decoder
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

#Clean up all temporary files
rm -rf "$TEMP_DIR"
echo "Cleaned up all temporary files"
