#!/bin/bash

### IN THE FUTURE, THE BEHAVIOR OF THIS SHELL SCRIPT AS WELL AS TRAIN DECODER MULTIPLE DIR AND COPY FILES FROM WAVDIR TO MULTIPLE EVENT DIRS SHOULD BE REFACTORED

# Navigate up one directory
cd ..

# Variables for model and bird names
BIRD_NAME="llb3"
MODEL_NAME="LLB_Model_For_Paper"
WAV_FOLDER="/media/george-vengrovski/George-SSD/llb_stuff/llb_birds/yarden_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json"
TEST_SET_FILE="experiments/LLB_Model_For_Paper/test_files.txt"  # Add your test set file path here

# Create temp directory
TEMP_DIR="./temp"
SPECS="$TEMP_DIR/specs"

# Ensure TEMP_DIR and SPECS directories exist
echo "Setting up directories..."
mkdir -p "$TEMP_DIR"
mkdir -p "$SPECS"

# Create folds from test set
echo "Creating folds from test set..."
python scripts/copy_files_from_wavdir_to_multiple_event_dirs.py \
    "$WAV_FOLDER" \
    "$SONG_DETECTION_JSON_PATH" \
    "$TEMP_DIR/folds" \
    --test_set_file "$TEST_SET_FILE" \
    --songs_per_fold 100

# Process each fold
for fold_dir in "$TEMP_DIR/folds"/fold_*; do
    if [ -d "$fold_dir" ]; then
        fold_number=$(basename "$fold_dir" | grep -oP 'fold_\K\d+')
        spec_dir="$SPECS/fold_$fold_number"
        mkdir -p "$spec_dir"
        
        echo "Generating spectrograms for fold $fold_number..."
        echo "Source directory: $fold_dir"
        echo "Destination directory: $spec_dir"
        python src/spectogram_generator.py \
            --src_dir "$fold_dir" \
            --dst_dir "$spec_dir" \
            --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
            || echo "Error generating spectrograms"
        
        save_name="${BIRD_NAME}_fold${fold_number}"
        echo "Running UMAP for Fold: $fold_number"
        python figure_generation_scripts/dim_reduced_birdsong_plots.py \
            --experiment_folder "experiments/$MODEL_NAME" \
            --data_dir "$spec_dir" \
            --save_name "$save_name" \
            --samples 1000
    fi
done

# Delete temp files 
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"