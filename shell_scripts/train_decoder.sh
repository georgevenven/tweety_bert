#!/bin/bash

# Navigate up one directory
cd ..

# variable for model name 
BIRD_NAME="llb3_GMM_Test"
MODEL_NAME="LLB_Model_For_Paper"
WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_songs"
SONG_DETECTION_JSON_PATH="merged_output.json"

# generate spectrograms for UMAP
TEMP_DIR="./temp"
UMAP_FILES="$TEMP_DIR/umap_files"

if [ ! -d "$TEMP_DIR" ]; then
    mkdir -p "$TEMP_DIR"
    echo "Created temporary directory: $TEMP_DIR"
fi

if [ ! -d "$UMAP_FILES" ]; then
    mkdir -p "$UMAP_FILES"
    echo "Created training directory: $UMAP_FILES"
fi

# point to wave folder, and generate 1000 files 
python src/spectogram_generator.py --src_dir "$WAV_FOLDER" --dst_dir "$UMAP_FILES" --song_detection_json_path "$SONG_DETECTION_JSON_PATH" --generate_random_files_number 250 --single_threaded false 

# UMAP
python figure_generation_scripts/dim_reduced_birdsong_plots.py \
    --experiment_folder "experiments/$MODEL_NAME" \
    --data_dir "$UMAP_FILES" \
    --save_name "$BIRD_NAME" \
    --samples 1e5 \
    --raw_spectogram false \
    --state_finding_algorithm "GHMM"

# Train and save Decoder 
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

# delete UMAP files
rm -rf "$UMAP_FILES"