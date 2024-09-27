#!/bin/bash

# variable for model name 
BIRD_NAME="test"
MODEL_NAME="LLB16_Whisperseg_NoNormNoThresholding"
WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/home/george-vengrovski/Documents/projects/tweety_net_song_detector/output/onset_offset_results.json"

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
python src/spectogram_generator.py --src_dir "$WAV_FOLDER" --dst_dir "$UMAP_FILES" --song_detection_json_path "$SONG_DETECTION_JSON_PATH" --generate_random_files_number 1000

# UMAP
python figure_generation_scripts/dim_reduced_birdsong_plots.py \
    --experiment_folder "experiments/$MODEL_NAME" \
    --data_dir "$UMAP_FILES" \
    --save_name "$BIRD_NAME" \
    --samples 5000

# delete UMAP files
rm -rf "$TEMP_DIR/$UMAP_FILES"

# Train and save Decoder 
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"