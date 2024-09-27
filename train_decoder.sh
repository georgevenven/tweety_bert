#!/bin/bash

# variable for model name 
BIRD_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"
MODEL_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"
WAV_FOLDER="/media/george-vengrovski/disk2/temp/temp1"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/flash-drive/jsons/merged_output.json"

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
    --experiment_folder "experiments/$UMAP_FILES" \
    --data_dir "$TEMP_DIR/$UMAP_FILES" \
    --save_name "$BIRD_NAME"

# Train and save Decoder 
python src/decoder.py --experiment_name "$BIRD_NAME"