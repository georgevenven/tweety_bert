#!/bin/bash

# Navigate up one directory
cd ..

# variable for model name 
BIRD_NAME="LLB16_order_randomized"
MODEL_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"
WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb16_data/llb16_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/Desk SSD/TweetyBERT/contains_llb.json"

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
python src/spectogram_generator.py --src_dir "$WAV_FOLDER" --dst_dir "$UMAP_FILES" --song_detection_json_path "$SONG_DETECTION_JSON_PATH" --generate_random_files_number 100 --single_threaded false 

# UMAP
python figure_generation_scripts/dim_reduced_birdsong_plots.py \
    --experiment_folder "experiments/$MODEL_NAME" \
    --data_dir "$UMAP_FILES" \
    --save_name "$BIRD_NAME" \
    --samples 5e5 \
    --raw_spectogram false \
    --state_finding_algorithm "HDBSCAN" \
    --context 1000

# Train and save Decoder 
python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

# delete UMAP files
rm -rf "$UMAP_FILES"