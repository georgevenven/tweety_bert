#!/bin/bash

# Navigate up one directory
cd ..

MAIN_WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/onset_offset_results.json"
MODEL_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"
TEMP_DIR="./temp"
UMAP_FILES="$TEMP_DIR/umap_files"

# Function to process each bird
process_bird() {
    local BIRD_NAME=$1
    local WAV_FOLDER="$MAIN_WAV_FOLDER/${BIRD_NAME}_data/${BIRD_NAME}_songs"
    
    echo "Processing bird: $BIRD_NAME"

    # Create necessary directories if not exist
    if [ ! -d "$TEMP_DIR" ]; then
        mkdir -p "$TEMP_DIR"
        echo "Created temporary directory: $TEMP_DIR"
    fi

    if [ ! -d "$UMAP_FILES" ]; then
        mkdir -p "$UMAP_FILES"
        echo "Created UMAP directory: $UMAP_FILES"
    fi

    # Generate spectrograms for UMAP
    python src/spectogram_generator.py --src_dir "$WAV_FOLDER" --dst_dir "$UMAP_FILES" --song_detection_json_path "$SONG_DETECTION_JSON_PATH" --generate_random_files_number 2500

    # Run UMAP
    python figure_generation_scripts/dim_reduced_birdsong_plots.py \
        --experiment_folder "experiments/$MODEL_NAME" \
        --data_dir "$UMAP_FILES" \
        --save_name "$BIRD_NAME" \
        --samples 1000

    # Train and save Decoder
    python src/decoder.py --experiment_name "$MODEL_NAME" --bird_name "$BIRD_NAME"

    # Delete UMAP files
    rm -rf "$UMAP_FILES"

    # Call the inference script
    python src/inference.py --bird_name "$BIRD_NAME" --wav_dir "$WAV_FOLDER" --song_detection_json "$SONG_DETECTION_JSON_PATH" --apply_post_processing True
}

# Loop through bird folders in the main directory
for BIRD_DIR in "$MAIN_WAV_FOLDER"/*/; do
    BIRD_NAME=$(basename "$BIRD_DIR" | cut -d'_' -f1)
    process_bird "$BIRD_NAME"
done

echo "Batch processing complete."
