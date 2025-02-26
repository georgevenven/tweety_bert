#!/bin/bash

# NAVIGATE UP ONE DIRECTORY
cd ..

# CONSTANT VARIABLES
BIRD_NAMES=("llb3" "llb11" "llb16")
BASE_WAV_FOLDER="/media/george-vengrovski/George-SSD"
SONG_DETECTION_JSON_PATH="merged_output.json"
MODEL_NAME="LLB_Model_For_Paper"

TEMP_DIR="./temp"
UMAP_FILES="$TEMP_DIR/UMAP_FILES"

mkdir -p "$TEMP_DIR"

for BIRD in "${BIRD_NAMES[@]}"; do
    WAV_FOLDER="$BASE_WAV_FOLDER/${BIRD}_left_comp/${BIRD}_songs"
    
    echo "PROCESSING ${BIRD} RN..."
    mkdir -p "$UMAP_FILES"

    # GENERATE SPECTROGRAMS FOR UMAP
    python src/spectogram_generator.py \
        --src_dir "$WAV_FOLDER" \
        --dst_dir "$UMAP_FILES" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --generate_random_files_number 350 \
        --single_threaded false

    # REMOVE EMPTY YARDEN DATA FILES
    python scripts/remove_empty_yarden_data.py "$UMAP_FILES"

    for RAW_SPEC in true false; do
        echo "RUNNING UMAP WITH RAW_SPECTOGRAM=${RAW_SPEC}"
        python figure_generation_scripts/dim_reduced_birdsong_plots.py \
            --experiment_folder "experiments/$MODEL_NAME" \
            --data_dir "$UMAP_FILES" \
            --save_name "${BIRD}_raw_${RAW_SPEC}" \
            --samples 1e6 \
            --raw_spectogram "$RAW_SPEC"
    done

    echo "CLEANING UP..."
    rm -rf "$UMAP_FILES"
done

echo "ALL DONE!"
