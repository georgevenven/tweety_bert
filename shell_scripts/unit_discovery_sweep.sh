#!/bin/bash

# Navigate up one directory
cd ..

# Variable for model name 
BIRD_NAME="llb3"
MODEL_NAME="TweetyBERT_Pretrain_LLB_AreaX_FallSong"
WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json"


# Create temp directory
TEMP_DIR="./temp"
SPECS="$TEMP_DIR/specs"

# Ensure TEMP_DIR and SPECS directories exist
echo "Setting up directories..."
mkdir -p "$TEMP_DIR"
mkdir -p "$SPECS"

# Define search parameters
declare -a PCA_COMPONENTS=(32 64)
declare -a MIN_CLUSTER_SIZE=(500 5000)

# # Point to wave folder, and generate 1000 files 
# echo "Generating spectrograms..."
python src/spectogram_generator.py --src_dir "$WAV_FOLDER" --dst_dir "$SPECS" --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# Call count_timebins.py and store the return value
echo "Counting timebins and generating folds..."
python scripts/count_timebins.py --dir_path "$SPECS" --target_timebins 500000 --temp_folds_path "$TEMP_DIR/folds" | {
    while IFS= read -r line; do
        if [[ $line == "FOLD_PATHS_START" ]]; then
            read_paths=true
        elif [[ $line == "FOLD_PATHS_END" ]]; then
            read_paths=false
        elif [[ $read_paths == true ]]; then
            paths+=("$line")
        fi
    done

    for path in "${paths[@]}"; do
        echo "Processing fold at path: $path"
        # Extract fold number from path
        fold_number=$(basename "$path" | grep -oP 'fold_\K\d+')
        # Ensure $path is a directory
        if [ -d "$path" ]; then
            for pca in "${PCA_COMPONENTS[@]}"; do
                for cluster_size in "${MIN_CLUSTER_SIZE[@]}"; do
                    save_name="${BIRD_NAME}_fold${fold_number}_pca_${pca}_cluster_${cluster_size}"
                    echo "Running UMAP with PCA: $pca, Min Cluster Size: $cluster_size, Fold: $fold_number"
                    python figure_generation_scripts/dim_reduced_birdsong_plots.py \
                        --experiment_folder "experiments/$MODEL_NAME" \
                        --data_dir "$path" \
                        --save_name "$save_name" \
                        --samples 500000 \
                        --pca_components "$pca" \
                        --min_cluster_size "$cluster_size"
                done
            done
        else
            echo "Error: $path is not a directory"
        fi
    done
}

# Delete temp files 
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
