#!/bin/bash

# Params 
INPUT_DIR="/path/to/bird_ids"
SONG_DETECTION_JSON_PATH="/home/george-vengrovski/Documents/projects/tweety_net_song_detector/output/onset_offset_results.json"
TEST_PERCENTAGE=20

# Call the Python script and capture the output
output=$(python3 scripts/seperate_bird_ids_into_train_and_test_for_pretrain.py "$INPUT_DIR" "$TEST_PERCENTAGE")

# Split the output into train and test directories
train_dirs=$(echo "$output" | sed -n '1p')
test_dirs=$(echo "$output" | sed -n '2p')

# Use the directories in subsequent commands
echo "Train directories: $train_dirs"
echo "Test directories: $test_dirs"

# Create temp, train, and test directories if they do not exist
TEMP_DIR="/temp"
TRAIN_DIR="$TEMP_DIR/train_dir"
TEST_DIR="$TEMP_DIR/test_dir"

if [ ! -d "$TEMP_DIR" ]; then
    mkdir -p "$TEMP_DIR"
    echo "Created temporary directory: $TEMP_DIR"
fi

if [ ! -d "$TRAIN_DIR" ]; then
    mkdir -p "$TRAIN_DIR"
    echo "Created training directory: $TRAIN_DIR"
fi

if [ ! -d "$TEST_DIR" ]; then
    mkdir -p "$TEST_DIR"
    echo "Created testing directory: $TEST_DIR"
fi

for dir in "${train_dirs[@]}"; do
    python3 src/spectogram_generator.py --src_dir "$dir" --dst_dir "$TEMP_DIR/train_dir" --song_detection_json_path "$dir/song_detection.json"
done

for dir in "${test_dirs[@]}"; do
    python3 src/spectogram_generator.py --src_dir "$dir" --dst_dir "$TEMP_DIR/test_dir" --song_detection_json_path "$dir/song_detection.json"
done

python3 src/TweetyBERT.py --experiment_name "TweetyBERT_Pretrain_LLB_AreaX_FallSong" --train_dir "/temp/train_dir" --test_dir "/temp/test_dir"