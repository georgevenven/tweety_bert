#!/bin/bash

# Navigate up one directory
cd ..

# Parameters
INPUT_DIR="/home/george-vengrovski/Documents/testost_pretrain"
SONG_DETECTION_JSON_PATH=None
TEST_PERCENTAGE=20
EXPERIMENT_NAME="TESTOSTERONE_MODEL"
TRAIN_FILE_LIST="train_files.txt"
TEST_FILE_LIST="test_files.txt"

# Call the Python script to split files
python3 scripts/split_files_into_train_and_test.py "$INPUT_DIR" "$TEST_PERCENTAGE" --train_output "$TRAIN_FILE_LIST" --test_output "$TEST_FILE_LIST"

# Read the file lists into arrays
mapfile -t train_files < "$TRAIN_FILE_LIST"
mapfile -t test_files < "$TEST_FILE_LIST"

# Use the files in subsequent commands
echo "Train files:"
printf '%s\n' "${train_files[@]}"
echo "Test files:"
printf '%s\n' "${test_files[@]}"

# Create temp, train, and test directories if they do not exist at the same level as the script
TEMP_DIR="./temp"
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

# Copy train files into TRAIN_DIR, maintaining directory structure
for file in "${train_files[@]}"; do
    # Get the relative path of the file from INPUT_DIR
    rel_path="${file#$INPUT_DIR/}"
    # Get the directory of the relative path
    dir_path=$(dirname "$rel_path")
    # Create the directory in TRAIN_DIR
    mkdir -p "$TRAIN_DIR/$dir_path"
    # Copy the file to TRAIN_DIR
    cp "$file" "$TRAIN_DIR/$rel_path"
done

# Copy test files into TEST_DIR, maintaining directory structure
for file in "${test_files[@]}"; do
    # Get the relative path of the file from INPUT_DIR
    rel_path="${file#$INPUT_DIR/}"
    # Get the directory of the relative path
    dir_path=$(dirname "$rel_path")
    # Create the directory in TEST_DIR
    mkdir -p "$TEST_DIR/$dir_path"
    # Copy the file to TEST_DIR
    cp "$file" "$TEST_DIR/$rel_path"
done

# Use the same song detection JSON file for all directories
# Process the train directory
python3 src/spectogram_generator.py --src_dir "$TRAIN_DIR" --dst_dir "$TRAIN_DIR" --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# Process the test directory
python3 src/spectogram_generator.py --src_dir "$TEST_DIR" --dst_dir "$TEST_DIR" --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# Run TweetyBERT
python3 src/TweetyBERT.py --experiment_name "$EXPERIMENT_NAME" --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR"
