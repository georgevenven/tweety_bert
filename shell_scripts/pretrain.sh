#!/bin/bash

# Navigate up one directory
cd ..

# Parameters
INPUT_DIR="/home/george-vengrovski/Documents/testost_pretrain"
SONG_DETECTION_JSON_PATH=None
TEST_PERCENTAGE=20
EXPERIMENT_NAME="TESTOSTERONE_MODEL"
TEMP_DIR="./temp"
TRAIN_FILE_LIST="$TEMP_DIR/train_files.txt"
TEST_FILE_LIST="$TEMP_DIR/test_files.txt"

# Remove the temp directory if it exists to avoid interference
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "Removed existing temporary directory: $TEMP_DIR"
fi

# Create temp directory
mkdir -p "$TEMP_DIR"
echo "Created temporary directory: $TEMP_DIR"

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

# Create train_wav and test_wav directories inside TEMP_DIR
TRAIN_WAV_DIR="$TEMP_DIR/train_wav"
TEST_WAV_DIR="$TEMP_DIR/test_wav"

mkdir -p "$TRAIN_WAV_DIR"
echo "Created training WAV directory: $TRAIN_WAV_DIR"

mkdir -p "$TEST_WAV_DIR"
echo "Created testing WAV directory: $TEST_WAV_DIR"

# Copy train files into TRAIN_WAV_DIR, maintaining directory structure
for file in "${train_files[@]}"; do
    # Get the relative path of the file from INPUT_DIR
    rel_path="${file#$INPUT_DIR/}"
    # Get the directory of the relative path
    dir_path=$(dirname "$rel_path")
    # Create the directory in TRAIN_WAV_DIR
    mkdir -p "$TRAIN_WAV_DIR/$dir_path"
    # Copy the file to TRAIN_WAV_DIR
    cp "$file" "$TRAIN_WAV_DIR/$rel_path"
done

# Copy test files into TEST_WAV_DIR, maintaining directory structure
for file in "${test_files[@]}"; do
    # Get the relative path of the file from INPUT_DIR
    rel_path="${file#$INPUT_DIR/}"
    # Get the directory of the relative path
    dir_path=$(dirname "$rel_path")
    # Create the directory in TEST_WAV_DIR
    mkdir -p "$TEST_WAV_DIR/$dir_path"
    # Copy the file to TEST_WAV_DIR
    cp "$file" "$TEST_WAV_DIR/$rel_path"
done

# Create train_dir and test_dir for spectrograms
TRAIN_DIR="$TEMP_DIR/train_dir"
TEST_DIR="$TEMP_DIR/test_dir"

mkdir -p "$TRAIN_DIR"
echo "Created training spectrogram directory: $TRAIN_DIR"

mkdir -p "$TEST_DIR"
echo "Created testing spectrogram directory: $TEST_DIR"

# Use the same song detection JSON file for all directories
# Process the train WAV directory and output spectrograms to TRAIN_DIR
python3 src/spectogram_generator.py --src_dir "$TRAIN_WAV_DIR" --dst_dir "$TRAIN_DIR" --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# Process the test WAV directory and output spectrograms to TEST_DIR
python3 src/spectogram_generator.py --src_dir "$TEST_WAV_DIR" --dst_dir "$TEST_DIR" --song_detection_json_path "$SONG_DETECTION_JSON_PATH"

# Run TweetyBERT with the generated spectrogram directories
python3 src/TweetyBERT.py --experiment_name "$EXPERIMENT_NAME" --train_dir "$TRAIN_DIR" --test_dir "$TEST_DIR"

# Create experiments directory if it doesn't exist (it should exist by now, but just in case)
EXPERIMENT_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

# Copy train and test file lists to experiments directory
cp "$TRAIN_FILE_LIST" "$EXPERIMENT_DIR/train_files.txt"
cp "$TEST_FILE_LIST" "$EXPERIMENT_DIR/test_files.txt"
echo "Copied train and test file lists to: $EXPERIMENT_DIR"

# Delete the temp directory after processing is complete
rm -rf "$TEMP_DIR"
echo "Deleted temporary directory and its contents: $TEMP_DIR"
