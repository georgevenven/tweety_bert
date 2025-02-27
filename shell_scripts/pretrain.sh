#!/usr/bin/env bash

# Exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# Navigate up one directory
cd ..

# Required Parameters
INPUT_DIR="/media/george-vengrovski/Desk SSD/TweetyBERT/songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/Desk SSD/TweetyBERT/song_detecton_database.json"
TEST_PERCENTAGE=20
EXPERIMENT_NAME="TweetyBERT_Paper_Yarden_Model"

# Change Default Parameters (if needed)
BATCH_SIZE=42                    # Training batch size
LEARNING_RATE=3e-4              # Learning rate for training
MULTI_THREAD=true               # Set to false for single-thread spectrogram generation
STEP_SIZE=119                   # Step size for spectrogram generation
NFFT=1024                       # Number of FFT points for spectrogram

# Don't Change
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

# 1. Split files into train and test (Python writes train_files.txt and test_files.txt)
# Use --full_paths to ensure we get complete paths relative to INPUT_DIR
python3 scripts/split_files_into_train_and_test.py "$INPUT_DIR" "$TEST_PERCENTAGE" \
        --train_output "$TRAIN_FILE_LIST" \
        --test_output "$TEST_FILE_LIST" \
        --full_paths

# 2. Read the file lists into arrays
train_files=()
while IFS= read -r line; do
  train_files+=( "$line" )
done < "$TRAIN_FILE_LIST"

test_files=()
while IFS= read -r line; do
  test_files+=( "$line" )
done < "$TEST_FILE_LIST"

# 3. Print only counts (not all filenames)
echo "Found ${#train_files[@]} training files and ${#test_files[@]} testing files."

# 4. Create directories for WAV files
TRAIN_WAV_DIR="$TEMP_DIR/train_wav"
TEST_WAV_DIR="$TEMP_DIR/test_wav"

mkdir -p "$TRAIN_WAV_DIR"
mkdir -p "$TEST_WAV_DIR"

# 5. Copy train files into TRAIN_WAV_DIR, preserving directory structure
for file in "${train_files[@]}"; do
    # Get the full source path
    src_path="$INPUT_DIR/$file"
    # Create the destination directory structure
    dst_dir="$TRAIN_WAV_DIR/$(dirname "$file")"
    mkdir -p "$dst_dir"
    # Copy the file
    cp "$src_path" "$dst_dir/$(basename "$file")"
done

# 6. Copy test files into TEST_WAV_DIR, preserving directory structure
for file in "${test_files[@]}"; do
    # Get the full source path
    src_path="$INPUT_DIR/$file"
    # Create the destination directory structure
    dst_dir="$TEST_WAV_DIR/$(dirname "$file")"
    mkdir -p "$dst_dir"
    # Copy the file
    cp "$src_path" "$dst_dir/$(basename "$file")"
done

# 7. Create train_dir and test_dir for spectrograms
TRAIN_DIR="$TEMP_DIR/train_dir"
TEST_DIR="$TEMP_DIR/test_dir"

mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

# Determine number of processes for spectrogram generation
PROCESS_COUNT=1
if [ "$MULTI_THREAD" = true ]; then
    PROCESS_COUNT=$(nproc)
fi

# Generate spectrograms (train + test)
python3 src/spectogram_generator.py \
        --src_dir "$TRAIN_WAV_DIR" \
        --dst_dir "$TRAIN_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        $([ "$MULTI_THREAD" = true ] && echo "--single_threaded")

python3 src/spectogram_generator.py \
        --src_dir "$TEST_WAV_DIR" \
        --dst_dir "$TEST_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        $([ "$MULTI_THREAD" = true ] && echo "--single_threaded")

# Run TweetyBERT
python3 src/TweetyBERT.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --train_dir "$TRAIN_DIR" \
        --test_dir "$TEST_DIR" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE"

# 10. Save file lists into the experiment folder
EXPERIMENT_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

cp "$TRAIN_FILE_LIST" "$EXPERIMENT_DIR/train_files.txt"
cp "$TEST_FILE_LIST" "$EXPERIMENT_DIR/test_files.txt"
echo "Copied train and test file lists to: $EXPERIMENT_DIR"

# 11. Clean up temp directory
rm -rf "$TEMP_DIR"
echo "Deleted temporary directory and its contents: $TEMP_DIR"