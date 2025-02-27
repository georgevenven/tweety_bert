#!/usr/bin/env bash
# exit on errors, undefined variables, and propagate errors in pipelines
set -euo pipefail

# navigate up one directory
cd ..

# required parameters
INPUT_DIR="/media/george-vengrovski/disk1/canary/canary_recordings/USA5210"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/Desk SSD/TweetyBERT/contains_seasonality.json"
TEST_PERCENTAGE=20
EXPERIMENT_NAME="delete_me"

# change default parameters (if needed)
BATCH_SIZE=42                    # training batch size
LEARNING_RATE=3e-4              # learning rate for training
MULTI_THREAD=true               # set to false for single-thread spectrogram generation
STEP_SIZE=119                   # step size for spectrogram generation
NFFT=1024                       # number of fft points for spectrogram

# don't change
TEMP_DIR="./temp"
TRAIN_FILE_LIST="$TEMP_DIR/train_files.txt"
TEST_FILE_LIST="$TEMP_DIR/test_files.txt"

# remove the temp directory if it exists to avoid interference
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
    echo "removed existing temporary directory: $TEMP_DIR"
fi

# create temp directory
mkdir -p "$TEMP_DIR"
echo "created temporary directory: $TEMP_DIR"

# 1. split files into train and test (python writes train_files.txt and test_files.txt)
python3 scripts/split_files_into_train_and_test.py "$INPUT_DIR" "$TEST_PERCENTAGE" \
        --train_output "$TRAIN_FILE_LIST" \
        --test_output "$TEST_FILE_LIST"

# 2. read the file lists into arrays
train_files=()
while IFS= read -r line; do
  train_files+=( "$line" )
done < "$TRAIN_FILE_LIST"

test_files=()
while IFS= read -r line; do
  test_files+=( "$line" )
done < "$TEST_FILE_LIST"

# 3. print only counts (not all filenames)
echo "found ${#train_files[@]} training files and ${#test_files[@]} testing files."

# 4. create directories for wav files
TRAIN_WAV_DIR="$TEMP_DIR/train_wav"
TEST_WAV_DIR="$TEMP_DIR/test_wav"

mkdir -p "$TRAIN_WAV_DIR"
mkdir -p "$TEST_WAV_DIR"

# 5. copy train files into TRAIN_WAV_DIR (flat structure)
for file in "${train_files[@]}"; do
    base=$(basename "$file")
    # search recursively for the file in the input directory
    src_path=$(find "$INPUT_DIR" -type f -name "$base" -print -quit)
    if [ -z "$src_path" ]; then
         echo "warning: file $base not found in $INPUT_DIR" >&2
         continue
    fi
    cp "$src_path" "$TRAIN_WAV_DIR/"
done

# 6. similarly, copy test files into TEST_WAV_DIR (flat structure)
for file in "${test_files[@]}"; do
    base=$(basename "$file")
    src_path=$(find "$INPUT_DIR" -type f -name "$base" -print -quit)
    if [ -z "$src_path" ]; then
         echo "warning: file $base not found in $INPUT_DIR" >&2
         continue
    fi
    cp "$src_path" "$TEST_WAV_DIR/"
done

# 7. create train_dir and test_dir for spectrograms
TRAIN_DIR="$TEMP_DIR/train_dir"
TEST_DIR="$TEMP_DIR/test_dir"

mkdir -p "$TRAIN_DIR"
mkdir -p "$TEST_DIR"

# determine number of processes for spectrogram generation
PROCESS_COUNT=1
if [ "$MULTI_THREAD" = true ]; then
    PROCESS_COUNT=$(nproc)
fi

# generate spectrograms (train + test)
python3 src/spectogram_generator.py \
        --src_dir "$TRAIN_WAV_DIR" \
        --dst_dir "$TRAIN_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --single_threaded False 
python3 src/spectogram_generator.py \
        --src_dir "$TEST_WAV_DIR" \
        --dst_dir "$TEST_DIR" \
        --song_detection_json_path "$SONG_DETECTION_JSON_PATH" \
        --step_size "$STEP_SIZE" \
        --nfft "$NFFT" \
        --single_threaded False 

# run tweetybert
python3 src/TweetyBERT.py \
        --experiment_name "$EXPERIMENT_NAME" \
        --train_dir "$TRAIN_DIR" \
        --test_dir "$TEST_DIR" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE"

# 10. save file lists into the experiment folder
EXPERIMENT_DIR="experiments/$EXPERIMENT_NAME"
mkdir -p "$EXPERIMENT_DIR"

cp "$TRAIN_FILE_LIST" "$EXPERIMENT_DIR/train_files.txt"
cp "$TEST_FILE_LIST" "$EXPERIMENT_DIR/test_files.txt"
echo "copied train and test file lists to: $EXPERIMENT_DIR"

# 11. clean up temp directory
rm -rf "$TEMP_DIR"
echo "deleted temporary directory and its contents: $TEMP_DIR"
