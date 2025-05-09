import os
import random
import argparse
from pathlib import Path

def collect_all_files(input_dir):
    """
    Recursively collects all WAV, OGG, and MP3 files in the input directory.

    Args:
        input_dir (str): The directory to search for files.

    Returns:
        list: A list of file paths for audio files only.
    """
    file_paths = []
    audio_extensions = {'.wav', '.ogg', '.mp3'}
    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in audio_extensions:
                file_paths.append(os.path.join(root, name))
    return file_paths

def split_files(input_dir, test_percentage):
    """
    Splits the files in the input directory into training and testing sets based on the specified percentage.

    Args:
        input_dir (str): The directory containing files or subdirectories with files.
        test_percentage (float): The percentage of files to be used for testing.

    Returns:
        tuple: Two lists of file paths, one for training and one for testing.
    """
    # Collect all files in the input directory, including nested files
    all_files = collect_all_files(input_dir)

    # Calculate the number of files to be used for testing
    total_files = len(all_files)
    num_test_files = int(total_files * test_percentage / 100)

    # Shuffle the list of files
    random.shuffle(all_files)

    # Split the shuffled list into test and train sets
    test_files = all_files[:num_test_files]
    train_files = all_files[num_test_files:]

    return train_files, test_files

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split files into training and testing sets.")
    parser.add_argument("input_dir", type=str, help="The directory containing files or subdirectories with files.")
    parser.add_argument("test_percentage", type=float, help="The percentage of files to be used for testing.")
    parser.add_argument("--train_output", type=str, default="train_files.txt", help="File to write train file paths.")
    parser.add_argument("--test_output", type=str, default="test_files.txt", help="File to write test file paths.")
    parser.add_argument("--full_paths", action="store_true", help="Store full paths instead of basenames")
    args = parser.parse_args()

    # Validate the input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist.")
        return

    # Validate the test percentage
    if args.test_percentage < 0 or args.test_percentage > 100:
        print("Error: Test percentage must be a number between 0 and 100.")
        return

    # Split the files into training and testing sets
    train_files, test_files = split_files(args.input_dir, args.test_percentage)

    # Write the lists of files to the specified output files
    with open(args.train_output, 'w') as f:
        for file_path in train_files:
            path = file_path if args.full_paths else Path(file_path).name
            f.write(f"{path}\n")

    with open(args.test_output, 'w') as f:
        for file_path in test_files:
            path = file_path if args.full_paths else Path(file_path).name
            f.write(f"{path}\n")

    print(f"Train files written to {args.train_output}")
    print(f"Test files written to {args.test_output}")

if __name__ == "__main__":
    main()
