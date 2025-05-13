#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
import sys
from pathlib import Path
import multiprocessing # To determine nproc for spectrogram generation

# --- Helper Function to Run Commands ---
def run_command(command, check=True, cwd=None):
    """Runs a command using subprocess and optionally checks for errors."""
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=check, text=True, cwd=cwd, stdout=None, stderr=None)
        if result.stdout:
            print("Output:", result.stdout)
        if result.stderr:
            print("Error Output:", result.stderr, file=sys.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout, file=sys.stderr)
        if e.stderr:
            print("STDERR:", e.stderr, file=sys.stderr)
        sys.exit(1) # Exit if command fails and check=True
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

# --- Main Script Logic ---
def main(args):
    """Main function to execute the pretraining pipeline."""

    # --- Get Absolute Paths ---
    # Since this script (pretrain.py) is now in the root, its parent IS the root.
    project_root = Path(__file__).resolve().parent
    print(f"Project Root detected as: {project_root}")
    
    input_dir = Path(args.input_dir).resolve()
    song_detection_json_path = Path(args.song_detection_json_path).resolve() if args.song_detection_json_path else None
    
    if not input_dir.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)
    if song_detection_json_path and not song_detection_json_path.is_file():
        print(f"Warning: Song detection JSON '{args.song_detection_json_path}' not found.", file=sys.stderr)
        # Decide if this is critical or just optional
        # sys.exit(1) 
        song_detection_json_path = None # Treat as if not provided

    # --- Define Paths ---
    temp_dir = project_root / "temp_pretrain"
    train_file_list = temp_dir / "train_files.txt"
    test_file_list = temp_dir / "test_files.txt"
    train_dir_specs = temp_dir / "train_dir"
    test_dir_specs = temp_dir / "test_dir"
    final_experiment_dir = project_root / "experiments" / args.experiment_name
    
    split_script_path = project_root / "scripts" / "split_files_into_train_and_test.py"
    spec_gen_script_path = project_root / "src" / "spectogram_generator.py"
    tweetybert_script_path = project_root / "src" / "TweetyBERT.py"

    # --- Cleanup ---
    if temp_dir.exists():
        print(f"Removing existing temporary directory: {temp_dir}")  # temp comment
        shutil.rmtree(temp_dir)  # temp comment

    # --- Create Temp Directory ---
    print(f"Creating temporary directory: {temp_dir}")  # temp comment
    temp_dir.mkdir(parents=True, exist_ok=True)  # temp comment

    # --- 1. Split files into train and test ---
    split_cmd = [  # temp comment
        sys.executable,  # temp comment
        str(split_script_path),  # temp comment
        str(input_dir),  # temp comment
        str(args.test_percentage),  # temp comment
        "--train_output", str(train_file_list),  # temp comment
        "--test_output", str(test_file_list),  # temp comment
        "--full_paths"  # temp comment
    ]  # temp comment
    run_command(split_cmd)  # temp comment

    # --- 2. Print counts ---
    try:  # temp comment
        with open(train_file_list, 'r') as f:  # temp comment
            train_count = sum(1 for _ in f)  # temp comment
        with open(test_file_list, 'r') as f:  # temp comment
            test_count = sum(1 for _ in f)  # temp comment
        print(f"Found {train_count} training files and {test_count} testing files.")  # temp comment
    except FileNotFoundError:  # temp comment
        print("Error: Train/Test list files not found after split step.", file=sys.stderr)  # temp comment
        sys.exit(1)  # temp comment

    # --- 3. Create directories for spectrograms ---
    train_dir_specs.mkdir(exist_ok=True)  # temp comment
    test_dir_specs.mkdir(exist_ok=True)  # temp comment
    print(f"Created spectrogram directory: {train_dir_specs}")  # temp comment
    print(f"Created spectrogram directory: {test_dir_specs}")  # temp comment
    
    # --- 4. Generate Spectrograms ---
    single_threaded_arg = "true" if not args.multi_thread else "false"  # temp comment
    spec_common_args = [  # temp comment
        "--step_size", str(args.step_size),  # temp comment
        "--nfft", str(args.nfft),  # temp comment
        "--single_threaded", single_threaded_arg  # temp comment
    ]  # temp comment
    if song_detection_json_path:  # temp comment
        spec_common_args.extend(["--song_detection_json_path", str(song_detection_json_path)])  # temp comment
    else:  # temp comment
         spec_common_args.extend(["--song_detection_json_path", "None"])  # temp comment
    spec_train_cmd = [  # temp comment
        sys.executable,  # temp comment
        str(spec_gen_script_path),  # temp comment
        "--file_list", str(train_file_list),  # temp comment
        "--dst_dir", str(train_dir_specs),  # temp comment
    ] + spec_common_args  # temp comment
    run_command(spec_train_cmd)  # temp comment
    spec_test_cmd = [  # temp comment
        sys.executable,  # temp comment
        str(spec_gen_script_path),  # temp comment
        "--file_list", str(test_file_list),  # temp comment
        "--dst_dir", str(test_dir_specs),  # temp comment
    ] + spec_common_args  # temp comment
    run_command(spec_test_cmd)  # temp comment

    # --- 5. Run TweetyBERT Training ---
    print("\n--- Step 5: Running TweetyBERT training ---")
    tweetybert_cmd = [
        sys.executable,
        str(tweetybert_script_path),
        "--experiment_name", args.experiment_name,
        "--train_dir", str(train_dir_specs),
        "--test_dir", str(test_dir_specs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--m", str(args.m), # Assuming m=100 from bash script
        "--context", str(args.context),
        "--continue_training", str(args.continue_training),
        "--early_stopping", str(args.early_stopping),
        "--patience", str(args.patience),
        "--trailing_avg_window", str(args.trailing_avg_window)
    ]
    run_command(tweetybert_cmd)

    # --- 6. Save file lists into the experiment folder ---
    print("\n--- Step 6: Saving file lists ---")
    final_experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_file_list, final_experiment_dir / "train_files.txt")
    shutil.copy2(test_file_list, final_experiment_dir / "test_files.txt")
    print(f"Copied train and test file lists to: {final_experiment_dir}")

    # --- 7. Clean up temp directory ---
    print("\n--- Step 7: Cleaning up temporary directory ---")
    shutil.rmtree(temp_dir)  # temp comment
    print(f"Deleted temporary directory and its contents: {temp_dir}")

    print("\n--- Pretraining Pipeline Completed Successfully! ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TweetyBERT Pretraining Pipeline")

    # Required parameters
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the folder containing WAV files (can have subdirectories).")
    

    # Default changeable parameters
    parser.add_argument("--song_detection_json_path", type=str, default=None, # Made optional
                        help="Path to the JSON file containing song detection data (onsets/offsets).")
    parser.add_argument("--test_percentage", type=float, default=20,
                        help="Percentage of files to allocate to the test set (0-100).")
    parser.add_argument("--experiment_name", type=str, default="TweetyBERT_Pretrain_Experiment",
                        help="Name for the training experiment. Results will be saved under experiments/<name>.")
    parser.add_argument("--batch_size", type=int, default=42,
                        help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for training.")
    parser.add_argument("--multi_thread", action='store_true',
                        help="Use multi-threading for spectrogram generation (uses nproc). Default is single-threaded.")
    parser.add_argument("--step_size", type=int, default=119,
                        help="Step size (hop length) for spectrogram generation.")
    parser.add_argument("--nfft", type=int, default=1024,
                        help="Number of FFT points for spectrogram generation.")
    # Added m and context based on the bash script's TweetyBERT call
    parser.add_argument("--m", type=int, default=250,
                        help="Parameter 'm' for TweetyBERT.")
    parser.add_argument("--context", type=int, default=1000,
                        help="Parameter 'context' for TweetyBERT.")
    parser.add_argument("--early_stopping", type=bool, default=False,
                        help="Whether to use early stopping. If true, means yes.")
    parser.add_argument("--patience", type=int, default=8,
                        help="Number of epochs to wait before early stopping.")
    parser.add_argument("--trailing_avg_window", type=int, default=200,
                        help="Number of epochs to consider for trailing average.")
    
    # functionality not implemented yet, spec gen process needs to pull the train, test txt files from the experiment folder
    parser.add_argument("--continue_training", type=bool, default=False,
                        help="Whether to continue training from a checkpoint. If true, means yes.")

    args = parser.parse_args()

    # Basic validation
    if not 0 <= args.test_percentage <= 100:
        print("Error: test_percentage must be between 0 and 100.", file=sys.stderr)
        sys.exit(1)

    main(args)