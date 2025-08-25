#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(command, check=True, cwd=None, capture_output=False):
    """Runs a command using subprocess and optionally checks for errors."""
    print(f"Running command: {' '.join(command)}")
    try:
        # Let stdout/stderr pass through unless capture_output is True
        result = subprocess.run(command, check=check, capture_output=capture_output, text=True, cwd=cwd)
        if capture_output:
            if result.stdout:
                print("Output:", result.stdout)
            if result.stderr:
                 # Don't print stderr if it looks like a tqdm progress bar
                if 'it/s]' not in result.stderr and '%' not in result.stderr:
                     print("Error Output:", result.stderr, file=sys.stderr)
        else:
            # Print the last line of tqdm status to stderr
            if result.stderr and result.stderr.strip():
                last_line = result.stderr.strip().split('\n')[-1]
                print(last_line, file=sys.stderr) # Print tqdm status concisely
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

def main(args):
    """Main function to execute the inference pipeline."""

    # --- Get Absolute Paths & Validate ---
    project_root = Path(__file__).resolve().parent
    print(f"Project Root detected as: {project_root}")

    wav_folder = Path(args.wav_folder).resolve()
    # Make song_detection_json optional as in the src/inference.py script
    song_detection_json_path = Path(args.song_detection_json).resolve() if args.song_detection_json else None

    if not wav_folder.is_dir():
        print(f"Error: WAV folder '{args.wav_folder}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)
    if song_detection_json_path and not song_detection_json_path.is_file():
        print(f"Warning: Song detection JSON '{args.song_detection_json}' not found.", file=sys.stderr)
        # Allow running without it, as src/inference.py might handle None
        song_detection_json_path = None

    # --- Define Path to the actual inference script ---
    inference_script_path = project_root / "src" / "inference.py"
    if not inference_script_path.is_file():
        print(f"Error: Cannot find the inference script at '{inference_script_path}'", file=sys.stderr)
        sys.exit(1)

    # --- Build the command ---
    print("\n--- Calling src/inference.py ---")
    inference_cmd = [
        sys.executable, # Use the current python interpreter
        str(inference_script_path),
        "--bird_name", args.bird_name,
        "--wav_dir", str(wav_folder),
        # Pass song_detection_json only if it's provided and valid
        *(["--song_detection_json", str(song_detection_json_path)] if song_detection_json_path else []),
        "--apply_post_processing", str(args.apply_post_processing), # Pass boolean as string
        "--window_size", str(args.window_size),
        "--visualize", str(args.visualize) # Pass boolean as string
    ]

    # --- Run the command ---
    run_command(inference_cmd, cwd=project_root) # Run from project root

    print("\n--- Inference Script Execution Finished ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TweetyBERT Inference")

    # Arguments matching the bash script variables
    parser.add_argument("--bird_name", type=str, required=True,
                        help="Identifier for the bird/dataset.")
    parser.add_argument("--wav_folder", type=str, required=True,
                        help="Path to the directory containing WAV files for inference.")
    parser.add_argument("--song_detection_json", type=str, default=None,
                        help="Path to the JSON file containing song detection data (optional).")
    parser.add_argument("--apply_post_processing", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Apply post-processing (smoothing). Default: True.")
    parser.add_argument("--window_size", type=int, default=200,
                        help="Window size for label smoothing if post-processing is enabled. Default: 200.")
    parser.add_argument("--visualize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Generate visualization plots during inference. Default: True.")

    args = parser.parse_args()

    # Convert boolean args back to string 'True'/'False' for the subprocess call if needed by src/inference.py
    # (Adjust if src/inference.py handles boolean args directly)
    args.apply_post_processing = str(args.apply_post_processing)
    args.visualize = str(args.visualize)

    main(args)
