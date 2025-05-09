#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import shutil
import json
from pathlib import Path
from glob import glob # Import glob to find the generated JSON files

# --- Configuration ---
SONG_DETECTOR_REPO = "https://github.com/georgevenven/tweety_net_song_detector.git"
SONG_DETECTOR_DIR_NAME = "song_detector" # Local directory name for the repo

# --- Helper Function to Run Commands ---
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
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found. Is git installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred running command {' '.join(command)}: {e}", file=sys.stderr)
        sys.exit(1)

# --- Main Script Logic ---
def main(args):
    """Clones detector if needed, runs song detection, collects JSON paths, and calls external merge script."""

    project_root = Path(__file__).resolve().parent
    print(f"Project Root detected as: {project_root}")

    input_dir_path = Path(args.input_dir).resolve()
    output_json_path = Path(args.output_json).resolve()
    song_detector_path = project_root / SONG_DETECTOR_DIR_NAME
    # Define path to the external merge script
    merge_script_path = project_root / "scripts" / "merge_output_label_jsons.py" # Path relative to project root

    # --- Validate Input Directory ---
    if not input_dir_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)

    # --- Validate Merge Script Path ---
    if not merge_script_path.is_file():
        print(f"Error: Merge script not found at '{merge_script_path}'.", file=sys.stderr)
        sys.exit(1)

    # --- Ensure Output Directory Exists ---
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Check/Clone Song Detector Repository ---
    print(f"\n--- Checking for Song Detector Repository at '{song_detector_path}' ---")
    if not song_detector_path.is_dir():
        print(f"Directory '{SONG_DETECTOR_DIR_NAME}' not found. Cloning repository...")
        clone_cmd = ["git", "clone", SONG_DETECTOR_REPO, str(song_detector_path)]
        run_command(clone_cmd, cwd=project_root) # Clone into the project root
        print("Repository cloned successfully.")
    else:
        print(f"Directory '{SONG_DETECTOR_DIR_NAME}' already exists. Skipping clone.")

    # --- Check if the detector's inference script exists ---
    detector_inference_script = song_detector_path / "src" / "inference.py"
    if not detector_inference_script.is_file():
        print(f"Error: Could not find the inference script at '{detector_inference_script}'.", file=sys.stderr)
        print("Please ensure the repository cloned correctly and contains the expected structure.", file=sys.stderr)
        sys.exit(1)

    # --- Run Song Detection ---
    print(f"\n--- Running Song Detection on '{input_dir_path}' ---")
    print(f"Output will be saved to: '{output_json_path}'")

    temp_detector_output_dir = project_root / "temp_song_detection_output"
    if temp_detector_output_dir.exists():
        print(f"Removing existing temporary detector output directory: {temp_detector_output_dir}")
        shutil.rmtree(temp_detector_output_dir)
    temp_detector_output_dir.mkdir()
    print(f"Created temporary directory for detector output: {temp_detector_output_dir}")


    detection_cmd = [
        sys.executable,
        str(detector_inference_script),
        "--mode", "local_dir",
        "--input", str(input_dir_path),
        "--output", str(temp_detector_output_dir) # Detector saves individual files here
    ]
    if args.plot_spec:
        detection_cmd.append("--plot_spec")

    # Run the detector (allow output to stream)
    run_command(detection_cmd, cwd=song_detector_path, capture_output=False)

    # --- Collect the paths of the generated JSON files ---
    print(f"\n--- Collecting JSON file paths from '{temp_detector_output_dir}' ---")
    # Use glob to find all .json files and convert them to strings
    json_file_paths = [str(p.resolve()) for p in temp_detector_output_dir.glob('*.json')] # Use resolve() for absolute paths

    if not json_file_paths:
        print("Warning: No JSON files found in the detector's temporary output directory.", file=sys.stderr)
        # Create an empty output file if no inputs were generated
        with open(output_json_path, 'w') as f:
            json.dump([], f)
        print(f"Created empty output file: {output_json_path}")
    else:
        # --- Merge Individual JSONs using the external script ---
        print(f"Found {len(json_file_paths)} JSON files. Merging using '{merge_script_path.name}'...")

        # Build the command for the merge script
        # The command structure is: python merge_script.py [input_file1 input_file2 ...] output_file
        merge_cmd = [
            sys.executable,
            str(merge_script_path) # Path to the merge script
        ] + json_file_paths + [ # Add the list of input file paths here
            str(output_json_path) # Add the final output file path last
        ]

        # Run the merge script
        run_command(merge_cmd, cwd=project_root) # Run from project root

    # --- Cleanup Temporary Detector Output ---
    # Optional: Keep the temp dir for debugging if needed by commenting out the next two lines
    print(f"\n--- Cleaning up temporary detector output directory ---")
    shutil.rmtree(temp_detector_output_dir)
    print(f"Deleted temporary directory: {temp_detector_output_dir}")
    # print(f"Keeping temporary directory for inspection: {temp_detector_output_dir}") # Keep for debugging

    print("\n--- Song Detection Script Finished ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TweetyNet Song Detector on a directory of WAV files and merge results using external script.")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing WAV files to process.")
    parser.add_argument("--output_json", type=str, default="files/song_detection.json",
                        help="Path to save the final merged JSON output file.")
    parser.add_argument("--plot_spec", action="store_true",
                        help="If set, plot spectrograms during detection.")

    args = parser.parse_args()
    main(args)
