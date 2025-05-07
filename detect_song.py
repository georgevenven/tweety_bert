#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import shutil
import json
from pathlib import Path

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
    """Clones detector if needed and runs song detection."""

    project_root = Path(__file__).resolve().parent
    print(f"Project Root detected as: {project_root}")

    # Correct argument name from input_dir to wav_dir
    input_dir_path = Path(args.input_dir).resolve()
    output_json_path = Path(args.output_json).resolve()
    song_detector_path = project_root / SONG_DETECTOR_DIR_NAME

    # --- Validate Input Directory ---
    if not input_dir_path.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found or not a directory.", file=sys.stderr)
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

    # Note: The TweetyNet song detector script seems to expect '--input' for the directory
    # and '--output' for the *directory* where results (like JSON per file) are saved.
    # It doesn't directly output a single combined JSON.
    # We will create a temporary output dir for the detector and then merge the results.

    temp_detector_output_dir = project_root / "temp_song_detection_output"
    if temp_detector_output_dir.exists():
        print(f"Removing existing temporary detector output directory: {temp_detector_output_dir}")
        shutil.rmtree(temp_detector_output_dir)
    temp_detector_output_dir.mkdir()
    print(f"Created temporary directory for detector output: {temp_detector_output_dir}")


    detection_cmd = [
        sys.executable,
        str(detector_inference_script),
        "--mode", "local_dir", # Corrected mode argument
        "--input", str(input_dir_path),
        "--output", str(temp_detector_output_dir)          # Detector saves individual files here
        # Add other necessary args for the detector if needed (e.g., --threshold)
    ]
    if args.plot_spec:
        detection_cmd.append("--plot_spec")

    # Run the detector (allow output to stream)
    run_command(detection_cmd, cwd=song_detector_path, capture_output=False) # Run from detector's directory

    # --- Merge Individual JSONs ---
    print(f"\n--- Merging individual JSON outputs from '{temp_detector_output_dir}' ---")
    merged_data = []
    json_files = list(temp_detector_output_dir.glob('*.json'))

    if not json_files:
        print("Warning: No JSON files found in the detector's temporary output directory.", file=sys.stderr)
    else:
        print(f"Found {len(json_files)} JSON files to merge.")
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Assume the detector output format needs slight adjustment
                    # to match the TweetyBERT expected format if necessary.
                    # Example: Rename keys or add 'song_present' based on segments
                    if isinstance(data, dict): # Ensure it's a dictionary
                        if "segments" in data and data["segments"]:
                            data["song_present"] = True
                        else:
                            data["song_present"] = False
                            data["segments"] = [] # Ensure segments key exists

                        # Add default spec_parameters if missing
                        if "spec_parameters" not in data:
                             data["spec_parameters"] = {"step_size": 119, "nfft": 1024} # Add defaults
                        # Add empty syllable_labels if missing
                        if "syllable_labels" not in data:
                             data["syllable_labels"] = {}

                        merged_data.append(data)
                    else:
                        print(f"Warning: Skipping non-dictionary JSON content in {json_file.name}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {json_file.name}. Skipping.", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error processing {json_file.name}: {e}. Skipping.", file=sys.stderr)

    # --- Save Merged JSON ---
    try:
        with open(output_json_path, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Successfully merged {len(merged_data)} results into '{output_json_path}'")
    except Exception as e:
        print(f"Error writing merged JSON to {output_json_path}: {e}", file=sys.stderr)

    # --- Cleanup Temporary Detector Output ---
    print(f"\n--- Cleaning up temporary detector output directory ---")
    shutil.rmtree(temp_detector_output_dir)
    print(f"Deleted temporary directory: {temp_detector_output_dir}")

    print("\n--- Song Detection Script Finished ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TweetyNet Song Detector on a directory of WAV files.")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing WAV files to process.")
    parser.add_argument("--output_json", type=str, default="files/song_detection.json",
                        help="Path to save the final merged JSON output file.")
    parser.add_argument("--plot_spec", action="store_true",
                        help="If set, plot spectrograms during detection.")

    args = parser.parse_args()
    main(args)
