#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shutil
import json
from pathlib import Path

# --- Configuration ---
SONG_DETECTOR_REPO = "https://github.com/georgevenven/tweety_net_song_detector.git"
SONG_DETECTOR_DIR_NAME = "song_detector"

# --- Helper Function to Run Commands ---
def run_command(command, cwd=None):
    """Runs a command using subprocess."""
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, cwd=cwd)

# --- Main Script Logic ---
def main(args):
    project_root = Path(__file__).resolve().parent

    input_dir_path = Path(args.input_dir).resolve()
    output_json_path = Path(args.output_json).resolve()

    # SAFETY FIX: If output path is a directory, append a default filename
    if output_json_path.is_dir():
        print(f"Warning: Provided output path '{output_json_path}' is a directory.")
        output_json_path = output_json_path / "song_detection_results.json"
        print(f"Saving to: {output_json_path}")

    song_detector_path = project_root / SONG_DETECTOR_DIR_NAME
    merge_script_path = project_root / "scripts" / "merge_output_label_jsons.py"

    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not song_detector_path.is_dir():
        clone_cmd = ["git", "clone", SONG_DETECTOR_REPO, str(song_detector_path)]
        run_command(clone_cmd, cwd=project_root)

    detector_inference_script = song_detector_path / "src" / "inference.py"

    temp_detector_output_dir = project_root / "temp_song_detection_output"
    if temp_detector_output_dir.exists():
        shutil.rmtree(temp_detector_output_dir)
    temp_detector_output_dir.mkdir()

    detection_cmd = [
        sys.executable,
        str(detector_inference_script),
        "--mode", "local_dir",
        "--input", str(input_dir_path),
        "--output", str(temp_detector_output_dir)
    ]
    if args.plot_spec:
        detection_cmd.append("--plot_spec")

    run_command(detection_cmd, cwd=song_detector_path)

    json_file_paths = [str(p.resolve()) for p in temp_detector_output_dir.glob('*.json')]

    if not json_file_paths:
        with open(output_json_path, 'w') as f:
            json.dump([], f)
        shutil.rmtree(temp_detector_output_dir)
    else:
        merge_cmd = [
            sys.executable,
            str(merge_script_path)
        ] + json_file_paths + [
            str(output_json_path)
        ]
        
        # SAFETY FIX: Check return code and only delete if successful
        result = subprocess.run(merge_cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("Merge successful. Cleaning up temporary files.")
            shutil.rmtree(temp_detector_output_dir)
        else:
            print(f"Temporary output files HAVE NOT BEEN DELETED so you can recover your data.")
            print(f"Location: {temp_detector_output_dir}")

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