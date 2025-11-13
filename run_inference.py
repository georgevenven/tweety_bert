#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, cwd=cwd)

def main(args):
    project_root = Path(__file__).resolve().parent

    wav_folder = Path(args.wav_folder).resolve()
    song_detection_json_path = Path(args.song_detection_json).resolve() if args.song_detection_json else None

    inference_script_path = project_root / "src" / "inference.py"

    inference_cmd = [
        sys.executable,
        str(inference_script_path),
        "--bird_name", args.bird_name,
        "--wav_dir", str(wav_folder),
        *(["--song_detection_json", str(song_detection_json_path)] if song_detection_json_path else []),
        "--apply_post_processing", str(args.apply_post_processing),
        "--window_size", str(args.window_size),
        "--visualize", str(args.visualize)
    ]

    run_command(inference_cmd, cwd=project_root)


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
    parser.add_argument("--window_size", type=int, default=50,
                        help="Window size for label smoothing if post-processing is enabled. Default: 200.")
    parser.add_argument("--visualize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Generate visualization plots during inference. Default: True.")

    args = parser.parse_args()
    args.apply_post_processing = str(args.apply_post_processing)
    args.visualize = str(args.visualize)
    main(args)
