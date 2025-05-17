#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
import sys
import random
import glob
from pathlib import Path
import multiprocessing # To determine nproc for spectrogram generation

# --- Helper Function to Run Commands ---
def run_command(command, check=True, cwd=None):
    """Runs a command using subprocess and optionally checks for errors."""
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=check, capture_output=True, text=True, cwd=cwd)
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
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

# --- Function to collect all files (needed for single mode random selection) ---
def collect_all_files(input_dir, extension=".wav"):
    """Recursively collects all files with a specific extension."""
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(extension):
                file_paths.append(os.path.join(root, name))
    return file_paths

# --- Main Script Logic ---
def main(args):
    """Main function to execute the decoding pipeline."""

    # --- Get Absolute Paths & Validate ---
    project_root = Path(__file__).resolve().parent
    print(f"Project Root detected as: {project_root}")

    wav_folder = Path(args.wav_folder).resolve()
    song_detection_json_path = None
    if args.song_detection_json_path:
        song_detection_json_path = Path(args.song_detection_json_path).resolve()
        if not song_detection_json_path.is_file():
            print(f"Error: Song detection JSON '{args.song_detection_json_path}' not found.", file=sys.stderr)
            sys.exit(1)

    if not wav_folder.is_dir():
        print(f"Error: WAV folder '{args.wav_folder}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)

    # --- Define Paths ---
    temp_dir = project_root / "temp_decode"
    # Define experiment dir path using model name
    experiment_dir = project_root / "experiments" / args.model_name

    # Script paths
    copy_script_path = project_root / "scripts" / "copy_files_from_wavdir_to_multiple_event_dirs.py"
    spec_gen_script_path = project_root / "src" / "spectogram_generator.py"
    umap_script_path = project_root / "figure_generation_scripts" / "dim_reduced_birdsong_plots.py"
    decoder_script_path = project_root / "src" / "decoder.py"

    # --- Cleanup ---
    if temp_dir.exists():
        print(f"Removing existing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

    # --- Create Temp Directory ---
    print(f"Creating temporary directory: {temp_dir}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    data_dirs_umap = [] # List to hold paths to spectrogram directories for UMAP

    # --- Mode-Specific Preparation ---
    if args.mode == 'grouping':
        print("\n--- Mode: Grouping ---")
        if not song_detection_json_path:
            print(f"Error: --song_detection_json_path is required for grouping mode.", file=sys.stderr)
            sys.exit(1)
        # --- 1a. Determine file groupings ---
        # Run the grouping script. NOTE: This script *still copies* WAV files temporarily by default.
        # We will use these temporary directories ONLY to figure out which files belong to which group,
        # generate file lists pointing to ORIGINALS, and then delete the temporary WAV groups.
        # A better long-term solution would modify the grouping script itself to output lists directly.
        print("\n--- Step 1a: Determining file groups (using copy script) ---")
        grouping_cmd = [
            sys.executable,
            str(copy_script_path),
            str(wav_folder),
            str(song_detection_json_path),
            str(temp_dir) # Output base path for groups
            # Add other args for this script if needed (like event date, num groups etc.)
        ]
        run_command(grouping_cmd)

        # --- 1b. Generate file lists from temporary groups & Generate Spectrograms ---
        print("\n--- Step 1b: Generating file lists and Spectrograms for Groups ---")
        group_wav_dirs = sorted(glob.glob(str(temp_dir / "group_*")))
        if not group_wav_dirs:
             print(f"Error: No group directories found in {temp_dir} after running grouping script.", file=sys.stderr)
             sys.exit(1)
        print(f"Found {len(group_wav_dirs)} temporary group directories.")

        single_threaded_arg = "true" if args.single_threaded_spec else "false"
        spec_common_args = [
            "--song_detection_json_path", str(song_detection_json_path) if song_detection_json_path else "None",
            "--single_threaded", single_threaded_arg
            # Add --step_size, --nfft if needed
        ]

        # Dictionary to map original basenames to their full paths
        original_wav_paths = {p.name: str(p) for p in wav_folder.rglob('*.wav')}

        for i, group_src_dir_path_str in enumerate(group_wav_dirs):
            group_src_dir_path = Path(group_src_dir_path_str)
            group_file_list_path = temp_dir / f"group_{i+1}_files.txt"
            group_spec_dir = temp_dir / f"spec_files_group_{i+1}"
            group_spec_dir.mkdir(exist_ok=True)

            print(f"Generating file list for group {i+1} from {group_src_dir_path}...")
            with open(group_file_list_path, 'w') as list_f:
                copied_wav_files = list(group_src_dir_path.glob('*.wav'))
                print(f"  Found {len(copied_wav_files)} WAV files in temp group dir.")
                for temp_wav_file in copied_wav_files:
                    original_path = original_wav_paths.get(temp_wav_file.name)
                    if original_path:
                        list_f.write(f"{original_path}\n")
                    else:
                        print(f"  Warning: Could not find original path for {temp_wav_file.name}", file=sys.stderr)

            print(f"Generating spectrograms for group {i+1} using list {group_file_list_path} -> {group_spec_dir}")

            spec_group_cmd = [
                sys.executable,
                str(spec_gen_script_path),
                "--file_list", str(group_file_list_path), # Use the generated file list
                "--dst_dir", str(group_spec_dir),
            ] + spec_common_args
            run_command(spec_group_cmd)
            data_dirs_umap.append(str(group_spec_dir)) # Store path for UMAP

            # Optional: Delete the temporary WAV group directory immediately
            # print(f"  Removing temporary WAV directory: {group_src_dir_path}")
            # shutil.rmtree(group_src_dir_path)

    elif args.mode == 'single':
        print("\n--- Mode: Single Directory ---")
        # --- 1a. Select random files and create file list ---
        print("\n--- Step 1a: Selecting random files ---")
        all_wav_files = collect_all_files(str(wav_folder))
        if len(all_wav_files) < args.num_random_files_spec:
            print(f"Warning: Requested {args.num_random_files_spec} files, but only {len(all_wav_files)} found in {wav_folder}. Using all found files.")
            selected_files = all_wav_files
        else:
            selected_files = random.sample(all_wav_files, args.num_random_files_spec)

        single_file_list_path = temp_dir / "single_mode_files.txt"
        with open(single_file_list_path, 'w') as f:
            for file_path in selected_files:
                f.write(f"{file_path}\n")
        print(f"Created file list for single mode: {single_file_list_path}")

        # --- 1b. Generate Spectrograms using file list ---
        print("\n--- Step 1b: Generating Spectrograms for Single Mode ---")
        single_spec_dir = temp_dir / "umap_specs"
        single_spec_dir.mkdir(exist_ok=True)

        single_threaded_arg = "true" if args.single_threaded_spec else "false"
        spec_common_args = [
            "--song_detection_json_path", str(song_detection_json_path) if song_detection_json_path else "None",
            "--nfft", str(args.nfft),
            "--single_threaded", single_threaded_arg
        ]

        spec_single_cmd = [
            sys.executable,
            str(spec_gen_script_path),
            "--file_list", str(single_file_list_path), # Use file_list argument
            "--dst_dir", str(single_spec_dir),
        ] + spec_common_args
        run_command(spec_single_cmd)
        data_dirs_umap.append(str(single_spec_dir)) # Store path for UMAP

    else:
        print(f"Error: Invalid mode '{args.mode}'. Choose 'grouping' or 'single'.", file=sys.stderr)
        sys.exit(1)


    # --- 2. Run UMAP ---
    print("\n--- Step 2: Running UMAP ---")
    umap_cmd = [
        sys.executable,
        str(umap_script_path),
        "--experiment_folder", str(experiment_dir),
        "--save_name", args.bird_name,
        "--samples", str(args.num_samples_umap),
        "--raw_spectogram", str(args.raw_spectrogram_umap).lower(),
        "--state_finding_algorithm", args.state_finding_algorithm_umap,
        "--context", str(args.context_umap),
        "--data_dirs" # Note: Argparse with nargs='+' expects this first
    ] + data_dirs_umap # Then add all the directory paths

    run_command(umap_cmd)

    # --- 3. Train Decoder ---
    print("\n--- Step 3: Training Decoder ---")
    decoder_cmd = [
        sys.executable,
        str(decoder_script_path),
        "--experiment_name", args.model_name,
        "--bird_name", args.bird_name
    ]
    run_command(decoder_cmd)

    # --- 4. Final Cleanup ---
    print("\n--- Step 4: Cleaning up temporary directory ---")
    shutil.rmtree(temp_dir)
    print(f"Deleted temporary directory and its contents: {temp_dir}")
    print("\n--- Decoding Pipeline Completed Successfully! ---")


# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TweetyBERT Decoding Pipeline (UMAP + Decoder Training)")

    # Core arguments
    parser.add_argument("--mode", type=str, default="grouping", choices=['grouping', 'single'],
                        help="Mode of operation: 'grouping' (temporal groups) or 'single' (random subset).")
    parser.add_argument("--bird_name", type=str, required=True,
                        help="Identifier for the bird/dataset (used for saving UMAP/decoder).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the pretrained TweetyBERT experiment folder.")
    parser.add_argument("--wav_folder", type=str, required=True,
                        help="Path to the main directory containing WAV files.")
    parser.add_argument("--song_detection_json_path", type=str, required=False,
                        help="Path to the JSON file containing song detection data.")

    # Spectrogram Generation arguments
    parser.add_argument("--single_threaded_spec", action='store_true',
                        help="Force single-threaded spectrogram generation. Default uses multi-threading.")
    # Add step_size, nfft args if they need to be configurable, otherwise defaults in spectogram_generator.py are used

    # UMAP arguments
    parser.add_argument("--num_samples_umap", type=str, default="5e5",
                        help="Number of samples for UMAP (e.g., '1e6', '500000').")
    parser.add_argument("--raw_spectrogram_umap", action='store_true',
                        help="Use raw spectrograms for UMAP instead of model embeddings.")
    parser.add_argument("--state_finding_algorithm_umap", type=str, default="HDBSCAN",
                        help="Algorithm for state finding in UMAP (e.g., HDBSCAN).")
    parser.add_argument("--context_umap", type=int, default=1000,
                        help="Context size used for UMAP generation.")

    # Single Mode specific arguments
    parser.add_argument("--num_random_files_spec", type=int, default=1000,
                        help="Number of random WAV files to use for spectrogram generation in 'single' mode.")
    
    parser.add_argument("--nfft", type=int, default=1024, help="Number of FFT points for spectrogram generation.")

    # Grouping Mode specific arguments (Currently handled by copy script interaction)
    # parser.add_argument("--num_groups", type=int, default=4, help="Number of groups for temporal splitting in 'grouping' mode.")
    # parser.add_argument("--event_date", type=str, help="Event date (MM-DD) for alignment in 'grouping' mode.")

    args = parser.parse_args()

    # Validate arguments if needed (e.g., check samples format)
    try:
        # Try converting UMAP samples to float to validate format
        float(args.num_samples_umap)
    except ValueError:
        print(f"Error: Invalid format for --num_samples_umap '{args.num_samples_umap}'. Use scientific notation (e.g., 1e6) or integer.", file=sys.stderr)
        sys.exit(1)


    main(args)