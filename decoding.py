#!/usr/bin/env python3
import os
import argparse
import subprocess
import shutil
import sys
import random
import glob
from pathlib import Path

def run_command(command, cwd=None):
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, cwd=cwd)

def collect_all_files(input_dir, extension=".wav"):
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(extension):
                file_paths.append(os.path.join(root, name))
    return file_paths

# --- Main Script Logic ---
def main(args):
    project_root = Path(__file__).resolve().parent

    wav_folder = Path(args.wav_folder).resolve()
    song_detection_json_path = None
    if args.song_detection_json_path:
        song_detection_json_path = Path(args.song_detection_json_path).resolve()

    temp_dir = project_root / "temp_decode"
    experiment_dir = project_root / "experiments" / args.model_name

    copy_script_path = project_root / "scripts" / "copy_files_from_wavdir_to_multiple_event_dirs.py"
    spec_gen_script_path = project_root / "src" / "spectogram_generator.py"
    umap_script_path = project_root / "figure_generation_scripts" / "dim_reduced_birdsong_plots.py"
    decoder_script_path = project_root / "src" / "decoder.py"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    data_dirs_umap = [] # List to hold paths to spectrogram directories for UMAP

    if args.mode == 'grouping':
        grouping_cmd = [
            sys.executable,
            str(copy_script_path),
            str(wav_folder),
            str(song_detection_json_path),
            str(temp_dir)
        ]
        run_command(grouping_cmd)

        group_wav_dirs = sorted(glob.glob(str(temp_dir / "group_*")))

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

            with open(group_file_list_path, 'w') as list_f:
                copied_wav_files = list(group_src_dir_path.glob('*.wav'))
                for temp_wav_file in copied_wav_files:
                    original_path = original_wav_paths.get(temp_wav_file.name)
                    if original_path:
                        list_f.write(f"{original_path}\n")

            spec_group_cmd = [
                sys.executable,
                str(spec_gen_script_path),
                "--file_list", str(group_file_list_path),
                "--dst_dir", str(group_spec_dir),
            ] + spec_common_args
            run_command(spec_group_cmd)
            data_dirs_umap.append(str(group_spec_dir))

    elif args.mode == 'single':
        all_wav_files = collect_all_files(str(wav_folder))
        if len(all_wav_files) < args.num_random_files_spec:
            selected_files = all_wav_files
        else:
            selected_files = random.sample(all_wav_files, args.num_random_files_spec)

        single_file_list_path = temp_dir / "single_mode_files.txt"
        with open(single_file_list_path, 'w') as f:
            for file_path in selected_files:
                f.write(f"{file_path}\n")

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
            "--file_list", str(single_file_list_path),
            "--dst_dir", str(single_spec_dir),
        ] + spec_common_args
        run_command(spec_single_cmd)
        data_dirs_umap.append(str(single_spec_dir))

    umap_cmd = [
        sys.executable,
        str(umap_script_path),
        "--experiment_folder", str(experiment_dir),
        "--save_name", args.bird_name,
        "--samples", str(args.num_samples_umap),
        "--raw_spectogram", str(args.raw_spectrogram_umap).lower(),
        "--state_finding_algorithm", args.state_finding_algorithm_umap,
        "--context", str(args.context_umap),
        "--data_dirs"
    ] + data_dirs_umap

    run_command(umap_cmd)

    decoder_cmd = [
        sys.executable,
        str(decoder_script_path),
        "--experiment_name", args.model_name,
        "--bird_name", args.bird_name,
        "--context_length", str(args.context_umap)
    ]

    if args.freeze_backbone:
        decoder_cmd.append("--freeze_backbone")

    run_command(decoder_cmd)

    shutil.rmtree(temp_dir)


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
    parser.add_argument("--num_samples_umap", type=str, default="1e6",
                        help="Number of samples for UMAP (e.g., '1e6', '500000').")
    parser.add_argument("--raw_spectrogram_umap", action='store_true',
                        help="Use raw spectrograms for UMAP instead of model embeddings.")
    parser.add_argument("--state_finding_algorithm_umap", type=str, default="HDBSCAN",
                        help="Algorithm for state finding in UMAP (e.g., HDBSCAN).")
    parser.add_argument("--context_umap", type=int, default=1000,
                        help="Context size used for UMAP generation.")

    # Single Mode specific arguments
    parser.add_argument("--num_random_files_spec", type=int, default=5000,
                        help="Number of random WAV files to use for spectrogram generation in 'single' mode.")
    
    parser.add_argument("--nfft", type=int, default=1024, help="Number of FFT points for spectrogram generation.")

    parser.add_argument("--freeze_backbone", action="store_true", help="If set, freezes the backbone (linear probe mode). Default is fine-tuning.")

    # Grouping Mode specific arguments (Currently handled by copy script interaction)
    # parser.add_argument("--num_groups", type=int, default=4, help="Number of groups for temporal splitting in 'grouping' mode.")
    # parser.add_argument("--event_date", type=str, help="Event date (MM-DD) for alignment in 'grouping' mode.")

    args = parser.parse_args()
    main(args)