#!/usr/bin/env python3
"""
Simple Wrapper Script: Decoder Creation + Inference for Birds

This script runs the existing decoding.py and run_inference.py scripts for birds
using the TweetyBERT-AreaX model. It can operate in batch mode for multiple birds or single mode.
It creates decoders first, then runs inference with comprehensive logging.

Usage:
    python decoding_and_inference_batch.py --config config_files/inference_wrapper_config.yaml
"""

import yaml

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime

def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required keys
        required_keys = [
            'song_detection_dir',
            'skip_decoder_creation', 'decoding_inputs', 'run_inference_inputs', 'batch_mode'
        ]
        
        for key in required_keys:
            if key not in config:
                print(f"Error: Missing required key '{key}' in configuration file")
                sys.exit(1)
        
        if config['batch_mode']:
            if 'parent_dir' not in config:
                print("Error: 'parent_dir' is required when batch_mode is true")
                sys.exit(1)
        else:
            if 'bird_folder' not in config:
                print("Error: 'bird_folder' is required when batch_mode is false")
                sys.exit(1)
        
        # Validate decoding_inputs
        required_decoder_keys = [
            'mode', 'model_name', 'num_random_files_spec'
        ]
        
        for key in required_decoder_keys:
            if key not in config['decoding_inputs']:
                print(f"Error: Missing required decoder key '{key}' in configuration file")
                sys.exit(1)
        
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)

def setup_logging(log_file_path):
    """Set up logging to both file and console."""
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('inference_wrapper')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_decoding_for_bird(bird_folder, config, logger, overwrite=False):
    """Run decoding.py to create a decoder for a bird."""
    bird_name = bird_folder.name
    
    logger.info(f"Creating decoder for bird: {bird_name}")
    
    # Check if decoder already exists
    experiment_dir = Path("experiments") / f"{config['decoding_inputs']['model_name']}_{bird_name}_linear_decoder"
    if experiment_dir.exists() and (experiment_dir / "decoder_state").exists():
        if overwrite:
            logger.info(f"Decoder already exists for {bird_name}, but overwriting as requested")
            # Remove existing decoder directory
            import shutil
            shutil.rmtree(experiment_dir)
            logger.info(f"Removed existing decoder for {bird_name}")
        else:
            logger.info(f"Decoder already exists for {bird_name}, skipping creation")
            return str(experiment_dir)
    
    # Run decoding.py
    cmd = [
        sys.executable,
        "decoding.py",
        "--mode", config['decoding_inputs']['mode'],
        "--bird_name", f"{config['decoding_inputs']['model_name']}_{bird_name}",
        "--model_name", config['decoding_inputs']['model_name'],
        "--wav_folder", str(bird_folder),
        "--song_detection_json_path", str(Path(config['song_detection_dir']) / f"{bird_name}_song_detection.json"),
        "--num_random_files_spec", str(config['decoding_inputs']['num_random_files_spec']),
        "--num_samples_umap", str(config['decoding_inputs']['num_samples_umap'])
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        logger.info(f"Starting decoder creation for {bird_name}...")
        # Run command and capture all output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        logger.info(f"Command completed with return code: {result.returncode}")
        
        # Log all stdout and stderr
        if result.stdout:
            logger.info(f"[{bird_name} STDOUT] Output:")
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.info(f"[{bird_name} STDOUT] {line.strip()}")
        
        if result.stderr:
            logger.info(f"[{bird_name} STDERR] Errors/Warnings:")
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.info(f"[{bird_name} STDERR] {line.strip()}")
        
        if result.returncode == 0:
            logger.info(f"✓ Decoder created successfully for {bird_name}")
            return str(experiment_dir)
        else:
            logger.error(f"✗ Failed to create decoder for {bird_name}: Process returned {result.returncode}")
            return None
            
    except Exception as e:
        logger.error(f"✗ Exception occurred while creating decoder for {bird_name}: {e}")
        return None

def run_inference_for_bird(bird_folder, decoder_dir, logger, song_detection_json=None, 
                          apply_post_processing=True, visualize=False):
    """Run inference for a bird using the created decoder."""
    bird_name = bird_folder.name
    
    logger.info(f"Running inference for bird: {bird_name}")
    
    # Build command for run_inference.py
    cmd = [
        sys.executable,
        "run_inference.py",
        "--bird_name", decoder_dir,
        "--wav_folder", str(bird_folder),
        "--song_detection_json", str(song_detection_json),
        "--apply_post_processing", str(apply_post_processing),
        "--visualize", str(visualize)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        logger.info(f"Starting inference for {bird_name}...")
        # Run command and capture all output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        logger.info(f"Command completed with return code: {result.returncode}")
        
        # Log all stdout and stderr
        if result.stdout:
            logger.info(f"[{bird_name} STDOUT] Output:")
            for line in result.stdout.splitlines():
                if line.strip():
                    logger.info(f"[{bird_name} STDOUT] {line.strip()}")
        
        if result.stderr:
            logger.info(f"[{bird_name} STDERR] Errors/Warnings:")
            for line in result.stderr.splitlines():
                if line.strip():
                    logger.info(f"[{bird_name} STDERR] {line.strip()}")
        
        if result.returncode == 0:
            logger.info(f"✓ Inference completed successfully for {bird_name}")
            return True
        else:
            logger.error(f"✗ Inference failed for {bird_name}: Process returned {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Exception occurred during inference for {bird_name}: {e}")
        return False

def find_bird_folders(parent_dir):
    """Find all bird folders in the parent directory."""
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    if not parent_path.is_dir():
        raise ValueError(f"Parent path is not a directory: {parent_dir}")
    
    bird_folders = []
    skipped_no_wav = []
    
    for item in parent_path.iterdir():
        if item.is_dir():
            folder_name = item.name
            if any(c.isalpha() for c in folder_name) and any(c.isdigit() for c in folder_name):
                # Check if folder contains WAV files
                wav_files = list(item.rglob("*.wav"))
                if wav_files:
                    bird_folders.append(item)
                else:
                    skipped_no_wav.append(folder_name)
                    print(f"Warning: Skipping {folder_name} - no WAV files found (this bird will be skipped during processing)")
    
    return sorted(bird_folders), skipped_no_wav

def find_song_detection_json(bird_name, files_dir="files"):
    """Find the song detection JSON file for a given bird."""
    files_path = Path(files_dir)
    
    if not files_path.exists():
        return None
    
    possible_files = []
    for file in files_path.glob(f"{bird_name}_song_detection*.json"):
        possible_files.append(file)
    
    if not possible_files:
        return None
    
    return sorted(possible_files)[0]

def main():
    """Main function to create decoders and run inference."""
    parser = argparse.ArgumentParser(
        description="Create decoders and run inference for birds using TweetyBERT-AreaX"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Force recreation of existing decoders (overwrites existing ones)"
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logging/inference_wrapper_{timestamp}.log"
    Path("logging").mkdir(exist_ok=True)
    
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("Starting TweetyBERT-AreaX Decoder Creation + Inference")
    if config['batch_mode']:
        logger.info(f"Mode: Batch")
        logger.info(f"Parent directory: {config['parent_dir']}")
    else:
        logger.info(f"Mode: Single")
        logger.info(f"Bird folder: {config['bird_folder']}")
    logger.info(f"Model: {config['decoding_inputs']['model_name']}")
    logger.info(f"Skip decoder creation: {config['skip_decoder_creation']}")
    logger.info(f"Overwrite existing decoders: {args.overwrite or config.get('overwrite_existing_decoders', False)}")
    logger.info("=" * 80)
    
    try:
        # Get bird folders based on mode
        if config['batch_mode']:
            logger.info("Searching for bird folders...")
            bird_folders, skipped_no_wav = find_bird_folders(config['parent_dir'])
            
            if not bird_folders:
                logger.error("No bird folders found in the parent directory")
                sys.exit(1)
            
            logger.info(f"Found {len(bird_folders)} bird folders:")
            for folder in bird_folders:
                logger.info(f"  - {folder.name}")
            if skipped_no_wav:
                logger.warning(f"Skipped {len(skipped_no_wav)} folders due to no WAV files:")
                for folder_name in skipped_no_wav:
                    logger.warning(f"  ⏭️  {folder_name}")
        else:
            bird_folder_path = Path(config['bird_folder'])
            
            if not bird_folder_path.exists():
                logger.error(f"Bird folder does not exist: {bird_folder_path}")
                sys.exit(1)
            
            if not bird_folder_path.is_dir():
                logger.error(f"Bird path is not a directory: {bird_folder_path}")
                sys.exit(1)
            
            wav_files = list(bird_folder_path.rglob("*.wav"))
            if not wav_files:
                logger.error(f"No WAV files found in {bird_folder_path}")
                logger.error(f"This bird will be skipped during processing")
                sys.exit(1)
            
            bird_folders = [bird_folder_path]
            logger.info(f"Processing single bird folder: {bird_folders[0].name} with {len(wav_files)} WAV files")
        
        # All birds will be processed
        birds_to_process = bird_folders
        
        # Initialize tracking variables
        successful_birds = []
        failed_birds = []
        skipped_decoders = []  # Birds where decoder creation was skipped
        skipped_inference = []  # Birds where inference was skipped
        skipped_no_wav_files = []  # Birds skipped due to no WAV files
        
        # Track birds with no WAV files for batch mode
        if config['batch_mode']:
            skipped_no_wav_files = skipped_no_wav
        else:
            # For single mode, check if the bird folder has WAV files
            wav_files = list(bird_folders[0].rglob("*.wav"))
            if not wav_files:
                skipped_no_wav_files = [bird_folders[0].name]
                logger.warning(f"No WAV files found in {bird_folders[0].name}, skipping processing")
                logger.info("=" * 80)
                logger.info("FINAL SUMMARY")
                logger.info("=" * 80)
                logger.info(f"Total birds: {len(bird_folders)}")
                logger.info(f"Skipped (no WAV files): {len(skipped_no_wav_files)}")
                logger.info(f"\nSkipped due to no WAV files:")
                for bird in skipped_no_wav_files:
                    logger.info(f"  ⏭️  {bird}")
                logger.info(f"\nScript completed - no processing needed")
                logger.info(f"Log file: {log_file}")
                return
        
        # Phase 1: Create decoders (if not skipped)
        decoder_dirs = {}
        
        if not config['skip_decoder_creation']:
            logger.info(f"\n{'='*80}")
            logger.info("PHASE 1: CREATING DECODERS")
            logger.info(f"{'='*80}")
            
            logger.info(f"Creating decoders for {len(birds_to_process)} birds")
            
            for i, bird_folder in enumerate(birds_to_process, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Creating decoder {i}/{len(birds_to_process)}: {bird_folder.name}")
                logger.info(f"{'='*60}")
                
                # Check if decoder .npz file already exists
                bird_name = bird_folder.name
                npz_file = Path("files") / f"{config['decoding_inputs']['model_name']}_{bird_name}.npz"
                if npz_file.exists() and not (args.overwrite or config.get('overwrite_existing_decoders', False)):
                    logger.info(f"Decoder .npz already exists for {bird_name}, skipping decoder creation")
                    skipped_decoders.append(bird_name)
                    # Add to decoder_dirs so inference can still run
                    experiment_name = f"{config['decoding_inputs']['model_name']}_{bird_name}_linear_decoder"
                    experiment_dir = Path("experiments") / experiment_name
                    if experiment_dir.exists() and (experiment_dir / "decoder_state").exists():
                        # Store just the base name without _linear_decoder suffix
                        base_name = f"{config['decoding_inputs']['model_name']}_{bird_name}"
                        decoder_dirs[bird_folder.name] = base_name
                        logger.info(f"Found existing decoder directory for {bird_name}: {experiment_dir}")
                    else:
                        logger.warning(f"Decoder .npz exists but decoder directory not found for {bird_name}")
                        continue
                else:
                    try:
                        # Use command line flag or YAML config, with command line taking precedence
                        overwrite_flag = args.overwrite or config.get('overwrite_existing_decoders', False)
                        decoder_dir = run_decoding_for_bird(bird_folder, config, logger, overwrite_flag)
                        
                        if decoder_dir:
                            # Extract just the experiment name from the full path and remove _linear_decoder suffix
                            experiment_name = Path(decoder_dir).name
                            if experiment_name.endswith('_linear_decoder'):
                                base_name = experiment_name[:-len('_linear_decoder')]
                            else:
                                base_name = experiment_name
                            decoder_dirs[bird_folder.name] = base_name
                            logger.info(f"✓ Successfully created decoder for {bird_folder.name}")
                        else:
                            logger.error(f"✗ Failed to create decoder for {bird_folder.name}")
                            logger.warning(f"Will skip inference for {bird_folder.name} due to decoder creation failure")
                            failed_birds.append(bird_folder.name)
                    except Exception as e:
                        logger.error(f"✗ Unexpected error creating decoder for {bird_folder.name}: {str(e)}")
                        logger.warning(f"Will skip inference for {bird_folder.name} due to decoder creation error")
                        failed_birds.append(bird_folder.name)
                        continue
        else:
            logger.info("Skipping decoder creation as requested")
            # Try to find existing decoders
            for bird_folder in birds_to_process:
                experiment_name = f"{config['decoding_inputs']['model_name']}_{bird_folder.name}_linear_decoder"
                experiment_dir = Path("experiments") / experiment_name
                if experiment_dir.exists() and (experiment_dir / "decoder_state").exists():
                    # Store just the base name without _linear_decoder suffix
                    base_name = f"{config['decoding_inputs']['model_name']}_{bird_folder.name}"
                    decoder_dirs[bird_folder.name] = base_name
                    logger.info(f"Found existing decoder for {bird_folder.name}: {experiment_dir}")
                else:
                    logger.error(f"No existing decoder found for {bird_folder.name}")
        
        
        # Phase 2: Run inference
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2: RUNNING INFERENCE")
        logger.info(f"{'='*80}")
        
        logger.info(f"Starting inference for {len(birds_to_process)} birds")
        logger.info(f"Available decoders: {list(decoder_dirs.keys())}")
        
        for i, bird_folder in enumerate(birds_to_process, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running inference {i}/{len(birds_to_process)}: {bird_folder.name}")
            logger.info(f"{'='*60}")
            
            if bird_folder.name not in decoder_dirs:
                logger.error(f"No decoder available for {bird_folder.name}, skipping inference")
                # Don't add to failed_birds here since this is a skip, not a failure
                # The bird was already added to failed_birds during decoder creation if it failed
                continue
            
            # Check if final output JSON already exists
            bird_name = bird_folder.name
            output_file = Path("files") / f"{config['decoding_inputs']['model_name']}_{bird_name}_decoded_database.json"
            if output_file.exists():
                logger.info(f"Final output JSON already exists for {bird_name}, skipping inference")
                skipped_inference.append(bird_name)
                successful_birds.append(bird_name)  # Mark as successfully processed
                continue
            
            try:
                # Try to find song detection JSON
                song_detection_json = find_song_detection_json(bird_folder.name, config['song_detection_dir'])
                
                if song_detection_json:
                    logger.info(f"Found song detection JSON: {song_detection_json}")
                else:
                    logger.warning(f"No song detection JSON found for {bird_folder.name}")
                    song_detection_json = None
                
                # Run inference
                success = run_inference_for_bird(
                    bird_folder, 
                    decoder_dirs[bird_folder.name], 
                    logger, 
                    song_detection_json,
                    config['run_inference_inputs']['apply_post_processing'],
                    config['run_inference_inputs']['visualize']
                )
                
                if success:
                    successful_birds.append(bird_folder.name)
                    logger.info(f"✓ Successfully processed {bird_folder.name}")
                else:
                    failed_birds.append(bird_folder.name)
                    logger.error(f"✗ Failed to process {bird_folder.name}")
                    
            except Exception as e:
                logger.error(f"✗ Unexpected error during inference for {bird_folder.name}: {str(e)}")
                failed_birds.append(bird_folder.name)
                continue
            
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info(f"Total birds: {len(bird_folders)}")
        logger.info(f"Skipped (no WAV files): {len(skipped_no_wav_files)}")
        logger.info(f"Decoders available: {len(decoder_dirs)}")
        
        # DECODING STEP SUMMARY
        logger.info(f"\n{'='*60}")
        logger.info("DECODING STEP RESULTS")
        logger.info(f"{'='*60}")
        
        # Calculate decoding statistics
        decoding_successes = [bird for bird in decoder_dirs.keys() if bird not in skipped_decoders]
        decoding_failures = failed_birds  # Birds that failed during decoder creation
        
        logger.info(f"Decoding successes: {len(decoding_successes)}")
        logger.info(f"Decoding skips (existing decoders): {len(skipped_decoders)}")
        logger.info(f"Decoding failures: {len(decoding_failures)}")
        
        if decoding_successes:
            logger.info(f"\nDecoding successful birds:")
            for bird_name in decoding_successes:
                logger.info(f"  ✓ {bird_name}")
        
        if skipped_decoders:
            logger.info(f"\nDecoding skipped birds (existing decoders):")
            for bird_name in skipped_decoders:
                logger.info(f"  ⏭️  {bird_name}")
        
        if decoding_failures:
            logger.error(f"\nDecoding failed birds:")
            for bird_name in decoding_failures:
                logger.error(f"  ✗ {bird_name}")
        
        # INFERENCE STEP SUMMARY
        logger.info(f"\n{'='*60}")
        logger.info("INFERENCE STEP RESULTS")
        logger.info(f"{'='*60}")
        
        # Calculate inference statistics
        inference_successes = successful_birds
        inference_skips = skipped_inference
        inference_failures = [bird for bird in failed_birds if bird in decoder_dirs.keys()]  # Birds that had decoders but failed inference
        
        logger.info(f"Inference successes: {len(inference_successes)}")
        logger.info(f"Inference skips (existing outputs): {len(inference_skips)}")
        logger.info(f"Inference failures: {len(inference_failures)}")
        
        if inference_successes:
            logger.info(f"\nInference successful birds:")
            for bird_name in inference_successes:
                logger.info(f"  ✓ {bird_name}")
        
        if inference_skips:
            logger.info(f"\nInference skipped birds (existing outputs):")
            for bird_name in inference_skips:
                logger.info(f"  ⏭️  {bird_name}")
        
        if inference_failures:
            logger.error(f"\nInference failed birds:")
            for bird_name in inference_failures:
                logger.error(f"  ✗ {bird_name}")
        
        # OVERALL SUMMARY
        logger.info(f"\n{'='*60}")
        logger.info("OVERALL SUMMARY")
        logger.info(f"{'='*60}")
        
        total_processed = len(successful_birds) + len(skipped_inference)
        total_failed = len(failed_birds)
        
        logger.info(f"Total birds with WAV files: {len(bird_folders) - len(skipped_no_wav_files)}")
        logger.info(f"Total successfully processed: {total_processed}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Total skipped (no WAV files): {len(skipped_no_wav_files)}")
        
        # Clarify the distinction between failed and skipped
        logger.info(f"\nNote: 'Failed' means an exception occurred during processing.")
        logger.info(f"'Skipped' means the bird was intentionally not processed (existing outputs, no WAV files, etc.)")
        
        if failed_birds:
            logger.warning(f"\nSome birds failed to process. Check the log file for details: {log_file}")
            logger.warning(f"Failed birds: {', '.join(failed_birds)}")
        else:
            logger.info(f"\nAll birds processed successfully!")
        
        logger.info(f"\nScript completed successfully!")
        logger.info(f"Log file: {log_file}")
        
        # Final status
        if successful_birds:
            logger.info(f"\n🎉 Successfully processed {len(successful_birds)} birds!")
        if failed_birds:
            logger.warning(f"\n⚠️  {len(failed_birds)} birds failed to process (exceptions occurred)")
        if skipped_decoders:
            logger.info(f"\n⏭️  {len(skipped_decoders)} decoder creations were skipped (existing decoders)")
        if skipped_inference:
            logger.info(f"\n⏭️  {len(skipped_inference)} inference runs were skipped (existing outputs)")
        if skipped_no_wav_files:
            logger.info(f"\n⏭️  {len(skipped_no_wav_files)} birds were skipped due to no WAV files")
        
        logger.info(f"\nScript completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error("Check the log file for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 