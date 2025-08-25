#!/usr/bin/env python3
"""
Full TweetyBERT Processing Pipeline Wrapper

This script orchestrates the complete TweetyBERT pipeline:
1. Song Detection (using detect_song.py or detect_song_all_birds.py)
2. Model Training (using pretrain.py)
3. Decoder Training and UMAP Generation (using decoding.py)

Modes:
- 'single': Process one folder (all songs recursively from that folder)
  - Useful for training models with multiple birds' data for robustness
  - Creates one song detection JSON for all songs in the folder
- 'batch': Process parent folder with multiple child folders (each child folder separately)
  - Useful for processing multiple birds individually
  - Creates separate song detection JSON for each child folder

Usage:
    # Process using YAML configuration file
    python full_tweetybert_processing.py --config config_files/full_pipeline_config.yaml
"""

# --- Configuration ---
DEFAULT_MODEL = "canary_fall_nerve_llb-.01" # Default model to use for song detection

import os
import sys
import argparse
import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime
import json
import shutil
import yaml

def setup_logging(log_file_path):
    """Set up logging to both file and console."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger('full_tweetybert_processing')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_command_with_logging(cmd, logger, description, cwd=None):
    """Run a command and log the results."""
    logger.info(f"Running {description}: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✓ {description} completed successfully in {duration:.2f} seconds")
        
        # Log stdout if any
        if result.stdout.strip():
            logger.info(f"STDOUT for {description}:\n{result.stdout}")
        
        # Log stderr if any (but filter out tqdm progress bars)
        if result.stderr.strip():
            stderr_lines = result.stderr.strip().split('\n')
            filtered_stderr = []
            for line in stderr_lines:
                # Skip tqdm progress bars
                if 'it/s]' not in line and '%' not in line:
                    filtered_stderr.append(line)
            
            if filtered_stderr:
                logger.info(f"STDERR for {description}:\n{chr(10).join(filtered_stderr)}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"✗ {description} failed after {duration:.2f} seconds")
        logger.error(f"Return code: {e.returncode}")
        
        if e.stdout.strip():
            logger.error(f"STDOUT for {description}:\n{e.stdout}")
        
        if e.stderr.strip():
            logger.error(f"STDERR for {description}:\n{e.stderr}")
        
        return False
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"✗ Unexpected error in {description} after {duration:.2f} seconds: {str(e)}")
        return False

def find_existing_song_detection_json(bird_folder, output_dir, logger):
    """Find existing song detection JSON file for the bird folder."""
    bird_name = Path(bird_folder).name
    
    # Look for existing song detection JSON files
    possible_files = [
        output_dir / f"{bird_name}_song_detection.json",
        output_dir / f"labeled_ZF_song_detection_fixed.json",  # The fixed one we created for curated data
        output_dir / f"labeled_ZF_song_detection.json",  # The original one we created for curated data
        output_dir / f"{bird_name}_song_detection_canary_fall_nerve_llb-.01.json"
    ]
    
    for json_file in possible_files:
        if json_file.exists():
            logger.info(f"Found existing song detection JSON: {json_file}")
            return json_file
    
    logger.error(f"No existing song detection JSON found for {bird_name}")
    logger.error(f"Looked for: {[str(f) for f in possible_files]}")
    return None

def run_song_detection_single(bird_folder, output_dir, logger, plot_spec=False, overwrite_song=False, model=DEFAULT_MODEL, skip_detection=False):
    """Run song detection for a single folder (processes all songs recursively)."""
    bird_name = Path(bird_folder).name
    
    # If skipping song detection, find existing JSON file
    if skip_detection:
        logger.info(f"Skipping song detection for folder {bird_name} - using existing JSON file")
        existing_json = find_existing_song_detection_json(bird_folder, output_dir, logger)
        if existing_json:
            return True, existing_json
        else:
            logger.error("No existing song detection JSON found. Cannot skip detection.")
            return False, None
    
    # Create unique filename based on model to avoid conflicts
    model_suffix = f"_{model}" if model != DEFAULT_MODEL else ""
    output_json = output_dir / f"{bird_name}_song_detection{model_suffix}.json"
    
    # Check if output file already exists
    if output_json.exists() and not overwrite_song:
        logger.info(f"Skipping song detection for folder {bird_name} - JSON already exists: {output_json}")
        return True, output_json
    
    cmd = [
        sys.executable,
        "detect_song.py",
        "--input_dir", str(bird_folder),
        "--output_json", str(output_json),
        "--model", model
    ]
    
    if plot_spec:
        cmd.append("--plot_spec")
    
    success = run_command_with_logging(cmd, logger, f"song detection for folder {bird_name}")
    return success, output_json if success else None

def run_song_detection_batch(parent_dir, output_dir, logger, plot_spec=False, overwrite_song=False, model=DEFAULT_MODEL):
    """Run song detection for all child folders in a parent directory."""
    cmd = [
        sys.executable,
        "detect_song_all_birds.py",
        "--parent_dir", str(parent_dir),
        "--output_dir", str(output_dir),
        "--model", model
    ]
    
    if plot_spec:
        cmd.append("--plot_spec")
    
    if overwrite_song:
        cmd.append("--overwrite_song")
    
    success = run_command_with_logging(cmd, logger, "batch song detection (processes each child folder separately)")
    return success

def run_pretraining(bird_folder, song_detection_json, experiment_name, logger, overwrite_pretraining=False, **kwargs):
    """Run model pretraining."""
    # Check if model already exists and training completed successfully
    model_dir = Path("experiments") / experiment_name
    model_weights = model_dir / "model_weights.pth"
    training_loss_plot = model_dir / "training_loss.png"
    
    # Check if model directory exists AND has the required files
    if model_dir.exists() and model_weights.exists() and training_loss_plot.exists() and not overwrite_pretraining:
        logger.info(f"Skipping pretraining for {experiment_name} - model already exists and training completed: {model_dir}")
        logger.info(f"  - Model weights: {model_weights}")
        logger.info(f"  - Training loss plot: {training_loss_plot}")
        return True
    elif model_dir.exists() and (not model_weights.exists() or not training_loss_plot.exists()):
        logger.warning(f"Model directory exists but training appears incomplete for {experiment_name}")
        logger.warning(f"  - Model weights exist: {model_weights.exists()}")
        logger.warning(f"  - Training loss plot exists: {training_loss_plot.exists()}")
        logger.warning(f"Will re-run pretraining to ensure completion")
    
    cmd = [
        sys.executable,
        "pretrain.py",
        "--input_dir", str(bird_folder),
        "--song_detection_json_path", str(song_detection_json),
        "--experiment_name", experiment_name
    ]
    
    # Add optional parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    success = run_command_with_logging(cmd, logger, f"pretraining for experiment {experiment_name}")
    return success

def run_decoding(bird_folder, song_detection_json, experiment_name, bird_name, logger, overwrite_decoding=False, **kwargs):
    """Run decoding and UMAP generation."""
    # Check if decoding outputs already exist
    # UMAP file: decoding.py creates files/{bird_name}.npz
    umap_file = Path("files") / f"{bird_name}.npz"
    
    # Decoder files: decoding.py creates experiments/{bird_name}_linear_decoder/decoder_state/
    decoder_dir = Path("experiments") / f"{bird_name}_linear_decoder" / "decoder_state"
    decoder_weights_file = decoder_dir / "decoder_weights.pth"
    decoder_config_file = decoder_dir / "decoder_config.json"
    
    if (umap_file.exists() or decoder_weights_file.exists()) and not overwrite_decoding:
        logger.info(f"Skipping decoding for {bird_name} - outputs already exist:")
        if umap_file.exists():
            logger.info(f"  - UMAP: {umap_file}")
        if decoder_weights_file.exists():
            logger.info(f"  - Decoder weights: {decoder_weights_file}")
        if decoder_config_file.exists():
            logger.info(f"  - Decoder config: {decoder_config_file}")
        return True
    
    cmd = [
        sys.executable,
        "decoding.py",
        "--mode", "single",
        "--bird_name", bird_name,
        "--model_name", experiment_name,
        "--wav_folder", str(bird_folder),
        "--song_detection_json_path", str(song_detection_json)
    ]
    
    # Add optional parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    success = run_command_with_logging(cmd, logger, f"decoding for {bird_name}")
    return success

def load_config(config_file):
    """Load configuration from YAML file."""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ValueError(f"Configuration file does not exist: {config_file}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def find_bird_folders(parent_dir):
    """Find all child folders in the parent directory that look like bird folders."""
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    if not parent_path.is_dir():
        raise ValueError(f"Parent path is not a directory: {parent_dir}")
    
    # Look for folders that might be bird folders
    bird_folders = []
    
    for item in parent_path.iterdir():
        if item.is_dir():
            folder_name = item.name
            # Check if it looks like a bird folder (contains letters and numbers)
            if any(c.isalpha() for c in folder_name) and any(c.isdigit() for c in folder_name):
                bird_folders.append(item)
    
    return sorted(bird_folders)

def create_unique_experiment_name(base_name, model=DEFAULT_MODEL):
    """Create a unique experiment name that includes the model name to avoid conflicts."""
    model_suffix = f"_{model}" if model != DEFAULT_MODEL else ""
    return f"{base_name}{model_suffix}"

def main():
    """Main function to orchestrate the full TweetyBERT pipeline.
    
    Modes:
    - 'single': Process one folder (all songs recursively from that folder)
    - 'batch': Process parent folder with multiple child folders (each child folder separately)
    """
    parser = argparse.ArgumentParser(
        description="Run the complete TweetyBERT pipeline from song detection to decoding"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Validate required configuration
    required_fields = ["mode", "experiment_name"]
    for field in required_fields:
        if field not in config:
            print(f"Error: Missing required field '{field}' in configuration file")
            sys.exit(1)
    
    # Validate mode-specific fields
    if config["mode"] == "single":
        if "bird_folder" not in config:
            print("Error: 'bird_folder' is required for single mode (processes all songs recursively from one folder)")
            sys.exit(1)
        if not Path(config["bird_folder"]).exists():
            print(f"Error: Input folder does not exist: {config['bird_folder']}")
            sys.exit(1)
    elif config["mode"] == "batch":
        if "parent_dir" not in config:
            print("Error: 'parent_dir' is required for batch mode (processes each child folder separately)")
            sys.exit(1)
        if not Path(config["parent_dir"]).exists():
            print(f"Error: Parent directory does not exist: {config['parent_dir']}")
            sys.exit(1)
    else:
        print("Error: 'mode' must be 'single' (one folder, all songs) or 'batch' (parent folder, separate child folders)")
        sys.exit(1)
    
    # Set up logging
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logging/{script_name}_{timestamp}.log"
    
    # Create logging directory if it doesn't exist
    Path("logging").mkdir(exist_ok=True)
    
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info(f"Starting {script_name}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Mode: {config['mode']}")
    if config['mode'] == "single":
        logger.info(f"Input folder: {config['bird_folder']} (will process all songs recursively)")
    else:
        logger.info(f"Parent directory: {config['parent_dir']} (will process each child folder separately)")
    logger.info(f"Experiment name: {config['experiment_name']}")
    logger.info(f"Output directory: {config.get('output_dir', 'files')}")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        output_dir = Path(config.get('output_dir', 'files'))
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        if config['mode'] == "single":
            # Process single folder (all songs recursively)
            bird_folder = Path(config['bird_folder'])
            bird_name = bird_folder.name
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing single folder: {bird_name} (all songs recursively)")
            logger.info(f"{'='*60}")
            
            # Step 1: Song Detection
            logger.info(f"\n--- Step 1: Song Detection ---")
            song_detection_config = config.get('song_detection', {})
            overwrite_options = config.get('overwrite_options', {})
            skip_detection = song_detection_config.get('skip_song_detection', False)
            
            if skip_detection:
                logger.info("Song detection is SKIPPED - using existing JSON file")
            
            song_detection_success, song_detection_json = run_song_detection_single(
                bird_folder, output_dir, logger, 
                song_detection_config.get('plot_spec', False),
                overwrite_options.get('overwrite_song_detection', False),
                song_detection_config.get('model', DEFAULT_MODEL),
                skip_detection
            )
            
            if not song_detection_success:
                logger.error("Song detection failed. Stopping pipeline.")
                sys.exit(1)
            
            # Step 2: Pretraining
            logger.info(f"\n--- Step 2: Model Pretraining ---")
            pretraining_config = config.get('pretraining', {})
            
            # Create unique experiment name based on model
            song_detection_model = song_detection_config.get('model', DEFAULT_MODEL)
            unique_experiment_name = create_unique_experiment_name(config['experiment_name'], song_detection_model)
            
            pretraining_success = run_pretraining(
                bird_folder, song_detection_json, unique_experiment_name, logger,
                overwrite_pretraining=overwrite_options.get('overwrite_pretraining', False),
                test_percentage=pretraining_config.get('test_percentage', 20.0),
                batch_size=pretraining_config.get('batch_size', 42),
                learning_rate=pretraining_config.get('learning_rate', 3e-4),
                context=pretraining_config.get('context', 1000),
                m=pretraining_config.get('m', 250)
            )
            
            if not pretraining_success:
                logger.error("Pretraining failed. Stopping pipeline.")
                sys.exit(1)
            
            # Step 3: Decoding
            logger.info(f"\n--- Step 3: Decoding and UMAP Generation ---")
            decoding_config = config.get('decoding', {})
            decoding_success = run_decoding(
                bird_folder, song_detection_json, unique_experiment_name, bird_name, logger,
                overwrite_decoding=overwrite_options.get('overwrite_decoding', False),
                num_random_files_spec=decoding_config.get('num_random_files_spec', 100),
                num_samples_umap=decoding_config.get('num_samples_umap', '5e5')
            )
            
            if not decoding_success:
                logger.error("Decoding failed.")
                sys.exit(1)
            
            logger.info(f"\n✓ Successfully completed full pipeline for folder: {bird_name}!")
            
        else:
            # Process all child folders in batch mode
            logger.info(f"\n--- Step 1: Batch Song Detection ---")
            song_detection_config = config.get('song_detection', {})
            overwrite_options = config.get('overwrite_options', {})
            song_detection_success = run_song_detection_batch(
                config['parent_dir'], output_dir, logger,
                song_detection_config.get('plot_spec', False),
                overwrite_options.get('overwrite_song_detection', False),
                song_detection_config.get('model', DEFAULT_MODEL)
            )
            
            if not song_detection_success:
                logger.error("Batch song detection failed. Stopping pipeline.")
                sys.exit(1)
            
            # Find all child folders
            bird_folders = find_bird_folders(config['parent_dir'])
            logger.info(f"Found {len(bird_folders)} child folders for processing")
            
            successful_birds = []
            failed_birds = []
            
            for i, bird_folder in enumerate(bird_folders, 1):
                bird_name = bird_folder.name
                
                # Create model-specific JSON filename
                song_detection_model = song_detection_config.get('model', DEFAULT_MODEL)
                model_suffix = f"_{song_detection_model}" if song_detection_model != DEFAULT_MODEL else ""
                song_detection_json = output_dir / f"{bird_name}_song_detection{model_suffix}.json"
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing child folder {i}/{len(bird_folders)}: {bird_name}")
                logger.info(f"{'='*60}")
                
                # Check if song detection JSON exists
                if not song_detection_json.exists():
                    logger.error(f"Song detection JSON not found for child folder {bird_name}: {song_detection_json}")
                    failed_birds.append(bird_name)
                    continue
                
                # Step 2: Pretraining
                logger.info(f"\n--- Step 2: Model Pretraining for {bird_name} ---")
                pretraining_config = config.get('pretraining', {})
                
                # Create unique experiment name based on model
                unique_experiment_name = create_unique_experiment_name(f"{config['experiment_name']}_{bird_name}", song_detection_model)
                
                pretraining_success = run_pretraining(
                    bird_folder, song_detection_json, unique_experiment_name, logger,
                    overwrite_pretraining=overwrite_options.get('overwrite_pretraining', False),
                    test_percentage=pretraining_config.get('test_percentage', 20.0),
                    batch_size=pretraining_config.get('batch_size', 42),
                    learning_rate=pretraining_config.get('learning_rate', 3e-4),
                    context=pretraining_config.get('context', 1000),
                    m=pretraining_config.get('m', 250)
                )
                
                if not pretraining_success:
                    logger.error(f"✗ Pretraining failed for {bird_name}")
                    failed_birds.append(bird_name)
                    continue
                
                # Step 3: Decoding
                logger.info(f"\n--- Step 3: Decoding and UMAP Generation for {bird_name} ---")
                decoding_config = config.get('decoding', {})
                decoding_success = run_decoding(
                    bird_folder, song_detection_json, unique_experiment_name, bird_name, logger,
                    overwrite_decoding=overwrite_options.get('overwrite_decoding', False),
                    num_random_files_spec=decoding_config.get('num_random_files_spec', 100),
                    num_samples_umap=decoding_config.get('num_samples_umap', '5e5')
                )
                
                if decoding_success:
                    successful_birds.append(bird_name)
                    logger.info(f"✓ Successfully completed full pipeline for {bird_name}")
                else:
                    failed_birds.append(bird_name)
                    logger.error(f"✗ Decoding failed for {bird_name}")
            
            # Summary
            logger.info(f"\n{'='*80}")
            logger.info("PIPELINE SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"Total birds found: {len(bird_folders)}")
            logger.info(f"Successfully completed: {len(successful_birds)}")
            logger.info(f"Failed to complete: {len(failed_birds)}")
            
            if successful_birds:
                logger.info(f"\n✓ Successfully completed full pipeline for:")
                for bird_name in successful_birds:
                    logger.info(f"  - {bird_name}")
            
            if failed_birds:
                logger.error(f"\n✗ Failed to complete pipeline for:")
                for bird_name in failed_birds:
                    logger.error(f"  - {bird_name}")
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config_file": args.config,
            "mode": config['mode'],
            "experiment_name": config['experiment_name'],
            "output_directory": str(output_dir),
            "configuration": config
        }
        
        if config['mode'] == "single":
            summary["bird_folder"] = config['bird_folder']
            summary["bird_name"] = Path(config['bird_folder']).name
        else:
            summary["parent_directory"] = config['parent_dir']
            summary["total_birds_found"] = len(bird_folders)
            summary["successful_birds"] = len(successful_birds)
            summary["failed_birds"] = len(failed_birds)
            summary["successful_birds_list"] = successful_birds
            summary["failed_birds_list"] = failed_birds
        
        summary_file = output_dir / f"{script_name}_{config['experiment_name']}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSummary saved to: {summary_file}")
        logger.info(f"Log file: {log_file}")
        
        if config['mode'] == "batch" and failed_birds:
            logger.warning(f"\nSome birds failed to process. Check the log file for details.")
            sys.exit(1)
        else:
            logger.info(f"\nPipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Check the log file for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 