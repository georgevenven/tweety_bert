#!/usr/bin/env python3
"""
Song Detector All Birds Wrapper Script

This script runs song detection on all bird folders in a parent directory.
It processes each bird folder individually and generates a song detection JSON for each bird.

Usage:
    python detect_song_all_birds.py --parent_dir /path/to/bird/folders --output_dir /path/to/output
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
    logger = logging.getLogger('song_detector_all_birds')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_detect_song_for_bird(bird_folder, output_dir, logger, plot_spec=False, overwrite_song=False, model=DEFAULT_MODEL):
    """Run song detection for a single bird folder."""
    bird_name = bird_folder.name
    
    # Create unique filename based on model to avoid conflicts
    model_suffix = f"_{model}" if model != DEFAULT_MODEL else ""
    output_json_path = output_dir / f"{bird_name}_song_detection{model_suffix}.json"
    
    # Check if output file already exists
    if output_json_path.exists() and not overwrite_song:
        logger.info(f"Skipping {bird_name} - song detection JSON already exists: {output_json_path}")
        logger.info(f"Use --overwrite_song flag to force regeneration")
        return True, output_json_path
    
    logger.info(f"Processing bird: {bird_name}")
    logger.info(f"Input folder: {bird_folder}")
    logger.info(f"Output JSON: {output_json_path}")
    
    # Build the command to run detect_song.py
    cmd = [
        sys.executable,
        "detect_song.py",
        "--input_dir", str(bird_folder),
        "--output_json", str(output_json_path),
        "--model", model
    ]
    
    if plot_spec:
        cmd.append("--plot_spec")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command and capture output
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent  # Run from the same directory as this script
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Successfully processed {bird_name} in {duration:.2f} seconds")
        
        # Log stdout if any
        if result.stdout.strip():
            logger.info(f"STDOUT for {bird_name}:\n{result.stdout}")
        
        # Log stderr if any (but filter out tqdm progress bars)
        if result.stderr.strip():
            stderr_lines = result.stderr.strip().split('\n')
            filtered_stderr = []
            for line in stderr_lines:
                # Skip tqdm progress bars
                if 'it/s]' not in line and '%' not in line:
                    filtered_stderr.append(line)
            
            if filtered_stderr:
                logger.info(f"STDERR for {bird_name}:\n{chr(10).join(filtered_stderr)}")
        
        return True, output_json_path
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"Failed to process {bird_name} after {duration:.2f} seconds")
        logger.error(f"Return code: {e.returncode}")
        
        if e.stdout.strip():
            logger.error(f"STDOUT for {bird_name}:\n{e.stdout}")
        
        if e.stderr.strip():
            logger.error(f"STDERR for {bird_name}:\n{e.stderr}")
        
        return False, None
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"Unexpected error processing {bird_name} after {duration:.2f} seconds: {str(e)}")
        return False, None

def find_bird_folders(parent_dir):
    """Find all bird folders in the parent directory."""
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        raise ValueError(f"Parent directory does not exist: {parent_dir}")
    
    if not parent_path.is_dir():
        raise ValueError(f"Parent path is not a directory: {parent_dir}")
    
    # Look for folders that might be bird folders
    # Common patterns: USA####, LLB##, etc.
    bird_folders = []
    
    for item in parent_path.iterdir():
        if item.is_dir():
            folder_name = item.name
            # Check if it looks like a bird folder (contains letters and numbers)
            if any(c.isalpha() for c in folder_name) and any(c.isdigit() for c in folder_name):
                bird_folders.append(item)
    
    return sorted(bird_folders)

def main():
    """Main function to process all bird folders."""
    parser = argparse.ArgumentParser(
        description="Run song detection on all bird folders in a parent directory"
    )
    
    parser.add_argument(
        "--parent_dir", 
        type=str, 
        required=True,
        help="Path to the parent directory containing bird folders"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="files",
        help="Directory to save output JSON files (default: files/)"
    )
    
    parser.add_argument(
        "--plot_spec", 
        action="store_true",
        help="If set, plot spectrograms during detection"
    )
    
    parser.add_argument(
        "--overwrite_song",
        action="store_true",
        help="If set, overwrite existing song detection JSON files"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model to use for detection (e.g., 'bengalese_finch_test', 'canary_fall_nerve_llb-.01')"
    )
    
    args = parser.parse_args()
    
    # Set up logging with timestamp
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logging/{script_name}_{timestamp}.log"
    
    # Create logging directory if it doesn't exist
    Path("logging").mkdir(exist_ok=True)
    
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info(f"Starting {script_name}")
    logger.info(f"Parent directory: {args.parent_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Plot spectrograms: {args.plot_spec}")
    logger.info(f"Overwrite existing files: {args.overwrite_song}")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 80)
    
    try:
        # Find bird folders
        logger.info("Searching for bird folders...")
        bird_folders = find_bird_folders(args.parent_dir)
        
        if not bird_folders:
            logger.error("No bird folders found in the parent directory")
            logger.error("Expected folders with names containing letters and numbers (e.g., USA5288, LLB16)")
            sys.exit(1)
        
        logger.info(f"Found {len(bird_folders)} bird folders:")
        for folder in bird_folders:
            logger.info(f"  - {folder.name}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Process each bird folder
        successful_birds = []
        failed_birds = []
        
        for i, bird_folder in enumerate(bird_folders, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing bird {i}/{len(bird_folders)}: {bird_folder.name}")
            logger.info(f"{'='*60}")
            
            success, output_path = run_detect_song_for_bird(
                bird_folder, 
                output_dir, 
                logger, 
                args.plot_spec,
                args.overwrite_song,
                args.model
            )
            
            if success:
                successful_birds.append((bird_folder.name, output_path))
                logger.info(f"✓ Successfully processed {bird_folder.name}")
            else:
                failed_birds.append(bird_folder.name)
                logger.error(f"✗ Failed to process {bird_folder.name}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total birds found: {len(bird_folders)}")
        logger.info(f"Successfully processed: {len(successful_birds)}")
        logger.info(f"Failed to process: {len(failed_birds)}")
        
        if successful_birds:
            logger.info(f"\nSuccessfully processed birds:")
            for bird_name, output_path in successful_birds:
                logger.info(f"  ✓ {bird_name} -> {output_path}")
        
        if failed_birds:
            logger.error(f"\nFailed to process birds:")
            for bird_name in failed_birds:
                logger.error(f"  ✗ {bird_name}")
        
        # Create a summary JSON file
        summary = {
            "timestamp": datetime.now().isoformat(),
            "parent_directory": args.parent_dir,
            "output_directory": str(output_dir),
            "total_birds_found": len(bird_folders),
            "successful_birds": len(successful_birds),
            "failed_birds": len(failed_birds),
            "successful_birds_list": [bird_name for bird_name, _ in successful_birds],
            "failed_birds_list": failed_birds,
            "output_files": [str(output_path) for _, output_path in successful_birds]
        }
        
        summary_file = output_dir / f"{script_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nSummary saved to: {summary_file}")
        
        if failed_birds:
            logger.warning(f"\nSome birds failed to process. Check the log file for details: {log_file}")
            sys.exit(1)
        else:
            logger.info(f"\nAll birds processed successfully!")
            logger.info(f"Log file: {log_file}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Check the log file for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 