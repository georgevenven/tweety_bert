#!/usr/bin/env python3
"""
Auto-restart wrapper for umap_grid_search.py
Handles random CUDA crashes by automatically restarting the script until completion.
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path
from datetime import datetime

# Configuration
MAIN_SCRIPT = "umap_grid_search.py"
MAX_RESTARTS = 50  # Maximum number of restart attempts
RESTART_DELAY = 10  # Seconds to wait between restarts
LOG_FILE = "auto_restart_log.txt"

# CUDA-related error patterns that indicate we should restart
CUDA_ERROR_PATTERNS = [
    "CUDA out of memory",
    "CUDA error",
    "cupy.cuda.memory.OutOfMemoryError",
    "RuntimeError: CUDA",
    "CudaErrorMemoryAllocation",
    "bad_alloc",
    "out_of_memory",
    "device-side assert",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "cuml.common.exceptions.CumlException",
    "Segmentation fault",
    "core dumped"
]

# Success patterns that indicate completion
SUCCESS_PATTERNS = [
    "Adaptive search process complete",
    "All collected results saved to:",
    "Adaptive search complete"
]

def log_message(message):
    """Log message with timestamp to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry, flush=True)  # Force flush
    
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")
        f.flush()  # Force flush to file

def check_if_completed():
    """Check if the script has completed successfully by looking for output patterns"""
    if not Path(LOG_FILE).exists():
        return False
    
    try:
        with open(LOG_FILE, "r") as f:
            content = f.read()
            return any(pattern in content for pattern in SUCCESS_PATTERNS)
    except:
        return False

def is_cuda_related_crash(output):
    """Check if the crash is CUDA-related"""
    output_lower = output.lower()
    return any(pattern.lower() in output_lower for pattern in CUDA_ERROR_PATTERNS)

def run_main_script():
    """Run the main script and capture output"""
    log_message(f"Starting {MAIN_SCRIPT}...")
    
    try:
        # Force unbuffered output with PYTHONUNBUFFERED
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        # Run the script and capture both stdout and stderr
        process = subprocess.Popen(
            [sys.executable, "-u", MAIN_SCRIPT],  # -u for unbuffered
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=0,  # Unbuffered
            env=env
        )
        
        output_lines = []
        
        log_message("Subprocess started, waiting for output...")
        
        # Read output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Strip and print immediately
                line = output.rstrip()
                print(line, flush=True)  # Force immediate output
                output_lines.append(line)
                
                # Also log important lines to our log file
                if any(keyword in line.lower() for keyword in ['error', 'processing', 'rung', 'umap', 'completed']):
                    with open(LOG_FILE, "a") as f:
                        f.write(f"[SCRIPT] {line}\n")
                        f.flush()
        
        # Wait for process to complete
        return_code = process.wait()
        full_output = "\n".join(output_lines)
        
        log_message(f"Script finished with return code: {return_code}")
        log_message(f"Total output lines captured: {len(output_lines)}")
        
        return return_code, full_output
        
    except KeyboardInterrupt:
        log_message("Received keyboard interrupt. Stopping...")
        if 'process' in locals():
            process.terminate()
        raise
    except Exception as e:
        log_message(f"Error running script: {e}")
        return -1, str(e)

def main():
    """Main wrapper function"""
    log_message("=" * 60)
    log_message("Auto-restart wrapper for umap_grid_search.py started")
    log_message(f"Max restarts: {MAX_RESTARTS}")
    log_message(f"Restart delay: {RESTART_DELAY} seconds")
    log_message(f"Python executable: {sys.executable}")
    log_message(f"Working directory: {os.getcwd()}")
    log_message(f"Script exists: {Path(MAIN_SCRIPT).exists()}")
    log_message("=" * 60)
    
    restart_count = 0
    
    # Check if already completed
    if check_if_completed():
        log_message("Script appears to have completed successfully already. Exiting.")
        return 0
    
    while restart_count <= MAX_RESTARTS:
        try:
            # Run the main script
            return_code, output = run_main_script()
            
            # Check if completed successfully
            if return_code == 0:
                log_message("Script completed successfully!")
                
                # Double-check for success patterns in output
                if any(pattern in output for pattern in SUCCESS_PATTERNS):
                    log_message("Success patterns detected. Job complete!")
                    return 0
                else:
                    log_message("Script exited with code 0 but no success pattern found. Checking for completion...")
                    if check_if_completed():
                        log_message("Completion confirmed. Job complete!")
                        return 0
            
            # Script crashed or failed
            restart_count += 1
            
            if restart_count > MAX_RESTARTS:
                log_message(f"Maximum restart attempts ({MAX_RESTARTS}) reached. Giving up.")
                return 1
            
            # Analyze the crash
            if is_cuda_related_crash(output):
                log_message(f"CUDA-related crash detected (attempt {restart_count}/{MAX_RESTARTS})")
                log_message("This is expected - restarting with resume logic...")
            else:
                log_message(f"Non-CUDA crash detected (attempt {restart_count}/{MAX_RESTARTS})")
                log_message(f"Return code: {return_code}")
                if len(output) > 0:
                    log_message(f"Last few lines of output: {output[-500:]}")  # Last 500 chars
                log_message("Restarting anyway...")
            
            # Wait before restarting
            log_message(f"Waiting {RESTART_DELAY} seconds before restart...")
            time.sleep(RESTART_DELAY)
            
            log_message(f"Restarting... (attempt {restart_count + 1})")
            
        except KeyboardInterrupt:
            log_message("Keyboard interrupt received. Exiting gracefully.")
            return 130
        except Exception as e:
            log_message(f"Unexpected error in wrapper: {e}")
            restart_count += 1
            if restart_count <= MAX_RESTARTS:
                log_message(f"Restarting after unexpected error... (attempt {restart_count + 1})")
                time.sleep(RESTART_DELAY)
            else:
                log_message("Max restarts reached after unexpected errors. Exiting.")
                return 1
    
    log_message("Restart loop ended unexpectedly.")
    return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Fatal error in wrapper: {e}", flush=True)
        sys.exit(1) 