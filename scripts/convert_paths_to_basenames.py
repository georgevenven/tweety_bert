#!/usr/bin/env python3
"""
This script reads a text file containing a list of file paths (one per line), extracts only the basename 
(i.e., the filename itself, without any preceding directories) from each path, and writes these basenames 
to a new output file. The purpose of this script is to help normalize file references in downstream processes 
that require matching or processing just the filename rather than dealing with full directory paths.

Context:
--------
In certain workflows, external tools or data files may provide lists of files with full absolute paths, 
while other components of the pipeline expect only the filename (basename). For instance, a JSON file might 
store only basenames to identify audio samples, while a test set file or some other external resource 
provides full paths. This discrepancy can lead to data integration issues where the script or program 
responsible for filtering or grouping files cannot match entries because they differ in how the filename 
is specified.

By running this script, you can easily transform a file containing fully qualified paths into a file 
containing only the basenames. This step ensures that all components of the pipeline reference files 
consistently by their basenames, simplifying lookups and comparisons.

Example:
--------
Suppose you have a file `input_paths.txt` like this:

    /home/user/data/audio/llb11_03511_2018_05_11_09_04_14.wav
    /media/storage/llb16_songs/llb16_0310_2018_05_05_18_00_47.wav
    /absolute/path/to/some_other_file.wav

After running:
    python convert_paths_to_basenames.py input_paths.txt output_basenames.txt

The `output_basenames.txt` file would contain:
    llb11_03511_2018_05_11_09_04_14.wav
    llb16_0310_2018_05_05_18_00_47.wav
    some_other_file.wav

This output can then be used by scripts that expect filenames without directory paths.

Usage:
------
    python convert_paths_to_basenames.py <input_file> <output_file>

Arguments:
----------
- input_file: Path to the input text file that contains full file paths.
- output_file: Path where the resulting file with basenames should be written.

If either argument is missing, the script will exit with a usage message.
"""

import sys
from pathlib import Path

# Configure input and output files here
input_file = "experiments/LLB_Model_For_Paper/train_files.txt"    # Replace with your input file path
output_file = "experiments/LLB_Model_For_Paper/train_files.txt"    # Replace with your output file path

def convert_paths_to_basenames(input_file, output_file):
    # Read all lines from the input file
    with open(input_file, 'r') as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    # Extract basenames
    basenames = [Path(line).name for line in lines]

    # Write basenames to the output file
    with open(output_file, 'w') as f_out:
        for name in basenames:
            f_out.write(name + '\n')

# Direct execution without command line arguments
convert_paths_to_basenames(input_file, output_file)
