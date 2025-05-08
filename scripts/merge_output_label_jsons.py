#!/usr/bin/env python3
import json
import os
import argparse # Import argparse
import sys # Import sys for stderr

def merge_json_files(input_files, output_file):
    """
    Merges JSON data from a list of input files into a single output file.

    Args:
        input_files (list[str]): A list of paths to the JSON files to merge.
        output_file (str): Path to the file where the merged JSON data will be saved.
    """
    merged_data = []

    if not input_files:
        print(f"Warning: No input JSON files provided.", file=sys.stderr)
        # Write an empty list to the output file if no input files are given
        # Ensure the output directory exists before writing
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump([], f)
        print(f"Created empty output file: {output_file}")
        return

    print(f"Attempting to merge {len(input_files)} JSON files.")

    for file_path in input_files: # Iterate through the provided list of file paths
        if not os.path.exists(file_path):
            print(f"Warning: Input file not found: {file_path}. Skipping.", file=sys.stderr)
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Handle cases where the JSON file might contain a list or a single object
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {os.path.basename(file_path)}. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error processing {os.path.basename(file_path)}: {e}. Skipping.", file=sys.stderr)


    # --- Deduplication Logic (Optional but kept from original) ---
    # If you are sure the detector outputs unique files, you might remove this.
    unique_data = []
    seen = set()
    for item in merged_data:
        # Ensure item is serializable before adding to set
        try:
            item_str = json.dumps(item, sort_keys=True)
            if item_str not in seen:
                seen.add(item_str)
                unique_data.append(item)
        except TypeError:
             print(f"Warning: Skipping non-serializable item during deduplication: {item}", file=sys.stderr)
    # --- End Deduplication ---

    # Save the merged (and potentially deduplicated) data
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(unique_data, f, indent=4) # Use unique_data here
        print(f"Successfully merged {len(unique_data)} unique entries into '{output_file}'")
    except Exception as e:
        print(f"Error writing merged JSON to {output_file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    # --- Add Argument Parser ---
    parser = argparse.ArgumentParser(description="Merge JSON files from a list of paths.")
    # Change input argument to accept multiple file paths using nargs='+'
    # The output file path will be the last argument
    parser.add_argument("input_files", nargs='+', help="List of paths to the JSON files to merge, followed by the output file path.")
    # The last element in args.input_files will be treated as the output file
    args = parser.parse_args()

    # Separate the output file path from the input file paths
    output_file_path = args.input_files[-1]
    input_file_paths = args.input_files[:-1]

    # Validate that we have at least one input file and one output file
    if not input_file_paths:
         parser.error("No input files provided.")
    if not output_file_path:
         parser.error("Output file path is missing.")

    # --- End Argument Parser ---

    # Merge the JSON files using arguments
    merge_json_files(input_file_paths, output_file_path)

