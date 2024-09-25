import json
import os
from glob import glob

def merge_json_files(input_files, output_file):
    merged_data = []

    for file in input_files:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    # Specify the directory containing the JSON files
    json_directory = 'path/to/json/files'
    
    # Get a list of all JSON files in the directory
    json_files = glob(os.path.join(json_directory, '*.json'))
    
    # Specify the output file
    output_file = 'merged_output.json'
    
    # Merge the JSON files
    merge_json_files(json_files, output_file)
    
    print(f"Merged {len(json_files)} files into {output_file}")
