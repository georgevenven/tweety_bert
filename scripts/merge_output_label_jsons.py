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

    # Remove duplicates at the end
    unique_data = []
    seen = set()
    for item in merged_data:
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            seen.add(item_str)
            unique_data.append(item)

    with open(output_file, 'w') as f:
        json.dump(unique_data, f, indent=4)

if __name__ == "__main__":
    # Specify the directory containing the JSON files
    json_directory = '/media/george-vengrovski/flash-drive/jsons'
    
    # Get a list of all JSON files in the directory
    json_files = glob(os.path.join(json_directory, '*.json'))
    
    # Specify the output file
    output_file = '/media/george-vengrovski/flash-drive/jsons/merged_output.json'
    
    # Merge the JSON files
    merge_json_files(json_files, output_file)
    
    print(f"Merged {len(json_files)} files into {output_file}")
