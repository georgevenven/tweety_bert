import csv
import json

def process_files(source_files, json_file):
    source_data = {}

    # Read each source file
    for source_file in source_files:
        with open(source_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                audio_file = row['audio_file']
                label = row['label']
                onset = float(row['onset_s'])
                offset = float(row['offset_s'])
                if audio_file not in source_data:
                    source_data[audio_file] = {}
                if label not in source_data[audio_file]:
                    source_data[audio_file][label] = []
                source_data[audio_file][label].append((onset, offset))

    # Process the data to make it contiguous
    for audio_file in source_data:
        for label in source_data[audio_file]:
            source_data[audio_file][label].sort()
            contiguous = []
            for onset, offset in source_data[audio_file][label]:
                if not contiguous or onset > contiguous[-1][1]:
                    contiguous.append([onset, offset])
                else:
                    contiguous[-1][1] = max(contiguous[-1][1], offset)
            source_data[audio_file][label] = contiguous

    # Read and update the JSON file
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    for entry in json_data:
        filename = entry['filename']
        if filename in source_data:
            entry['syllable_labels'] = source_data[filename]
        else:
            entry['syllable_labels'] = {}

    # Write the updated JSON data back to the file
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)

# Usage
source_files = [
    '/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_annot.csv'
]
json_file = '/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json'

process_files(source_files=source_files, json_file=json_file)