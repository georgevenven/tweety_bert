import csv
import json
import os

def process_files(source_file, destination_file):
    # Read the source file
    source_data = {}
    with open(source_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            label, onset, offset, _, _, audio_file, _, _, _ = row
            audio_file = os.path.basename(audio_file)  # Extract filename only
            if audio_file not in source_data:
                source_data[audio_file] = {}
            if label not in source_data[audio_file]:
                source_data[audio_file][label] = []
            source_data[audio_file][label].append((float(onset), float(offset)))

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

    # Read the destination file and update it
    updated_rows = []
    with open(destination_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        header.append('phrase_label onset/offsets')
        updated_rows.append(header)
        for row in reader:
            audio_file = os.path.basename(row[0])
            if audio_file in source_data:
                json_data = json.dumps(source_data[audio_file])
                row.append(json_data)
            else:
                row.append('')
            updated_rows.append(row)

    # Write the updated data back to the destination file
    with open(destination_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(updated_rows)

# Usage
source_file = 'path/to/source.csv'
destination_file = 'path/to/destination.csv'
process_files(source_file, destination_file)