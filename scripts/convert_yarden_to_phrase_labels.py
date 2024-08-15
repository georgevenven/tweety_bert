import csv
import json

def process_files(source_file, intermediate_file, final_output_file):
    # Read the source file
    source_data = {}
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

    # Read the intermediate file and update it
    updated_rows = []
    with open(intermediate_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['phrase_label onset/offsets']
        updated_rows.append(fieldnames)
        for row in reader:
            file_name = row['file_name']
            if file_name in source_data:
                # Convert to JSON, replace double quotes with single quotes
                json_data = json.dumps(source_data[file_name]).replace('"', "'")
                row['phrase_label onset/offsets'] = json_data
            else:
                row['phrase_label onset/offsets'] = ''
            updated_rows.append(row.values())

    # Write the updated data to the new final output file
    with open(final_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(updated_rows)

# Usage
source_file = 'files/llb3_annot.csv'
intermediate_file = 'files/llb3_whisperseg.csv'
final_output_file = 'files/llb3_final_output.csv'

process_files(source_file=source_file, intermediate_file=intermediate_file, final_output_file=final_output_file)
