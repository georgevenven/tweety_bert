import os
import xml.etree.ElementTree as ET
import json

def parse_and_collate_annotations(base_dir, num_bird_folders, sampling_rate):
    """
    Parses Annotation.xml files from BirdN directories and collates syllable data.
    Labels are stored as integers if possible, otherwise as strings.
    """
    all_syllable_data = {} # Stores {"filename.wav": {label_int: [(on_ms, off_ms), ...]}}

    for i in range(num_bird_folders):
        bird_folder_name = f"Bird{i}"
        bird_number_str = str(i)
        xml_file_path = os.path.join(base_dir, bird_folder_name, "Annotation.xml")

        if not os.path.exists(xml_file_path):
            print(f"Warning: Annotation.xml not found in {os.path.join(base_dir, bird_folder_name)}. Skipping.")
            continue

        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            for sequence_elem in root.findall("Sequence"):
                wave_file_name_xml = sequence_elem.findtext("WaveFileName")
                if not wave_file_name_xml:
                    print(f"Warning: Missing WaveFileName in a sequence in {xml_file_path}. Skipping sequence.")
                    continue
                
                # Construct the full target filename as it appears in the JSON
                # e.g., "0.wav" in Bird3/Annotation.xml -> "0_bird3.wav"
                base_name = wave_file_name_xml.replace(".wav", "")
                target_json_filename = f"{base_name}_bird{bird_number_str}.wav"

                try:
                    sequence_position_samples = int(sequence_elem.findtext("Position", "0"))
                except ValueError:
                    print(f"Warning: Invalid Sequence Position in {xml_file_path} for {wave_file_name_xml}. Assuming 0. Skipping sequence element.")
                    continue

                if target_json_filename not in all_syllable_data:
                    all_syllable_data[target_json_filename] = {}

                for note_elem in sequence_elem.findall("Note"):
                    label_str = None # Initialize for robust error message
                    try:
                        note_position_samples = int(note_elem.findtext("Position"))
                        note_length_samples = int(note_elem.findtext("Length"))
                        label_str = note_elem.findtext("Label")
                        
                        if label_str is None:
                            print(f"Warning: Missing Label in a note in {xml_file_path} for {target_json_filename}. Skipping note.")
                            continue
                        
                        try:
                            label_key = int(label_str)  # Try to convert to integer
                        except ValueError:
                            label_key = label_str      # If not an integer (e.g., 'a', 'b'), use the string itself
                        
                    except (ValueError, TypeError) as e:
                        current_label_for_error = label_str if label_str is not None else 'N/A'
                        print(f"Warning: Invalid note data (e.g., Position/Length) in {xml_file_path} for {target_json_filename} (Label: {current_label_for_error}). Error: {e}. Skipping note.")
                        continue

                    # Calculate absolute onset and offset in milliseconds
                    syllable_onset_abs_samples = sequence_position_samples + note_position_samples
                    syllable_onset_ms = (syllable_onset_abs_samples / sampling_rate) * 1000
                    syllable_offset_ms = ((syllable_onset_abs_samples + note_length_samples) / sampling_rate) * 1000

                    # Use label_key (which can be int or str)
                    if label_key not in all_syllable_data[target_json_filename]:
                        all_syllable_data[target_json_filename][label_key] = []
                    
                    all_syllable_data[target_json_filename][label_key].append((syllable_onset_ms, syllable_offset_ms))

        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file_path}: {e}. Skipping this file.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {xml_file_path}: {e}. Skipping this file.")
            continue
            
    return all_syllable_data

def make_intervals_contiguous(syllable_data):
    """
    Sorts and merges overlapping/adjacent time intervals for each label (label_key can be int or str).
    """
    for filename in syllable_data:
        # MODIFICATION: label_int -> label_key
        for label_key in syllable_data[filename]: 
            # MODIFICATION: syllable_data[filename][label_int] -> syllable_data[filename][label_key]
            intervals = sorted(syllable_data[filename][label_key]) 
            if not intervals:
                syllable_data[filename][label_key] = [] # Ensure it's an empty list if no intervals
                continue
            merged = []
            current_start, current_end = intervals[0]

            for i in range(1, len(intervals)):
                next_start, next_end = intervals[i]
                if next_start <= current_end: # Overlapping or adjacent
                    current_end = max(current_end, next_end)
                else:
                    merged.append([current_start, current_end])
                    current_start, current_end = next_start, next_end
            
            merged.append([current_start, current_end]) # Add the last interval
            # MODIFICATION: syllable_data[filename][label_int] -> syllable_data[filename][label_key]
            syllable_data[filename][label_key] = merged
    return syllable_data

def update_json_file(json_file_path, all_syllable_data):
    """
    Loads the JSON file, adds syllable_labels, and saves it back.
    """
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}. Cannot update.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from {json_file_path}. Error: {e}. Cannot update.")
        return

    for entry in json_data:
        filename_json = entry.get("filename")
        if filename_json:
            if filename_json in all_syllable_data:
                entry['syllable_labels'] = all_syllable_data[filename_json]
            else:
                # Ensure syllable_labels key exists even if no annotations
                if 'syllable_labels' not in entry or not isinstance(entry['syllable_labels'], dict):
                     entry['syllable_labels'] = {}
        else:
            print(f"Warning: Entry in JSON without 'filename' key: {entry}") 