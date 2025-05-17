import os
import xml.etree.ElementTree as ET
import json

# --- Configuration ---
BASE_DIR = "/media/george-vengrovski/disk2/bengalese-finch/3470165"
JSON_FILE_PATH = "/home/george-vengrovski/Documents/projects/tweety_bert_paper/3470165_detect.json"
SAMPLING_RATE = 32000  # Hz
NUM_BIRD_FOLDERS = 11 # For Bird0 to Bird10

def parse_and_collate_annotations(base_dir, num_bird_folders, sampling_rate):
    """
    Parses Annotation.xml files from BirdN directories and collates syllable data.
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
                    try:
                        note_position_samples = int(note_elem.findtext("Position"))
                        note_length_samples = int(note_elem.findtext("Length"))
                        label_str = note_elem.findtext("Label")
                        
                        if label_str is None:
                            print(f"Warning: Missing Label in a note in {xml_file_path} for {target_json_filename}. Skipping note.")
                            continue
                        try:
                            label_int = int(label_str) # Convert label to integer
                        except (ValueError, TypeError):
                            # Handle non-integer labels by assigning a unique triple-digit integer
                            if '_nonint_label_map' not in locals():
                                _nonint_label_map = {}
                                _nonint_label_counter = 100
                            if target_json_filename not in _nonint_label_map:
                                _nonint_label_map[target_json_filename] = {}
                            if label_str not in _nonint_label_map[target_json_filename]:
                                # Find next available triple-digit integer not in use for this file
                                used_labels = set(all_syllable_data[target_json_filename].keys())
                                while _nonint_label_counter in used_labels:
                                    _nonint_label_counter += 1
                                _nonint_label_map[target_json_filename][label_str] = _nonint_label_counter
                                _nonint_label_counter += 1
                            label_int = _nonint_label_map[target_json_filename][label_str]

                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid note data in {xml_file_path} for {target_json_filename} (Label: {label_str if 'label_str' in locals() else 'N/A'}). Error: {e}. Skipping note.")
                        continue

                    # Calculate absolute onset and offset in seconds
                    syllable_onset_abs_samples = sequence_position_samples + note_position_samples
                    syllable_onset_sec = (syllable_onset_abs_samples / sampling_rate)
                    syllable_offset_sec = ((syllable_onset_abs_samples + note_length_samples) / sampling_rate)
                    
                    if label_int not in all_syllable_data[target_json_filename]:
                        all_syllable_data[target_json_filename][label_int] = []
                    
                    all_syllable_data[target_json_filename][label_int].append((syllable_onset_sec, syllable_offset_sec))

        except ET.ParseError as e:
            print(f"Error parsing XML file {xml_file_path}: {e}. Skipping this file.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {xml_file_path}: {e}. Skipping this file.")
            continue
            
    return all_syllable_data

def make_intervals_contiguous(syllable_data):
    """
    Sorts and merges overlapping/adjacent time intervals for each label.
    """
    for filename in syllable_data:
        for label_int in syllable_data[filename]:
            intervals = sorted(syllable_data[filename][label_int]) # Sort by onset time
            if not intervals:
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
            syllable_data[filename][label_int] = merged
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
                entry['syllable_labels'] = {} # No annotations for this file
        else:
            print(f"Warning: Entry in JSON without 'filename' key: {entry}")

    try:
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Successfully updated JSON file: {json_file_path}")
    except IOError as e:
        print(f"Error: Could not write updated JSON to {json_file_path}. Error: {e}")

# --- Main execution ---
if __name__ == "__main__":
    print("Starting annotation processing...")
    
    # 1. Parse XMLs and collate annotations
    raw_annotations = parse_and_collate_annotations(BASE_DIR, NUM_BIRD_FOLDERS, SAMPLING_RATE)
    
    # 2. Make intervals contiguous
    processed_annotations = make_intervals_contiguous(raw_annotations)
    
    # 3. Update the JSON file
    update_json_file(JSON_FILE_PATH, processed_annotations)
    
    print("Processing complete.")