import json
from datetime import datetime

def parse_date_time(file_path, format=None):
    parts = file_path.split('_')
    parts[-1] = parts[-1].replace('.wav', '')
    try:
        # Determine format based on filename prefix
        if format is None:
            format = "yarden" if file_path.startswith("llb") else "standard"

        if format == "yarden":
            # Example format: llb3_1688_2018_04_27_12_27_27.wav
            year = int(parts[2])
            month = int(parts[3])
            day = int(parts[4])
            hour = int(parts[5])
            minute = int(parts[6])
            second = int(parts[7])
            file_date = datetime(year, month, day, hour, minute, second)
            file_name = parts[0]
        elif format == "standard":
            # Example format adjustment as per user's requirement
            # Assuming parts: ['USA5288', '45355.32428022', '3', '4', '9', '0', '28']
            month = int(parts[2])
            day = int(parts[3])
            hour = int(parts[4])
            minute = int(parts[5])
            second = int(parts[6])
            file_date = datetime(2024, month, day, hour, minute, second)
            file_name = parts[0]
        else:
            print(f"Unknown format: {format}")
            return None, None
    except (ValueError, IndexError) as e:
        print("parts:", *[f" {part}" for part in parts])
        print(f"Invalid date format in file path: {file_path}\nError: {e}")
        return None, None

    return file_date, file_name

def add_datetime_to_json(json_data):
    for entry in json_data:
        filename = entry["filename"]
        # Use the parse_date_time function directly
        file_date, _ = parse_date_time(None, filename)  # Assuming parse_date_time is not a method of a class
        if file_date:
            entry["datetime"] = file_date.isoformat()
    return json_data

def main():
    # Load the JSON data
    with open('/media/george-vengrovski/flash-drive/new_season/onset_offset_results.json', 'r') as file:
        data = json.load(file)

    # Add datetime information
    updated_data = add_datetime_to_json(data)

    # Save the updated JSON data
    with open('/media/george-vengrovski/flash-drive/new_season/onset_offset_results.json', 'w') as file:
        json.dump(updated_data, file, indent=4)

if __name__ == "__main__":
    main()
