import os
import shutil
from datetime import datetime, timedelta

def split_files(source_folder, date_splits, output_directory):
    # Extract bird ID from the source folder name
    bird_id = os.path.basename(source_folder).split('_')[0]

    # Convert date splits to datetime objects
    split_dates = [datetime.strptime(date, "%m-%d").replace(year=2024) for date in date_splits]
    
    # Create destination folders with bird ID
    dest_folders = [f"{bird_id}_before_{date_splits[0]}"] + [f"{bird_id}_{date_splits[i]}_to_{date_splits[i+1]}" for i in range(len(date_splits)-1)] + [f"{bird_id}_after_{date_splits[-1]}"]
    for folder in dest_folders:
        os.makedirs(os.path.join(output_directory, folder), exist_ok=True)

    # Process files
    for filename in os.listdir(source_folder):
        if filename.endswith('.npz'):
            parts = filename.split('_')
            if len(parts) >= 7:
                try:
                    month, day, hour = int(parts[2]), int(parts[3]), int(parts[4])
                    if month < 1 or month > 12 or day < 1 or day > 31 or hour < 0 or hour > 23:
                        raise ValueError(f"Invalid date/time: month={month}, day={day}, hour={hour}")
                    
                    file_date = datetime(2024, month, day)
                    
                    # Determine which group the file belongs to
                    group_index = next((i for i, split_date in enumerate(split_dates) if file_date < split_date), len(split_dates))
                    dest_folder = dest_folders[group_index]
                    
                    # Copy the file to the appropriate folder
                    shutil.copy2(
                        os.path.join(source_folder, filename),
                        os.path.join(output_directory, dest_folder, filename)
                    )
                    print(f"Copied {filename} to {dest_folder}")
                except ValueError as e:
                    print(f"Skipping file due to error: {filename}. Error: {str(e)}")
            else:
                print(f"Skipping file with unexpected format: {filename}")

# Example usage
source_folder = "/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5271_specs"
output_directory = "/media/george-vengrovski/Extreme SSD/sham lesioned birds"
date_splits = ["3-07"]
split_files(source_folder, date_splits, output_directory)