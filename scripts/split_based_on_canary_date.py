import os
import shutil
from datetime import datetime, timedelta

def split_files(source_folder, date_splits, output_directory):
    # Convert date splits to datetime objects
    split_dates = [datetime.strptime(date, "%m-%d").replace(year=2000) for date in date_splits]
    
    # Create destination folders
    dest_folders = [f"before_{date_splits[0]}"] + [f"{date_splits[i]}_to_{date_splits[i+1]}" for i in range(len(date_splits)-1)] + [f"after_{date_splits[-1]}"]
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
                    
                    file_date = datetime(2000, month, day)
                    
                    # Adjust for year-end wraparound
                    if month < split_dates[0].month:
                        file_date = file_date.replace(year=2001)
                    
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
source_folder = "/media/george-vengrovski/disk1/usa_5288/usa_5288_test"
output_directory = "/media/george-vengrovski/disk1/usa_5288/split_output"
date_splits = ["3-20", "3-30", "4-15"]
split_files(source_folder, date_splits, output_directory)