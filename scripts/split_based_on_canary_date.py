import os
import shutil
from datetime import datetime, timedelta

def split_files_equal_groups(source_folder, experiment_date, num_groups, output_directory):
    # Extract bird ID from the source folder name
    bird_id = os.path.basename(source_folder).split('_')[0]

    # Convert experiment date to datetime object
    exp_date = datetime.strptime(experiment_date, "%m-%d").replace(year=2024)

    # Find the earliest and latest dates in the dataset
    earliest_date = None
    latest_date = None
    for filename in os.listdir(source_folder):
        if filename.endswith('.npz'):
            parts = filename.split('_')
            if len(parts) >= 7:
                try:
                    month, day = int(parts[2]), int(parts[3])
                    file_date = datetime(2024, month, day)
                    if earliest_date is None or file_date < earliest_date:
                        earliest_date = file_date
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                except ValueError:
                    continue

    # Ensure exp_date is within the dataset range
    if exp_date < earliest_date or exp_date > latest_date:
        raise ValueError("Experiment date is outside the dataset range")

    # Calculate the total date range
    total_days = (latest_date - earliest_date).days + 1

    # Adjust the start date to ensure exp_date is the first date of a group
    days_before_exp = (exp_date - earliest_date).days
    group_size = total_days // num_groups
    start_offset = days_before_exp % group_size
    adjusted_start = earliest_date + timedelta(days=start_offset)

    # Generate split dates
    split_dates = [adjusted_start + timedelta(days=i*group_size) for i in range(num_groups)]
    split_dates.append(latest_date + timedelta(days=1))  # Add one day to include the last day

    # Create destination folders
    dest_folders = [f"{bird_id}_group_{i+1}" for i in range(num_groups)]
    for folder in dest_folders:
        os.makedirs(os.path.join(output_directory, folder), exist_ok=True)

    # Initialize variables to track actual date ranges
    actual_ranges = [[] for _ in range(num_groups)]

    # Process files
    for filename in os.listdir(source_folder):
        if filename.endswith('.npz'):
            parts = filename.split('_')
            if len(parts) >= 7:
                try:
                    month, day = int(parts[2]), int(parts[3])
                    file_date = datetime(2024, month, day)
                    
                    # Determine which group the file belongs to
                    group_index = next((i for i, split_date in enumerate(split_dates[1:], 1) if file_date < split_date), 0) - 1
                    dest_folder = dest_folders[group_index]
                    
                    # Update actual date range for the group
                    actual_ranges[group_index].append(file_date)
                    
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

    # Print actual group date ranges
    for i, date_range in enumerate(actual_ranges):
        if date_range:
            start = min(date_range).strftime('%m-%d')
            end = max(date_range).strftime('%m-%d')
            print(f"Group {i+1}: {start} to {end}")
        else:
            print(f"Group {i+1}: No files")

# # Example usage
# source_folder = "/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5283_specs"
# output_directory = "/media/george-vengrovski/Extreme SSD/sham lesioned birds"
# experiment_date = "3-05"
# num_groups = 4
# split_files_equal_groups(source_folder, experiment_date, num_groups, output_directory)