import os

# This script deletes .npz files under 100KB in a specified directory.
# These files are assumed to be generated incorrectly and are not needed.

def delete_small_npz_files(directory):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file has a .npz extension
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            # Get the size of the file in bytes
            file_size = os.path.getsize(file_path)
            # Check if the file size is less than 100KB (100 * 1024 bytes)
            if file_size < 100 * 1024:
                # Delete the file if it is under 100KB
                os.remove(file_path)
                print(f"Deleted {file_path} (size: {file_size} bytes)")

# Specify the directory to clean up
directory_to_clean = '/path/to/your/directory'

# Call the function to delete small .npz files
delete_small_npz_files(directory_to_clean)
