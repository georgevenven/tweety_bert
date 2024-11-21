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

# List of directories to clean
directories_to_clean = [
    '/media/george-vengrovski/George-SSD/llb_stuff/llb3_test',
    '/media/george-vengrovski/George-SSD/llb_stuff/llb3_train',
    '/media/george-vengrovski/George-SSD/llb_stuff/llb11_test',
    '/media/george-vengrovski/George-SSD/llb_stuff/llb11_train',
    '/media/george-vengrovski/George-SSD/llb_stuff/llb16_test',
    '/media/george-vengrovski/George-SSD/llb_stuff/llb16_train'
]

# Process each directory
for directory in directories_to_clean:
    print(f"\nProcessing directory: {directory}")
    delete_small_npz_files(directory)
