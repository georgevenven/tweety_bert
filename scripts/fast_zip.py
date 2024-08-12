import os
import zipfile
from datetime import datetime
from tqdm import tqdm

def zip_folder(folder_path, output_path):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the output zip file name with timestamp
    zip_filename = f"{os.path.basename(folder_path)}_{timestamp}.zip"
    zip_path = os.path.join(output_path, zip_filename)
    
    # Get the total number of files in the folder
    total_files = sum(len(files) for _, _, files in os.walk(folder_path))
    
    # Create a ZipFile object with compression level 9 (maximum compression)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as zipf:
        # Walk through the directory tree
        with tqdm(total=total_files, unit='files', desc='Zipping progress') as pbar:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add each file to the zip archive
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
                    pbar.update(1)
    
    print(f"Folder zipped successfully. Output file: {zip_path}")

# Specify the folder path to be zipped
folder_to_zip = "/media/george-vengrovski/disk1/multispecies_data_set"

# Specify the output directory for the zip file
output_directory = "/media/george-vengrovski/disk1/zipped"

# Call the function to zip the folder
zip_folder(folder_to_zip, output_directory)