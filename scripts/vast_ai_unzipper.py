import zipfile
import os
from tqdm.notebook import tqdm  # Import tqdm for notebook to get a better-looking, more functional progress bar in Jupyter/Colab

# Define the path to the zip file and the extraction directory
zip_file_path = '/workspace/tweety_bert_paper.zip'  # Adjust this path if your file is in a different location
extraction_directory = '/workspace/'  # You can change this directory as per your requirement

# Create the extraction directory if it doesn't exist
if not os.path.exists(extraction_directory):
    os.makedirs(extraction_directory)

# Function to unzip the file with a progress bar
def unzip_file_with_progress(zip_path, extract_to):
    # Open the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get the list of file names inside the zip
        file_names = zip_ref.namelist()
        # Initialize the progress bar
        with tqdm(total=len(file_names), desc="Extracting files") as pbar:
            for file in file_names:
                # Extract each file to the directory
                zip_ref.extract(member=file, path=extract_to)
                # Update the progress bar
                pbar.update(1)
    print(f'Files extracted to {extract_to}')

# Unzip the file with a progress bar
unzip_file_with_progress(zip_file_path, extraction_directory)
