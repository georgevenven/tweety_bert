import numpy as np

# Load both files
indices = np.load('/home/george-vengrovski/Downloads/USA5494_Seasonality_4_Groups/file_indices.npy', allow_pickle=True)
file_map = np.load('/home/george-vengrovski/Downloads/USA5494_Seasonality_4_Groups/file_map.npy', allow_pickle=True)

# Print the type and shape of file_map
print("Type of file_map:", type(file_map))
print("Shape of file_map:", file_map.shape if hasattr(file_map, 'shape') else "No shape")
print("Content of file_map:", file_map)

# Extract unique months from file_map
months = set()
for item in file_map.item().values():  # Convert numpy array to dictionary and get values
    filename = item[0]  # Get the filename from the tuple
    # Split by underscore and get the month number (second underscore)
    month = int(filename.split('_')[2].split('.')[0])
    months.add(month)

# Print sorted months
print("Unique months:", sorted(list(months)))