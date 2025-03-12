import numpy as np
import os
folder = "/home/george-vengrovski/Music/labeled_song_dataset/train"

# # For song Labels
# Class 0: Non Song
# Class 1: Song
# Class 2: Calls 

# Dictionary to store counts for each class
class_counts = {}

# First pass - collect unique classes
for file in os.listdir(folder):
    if file.endswith(".npz"):
        f = np.load(f"{folder}/{file}", allow_pickle=True)
        song_data = f['song']
        
        # Get unique classes in this file and initialize counts if not seen before
        unique_classes = np.unique(song_data)
        for cls in unique_classes:
            if cls not in class_counts:
                class_counts[cls] = 0

# Second pass - count occurrences
for file in os.listdir(folder):
    if file.endswith(".npz"):
        f = np.load(f"{folder}/{file}", allow_pickle=True)
        song_data = f['song']
        
        # Count occurrences of each class
        unique_classes, counts = np.unique(song_data, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            class_counts[cls] += count

# Print results
print("\nTotal time bins per class:")
for cls in sorted(class_counts.keys()):
    print(f"Class {cls}: {class_counts[cls]} time bins")
