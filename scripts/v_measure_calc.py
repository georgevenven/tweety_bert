import sys
import os
import numpy as np
from glob import glob

sys.path.append("src")

from analysis import ComputerClusterPerformance

# Specify the directory containing the npz files
npz_directory = "/media/george-vengrovski/George-SSD/folds_for_paper_llb"

# Collect all npz files in the directory
npz_files = glob(os.path.join(npz_directory, "*.npz"))

v_measures = []

# Compute V-measure for each file
for npz_file in npz_files:
    cluster_performance = ComputerClusterPerformance(labels_path=[npz_file])
    v_measure_dict = cluster_performance.compute_vmeasure_score()
    
    # Extract the mean V-measure value from the dictionary
    # v_measure_dict['V-measure'] is a tuple (mean, std_error)
    # We just want the mean (first element)
    v_measures.append(v_measure_dict['V-measure'][0])

# Convert to a numpy array for easy math
v_measures = np.array(v_measures)

# Calculate statistics if we have any v-measures
if len(v_measures) > 0:
    # Print individual V-measures
    print("\nIndividual V-measures:")
    for i, v in enumerate(v_measures, 1):
        print(f"File {i}: {v}")
    print("\nSummary Statistics:")
    
    v_mean = np.mean(v_measures)
    v_std = np.std(v_measures)
    v_min = np.min(v_measures)
    v_max = np.max(v_measures)

    # Print the results
    print("V-measure scores for directory:", npz_directory)
    print("Mean:", v_mean)
    print("Standard Deviation:", v_std)
    print("Range: [", v_min, ",", v_max, "]")
else:
    print("No V-measure values were found. Check if the NPZ files contain valid data.")
