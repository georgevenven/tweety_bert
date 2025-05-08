import sys
import os
import numpy as np
from glob import glob
import argparse

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add both the project root and src directories to the Python path
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from src.analysis import ComputerClusterPerformance

# --- Add argparse setup ---
parser = argparse.ArgumentParser(description="Calculate V-Measure scores for UMAP folds.")
parser.add_argument("npz_directory", type=str, help="Directory containing the NPZ files for UMAP folds.")
args = parser.parse_args()
# --- End argparse setup ---

# Collect all npz files in the directory
npz_files = glob(os.path.join(args.npz_directory, "*.npz"))

# List to store all V-measures
all_measures = []

# Compute V-measure for each file
for npz_file in npz_files:
    cluster_performance = ComputerClusterPerformance(labels_path=[npz_file])
    v_measure_dict = cluster_performance.compute_vmeasure_score()
    v_measure = v_measure_dict['V-measure'][0]
    all_measures.append(v_measure)
    
    # Extract the filename for reporting
    filename = os.path.basename(npz_file)
    print(f"{filename}: {v_measure:.4f}")

# Calculate overall statistics
overall_avg = np.mean(all_measures)
overall_std = np.std(all_measures)
min_value = np.min(all_measures)
max_value = np.max(all_measures)

print(f"\nOverall Average V-measure: {overall_avg:.2f} ± {overall_std:.2f}")
print(f"Range: {min_value:.2f} - {max_value:.2f}")
