import sys
import os
import numpy as np
from glob import glob

# Update the path to include the project root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import ComputerClusterPerformance

# Specify the directory containing the npz files
npz_directory = "/media/george-vengrovski/George-SSD/umaps_across_all_layers"

# Collect all npz files in the directory
npz_files = glob(os.path.join(npz_directory, "*.npz"))

# Initialize dictionaries to store V-measures by layer and sublayer
layer_sublayer_measures = {}

# Compute V-measure for each file
for npz_file in npz_files:
    cluster_performance = ComputerClusterPerformance(labels_path=[npz_file])
    v_measure_dict = cluster_performance.compute_vmeasure_score()
    v_measure = v_measure_dict['V-measure'][0]
    
    # Extract the filename and parse layer and sublayer
    filename = os.path.basename(npz_file)
    
    parts = filename.split('_')
    layer = parts[1]  # e.g., 'layer3'
    sublayer = '_'.join(parts[2:])  # e.g., 'feed_forward_output.npz'
    
    # Store the V-measure in the dictionary
    if (layer, sublayer) not in layer_sublayer_measures:
        layer_sublayer_measures[(layer, sublayer)] = []
    layer_sublayer_measures[(layer, sublayer)].append(v_measure)

# Calculate averages and determine the best layer and sublayer
best_layer_sublayer = None
best_layer = None
best_sublayer = None
best_layer_sublayer_avg = -1
best_layer_avg = -1
best_sublayer_avg = -1

layer_averages = {}
sublayer_averages = {}

for (layer, sublayer), measures in layer_sublayer_measures.items():
    avg_measure = np.mean(measures)
    
    # Update best layer+sublayer
    if avg_measure > best_layer_sublayer_avg:
        best_layer_sublayer_avg = avg_measure
        best_layer_sublayer = (layer, sublayer)
    
    # Update layer averages
    if layer not in layer_averages:
        layer_averages[layer] = []
    layer_averages[layer].append(avg_measure)
    
    # Update sublayer averages
    if sublayer not in sublayer_averages:
        sublayer_averages[sublayer] = []
    sublayer_averages[sublayer].append(avg_measure)

# Calculate best layer and sublayer
for layer, measures in layer_averages.items():
    avg_measure = np.mean(measures)
    if avg_measure > best_layer_avg:
        best_layer_avg = avg_measure
        best_layer = layer

for sublayer, measures in sublayer_averages.items():
    avg_measure = np.mean(measures)
    if avg_measure > best_sublayer_avg:
        best_sublayer_avg = avg_measure
        best_sublayer = sublayer

# Sort and print results in rank order
sorted_layer_sublayer = sorted(layer_sublayer_measures.items(), key=lambda x: np.mean(x[1]), reverse=True)
sorted_layers = sorted(layer_averages.items(), key=lambda x: np.mean(x[1]), reverse=True)
sorted_sublayers = sorted(sublayer_averages.items(), key=lambda x: np.mean(x[1]), reverse=True)

print("\nRanked Layer+Sublayer Combinations:")
for (layer, sublayer), measures in sorted_layer_sublayer:
    print(f"{layer} {sublayer}: {np.mean(measures):.4f}")

print("\nRanked Layers:")
for layer, measures in sorted_layers:
    print(f"{layer}: {np.mean(measures):.4f}")

print("\nRanked Sublayers:")
for sublayer, measures in sorted_sublayers:
    print(f"{sublayer}: {np.mean(measures):.4f}")

# Print best results
print(f"\nBest Layer+Sublayer: {best_layer_sublayer} with average V-measure: {best_layer_sublayer_avg}")
print(f"Best Layer: {best_layer} with average V-measure: {best_layer_avg}")
print(f"Best Sublayer: {best_sublayer} with average V-measure: {best_sublayer_avg}")
