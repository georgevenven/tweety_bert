import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from collections import Counter
from typing import Dict, List, Tuple, Union
import numpy.typing as npt
from tqdm import tqdm
from scipy.stats import pearsonr

def syllable_to_phrase_labels(arr, silence=-1):
    new_arr = np.array(arr, dtype=int)
    current_syllable = None
    start_of_phrase_index = None
    first_non_silence_label = None  # To track the first non-silence syllable

    for i, value in enumerate(new_arr):
        if value != silence and value != current_syllable:
            if start_of_phrase_index is not None:
                new_arr[start_of_phrase_index:i] = current_syllable
            current_syllable = value
            start_of_phrase_index = i
            
            if first_non_silence_label is None:  # Found the first non-silence label
                first_non_silence_label = value

    if start_of_phrase_index is not None:
        new_arr[start_of_phrase_index:] = current_syllable

    # Replace the initial silence with the first non-silence syllable label
    if new_arr[0] == silence and first_non_silence_label is not None:
        for i in range(len(new_arr)):
            if new_arr[i] != silence:
                break
            new_arr[i] = first_non_silence_label

    return new_arr

def smooth_labels(labels, window_size=50):
    labels = np.array(labels)
    for i in range(len(labels)):
        if labels[i] == -1:
            left = right = i
            while left >= 0 or right < len(labels):
                if left >= 0 and labels[left] != -1:
                    labels[i] = labels[left]
                    break
                if right < len(labels) and labels[right] != -1:
                    labels[i] = labels[right]
                    break
                left -= 1
                right += 1
    
    smoothed_labels = np.zeros_like(labels)
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2 + 1)
        window = labels[start:end]
        unique, counts = np.unique(window, return_counts=True)
        smoothed_labels[i] = unique[np.argmax(counts)]
    return smoothed_labels

class SequenceAnalyzer:
    @staticmethod
    def create_shared_area_matrix(ground_truth: npt.NDArray, 
                                predicted: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Create and normalize shared area matrix between ground truth and predicted labels."""
        unique_ground_truth = np.unique(ground_truth)
        unique_predicted = np.unique(predicted)
        
        # Create shared area matrix
        shared_matrix = np.zeros((len(unique_ground_truth), len(unique_predicted)))
        
        for i, gt_label in enumerate(unique_ground_truth):
            for j, pred_label in enumerate(unique_predicted):
                shared_matrix[i, j] = np.sum((ground_truth == gt_label) & 
                                           (predicted == pred_label))
        
        # Normalize rows
        row_sums = shared_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(shared_matrix, row_sums, 
                                    where=row_sums != 0)
        
        return normalized_matrix, unique_ground_truth, unique_predicted

    @staticmethod
    def find_optimal_mapping(normalized_matrix: npt.NDArray,
                           unique_ground_truth: npt.NDArray,
                           unique_predicted: npt.NDArray) -> Tuple[Dict[int, int], npt.NDArray, npt.NDArray]:
        """Find optimal mapping between predicted and ground truth labels."""
        cost_matrix = -normalized_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping dictionary
        mapping = {unique_predicted[col]: unique_ground_truth[row]
                  for row, col in zip(row_ind, col_ind)}
        
        # Map unmapped labels to noise (-1)
        all_predicted = set(unique_predicted)
        mapped_predicted = set(mapping.keys())
        for label in (all_predicted - mapped_predicted):
            mapping[label] = -1
            
        return mapping, row_ind, col_ind

    @staticmethod
    def evaluate_mapping(ground_truth: npt.NDArray, 
                        predicted: npt.NDArray,
                        mapping: Dict[int, int]) -> Tuple[int, float]:
        """Evaluate mapping quality."""
        remapped = np.array([mapping[label] for label in predicted])
        differences = np.sum(remapped != ground_truth)
        difference_percentage = (differences / len(ground_truth)) * 100
        return differences, difference_percentage

    @staticmethod
    def visualize_mapping(normalized_matrix: npt.NDArray,
                         row_ind: npt.NDArray,
                         col_ind: npt.NDArray,
                         unique_ground_truth: npt.NDArray,
                         unique_predicted: npt.NDArray,
                         title: str) -> None:
        """Visualize the mapping matrix and save to disk."""
        optimal_matrix = normalized_matrix[row_ind][:, col_ind]
        optimal_ground_truth = unique_ground_truth[row_ind]
        optimal_predicted = unique_predicted[col_ind]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(optimal_matrix, annot=False, cmap="viridis",
                   xticklabels=optimal_predicted,
                   yticklabels=optimal_ground_truth)
        plt.xlabel('HDBSCAN Labels')
        plt.ylabel('Ground Truth Phrase Labels')
        plt.title(f'Optimally Arranged Normalized Shared Area Matrix\n{title}')
        
        # Save the plot to disk instead of showing it
        filename = f"mapping_matrix_{title.replace(' ', '_').replace(':', '_')}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free up memory

# Process the files
files = ["/media/george-vengrovski/flash-drive/llb3.npz", "/media/george-vengrovski/flash-drive/llb11.npz", "/media/george-vengrovski/flash-drive/llb16.npz"]
analyzer = SequenceAnalyzer()

# for file in files:
#     # Load data
#     data = np.load(file)
#     ground_truth_labels = data['ground_truth_labels']
#     hdbscan_labels = data['hdbscan_labels']
    
#     # Process labels using your functions
#     hdbscan_labels = smooth_labels(hdbscan_labels, window_size=0)
#     ground_truth_phrase_labels = syllable_to_phrase_labels(ground_truth_labels, silence=0)
    
#     # Create shared area matrix
#     normalized_matrix, unique_ground_truth, unique_predicted = analyzer.create_shared_area_matrix(
#         ground_truth_phrase_labels, hdbscan_labels)
    
#     # Find optimal mapping
#     mapping, row_ind, col_ind = analyzer.find_optimal_mapping(
#         normalized_matrix, unique_ground_truth, unique_predicted)
    
#     # Evaluate mapping
#     differences, difference_percentage = analyzer.evaluate_mapping(
#         ground_truth_phrase_labels, hdbscan_labels, mapping)
    
#     # Print results
#     print(f"\nFile: {file}")
#     print(f"Number of differences: {differences}")
#     print(f"Percentage of differences: {difference_percentage:.2f}%")
    
#     # Visualize results
#     analyzer.visualize_mapping(
#         normalized_matrix, row_ind, col_ind,
#         unique_ground_truth, unique_predicted,
#         f"File: {file}"
#     )

import numpy as np
from scipy.optimize import linear_sum_assignment

def create_merged_label_mapping(normalized_matrix, unique_ground_truth_labels, unique_hdbscan_labels):
    """
    Creates a mapping that allows multiple HDBSCAN labels to map to a single ground truth label
    when that ground truth label is the best match for each HDBSCAN label.
    """
    col_sums = normalized_matrix.sum(axis=0, keepdims=True)
    col_normalized_matrix = normalized_matrix / col_sums
    
    label_mapping = {}
    for hdbscan_idx, hdbscan_label in enumerate(unique_hdbscan_labels):
        best_match_idx = np.argmax(col_normalized_matrix[:, hdbscan_idx])
        label_mapping[hdbscan_label] = unique_ground_truth_labels[best_match_idx]
    
    return label_mapping

def reduce_phrases(arr, remove_silence=True):
    current_element = arr[0]
    reduced_list = []

    for i, value in enumerate(arr):
        if value != current_element:
            reduced_list.append(current_element)
            current_element = value

        # append last phrase
        if i == len(arr) - 1:
            reduced_list.append(current_element)

    if remove_silence:
        reduced_list = [value for value in reduced_list if value != 0]

    return np.array(reduced_list)

def calculate_average_phrase_entropy(labels, target_label):
    transition_counts = Counter()
    total_transitions = 0
    in_target_phrase = False

    for i in range(len(labels) - 1):
        current_label = labels[i]
        next_label = labels[i + 1]

        if current_label == target_label:
            in_target_phrase = True
        elif in_target_phrase:
            transition_counts[next_label] += 1
            total_transitions += 1
            in_target_phrase = False

    if total_transitions == 0:
        return 0.0

    entropy = 0.0
    for count in transition_counts.values():
        probability = count / total_transitions
        entropy -= probability * np.log(probability)

    return entropy

def calculate_average_phrase_length(labels, target_label):
    phrase_lengths = []
    current_length = 0

    for label in labels:
        if label == target_label:
            current_length += 1
        else:
            if current_length > 0:
                phrase_lengths.append(current_length)
                current_length = 0

    if current_length > 0:
        phrase_lengths.append(current_length)

    return np.mean(phrase_lengths) if phrase_lengths else 0

def calculate_pearson(y_true, y_pred):
    """Calculate Pearson correlation coefficient"""
    correlation, _ = pearsonr(y_true, y_pred)
    return correlation

# Function to process files with a given smoothing window
def process_files(smoothing_window):
    ground_truth_entropies = []
    hdbscan_entropies = []
    ground_truth_avg_phrase_lengths = []
    hdbscan_avg_phrase_lengths = []
    file_ids = []

    analyzer = SequenceAnalyzer()

    for file_id, file in enumerate(files):
        # Load data
        data = np.load(file)
        ground_truth_labels = data['ground_truth_labels']
        hdbscan_labels = data['hdbscan_labels']

        # Apply smoothing
        hdbscan_labels = smooth_labels(hdbscan_labels, window_size=smoothing_window)
        ground_truth_phrase_labels = syllable_to_phrase_labels(ground_truth_labels, silence=0)

        # Create shared area matrix
        normalized_matrix, unique_ground_truth, unique_predicted = analyzer.create_shared_area_matrix(
            ground_truth_phrase_labels, hdbscan_labels)

        # Find optimal linear mapping
        linear_mapping, row_ind, col_ind = analyzer.find_optimal_mapping(
            normalized_matrix, unique_ground_truth, unique_predicted)

        # Process linear mapping
        for hdbscan_label, gt_label in linear_mapping.items():
            if gt_label != -1:  # Exclude noise
                gt_entropy = calculate_average_phrase_entropy(ground_truth_phrase_labels, gt_label)
                hdbscan_entropy = calculate_average_phrase_entropy(hdbscan_labels, hdbscan_label)
                gt_phrase_length = calculate_average_phrase_length(ground_truth_phrase_labels, gt_label)
                hdbscan_phrase_length = calculate_average_phrase_length(hdbscan_labels, hdbscan_label)

                ground_truth_entropies.append(gt_entropy)
                hdbscan_entropies.append(hdbscan_entropy)
                ground_truth_avg_phrase_lengths.append(gt_phrase_length)
                hdbscan_avg_phrase_lengths.append(hdbscan_phrase_length)
                file_ids.append(file_id)

    return (ground_truth_entropies, hdbscan_entropies, ground_truth_avg_phrase_lengths, hdbscan_avg_phrase_lengths, file_ids)

# Modify the plot_and_calculate_r2 function
def plot_correlation_comparisons(results, smoothing_window, max_entropy, max_phrase_length):
    ground_truth_entropies, hdbscan_entropies, ground_truth_avg_phrase_lengths, hdbscan_avg_phrase_lengths, file_ids = results
    
    # Plot transition entropy
    plt.figure(figsize=(12, 6), dpi=300)  # Higher DPI for better resolution
    scatter = sns.scatterplot(x=ground_truth_entropies, y=hdbscan_entropies, hue=file_ids, 
                    size=ground_truth_entropies, sizes=(50, 200), palette='viridis', 
                    alpha=0.7, edgecolor='w', linewidth=0.5)
    
    # Plot y=x line
    x_range = np.linspace(0, max_entropy, 100)
    plt.plot(x_range, x_range, color='red', linestyle='--', label='y=x')
    
    # Calculate and display Pearson correlation
    pearson = calculate_pearson(np.array(ground_truth_entropies), np.array(hdbscan_entropies))
    plt.text(0.05, 0.95, f'Pearson r = {pearson:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Ground Truth Average Phrase Entropy', fontsize=12)
    plt.ylabel('HDBSCAN Average Phrase Entropy', fontsize=12)
    plt.title(f'Average Phrase Entropy Comparison\n(Smoothing Window: {smoothing_window})', fontsize=14)
    plt.legend(title='File ID', fontsize=10, title_fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, max_entropy)
    plt.ylim(0, max_entropy)
    plt.tight_layout()
    plt.savefig(f'phrase_entropy_correlation_{smoothing_window}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot average phrase length
    plt.figure(figsize=(12, 6), dpi=300)  # Higher DPI for better resolution
    sns.scatterplot(x=ground_truth_avg_phrase_lengths, y=hdbscan_avg_phrase_lengths, 
                    hue=file_ids, size=ground_truth_avg_phrase_lengths, sizes=(50, 200), 
                    palette='viridis', alpha=0.7, edgecolor='w', linewidth=0.5)
    
    # Plot y=x line
    x_range = np.linspace(0, max_phrase_length, 100)
    plt.plot(x_range, x_range, color='red', linestyle='--', label='y=x')
    
    # Calculate and display Pearson correlation
    pearson = calculate_pearson(np.array(ground_truth_avg_phrase_lengths), np.array(hdbscan_avg_phrase_lengths))
    plt.text(0.05, 0.95, f'Pearson r = {pearson:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Ground Truth Average Phrase Length', fontsize=12)
    plt.ylabel('HDBSCAN Average Phrase Length', fontsize=12)
    plt.title(f'Average Phrase Length Comparison\n(Smoothing Window: {smoothing_window})', fontsize=14)
    plt.legend(title='File ID', fontsize=10, title_fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, max_phrase_length)
    plt.ylim(0, max_phrase_length)
    plt.tight_layout()
    plt.savefig(f'phrase_length_correlation_{smoothing_window}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return pearson_entropy, pearson_phrase_length

# New function to process files for multiple window sizes
def process_multiple_window_sizes():
    window_sizes = list(range(0, 300, 100))
    pearson_values_entropy = []
    pearson_values_phrase_length = []
    pearson_values_combined = []

    for window_size in tqdm(window_sizes, desc="Processing window sizes"):
        results = process_files(smoothing_window=window_size)
        pearson_entropy = calculate_pearson(np.array(results[0]), np.array(results[1]))
        pearson_phrase_length = calculate_pearson(np.array(results[2]), np.array(results[3]))
        avg_pearson = (pearson_entropy + pearson_phrase_length) / 2
        pearson_values_entropy.append((window_size, pearson_entropy))
        pearson_values_phrase_length.append((window_size, pearson_phrase_length))
        pearson_values_combined.append((window_size, avg_pearson))

    return pearson_values_entropy, pearson_values_phrase_length, pearson_values_combined

# Calculate the maximum entropy and phrase length across all datasets
def calculate_max_values():
    max_entropy = 0
    max_phrase_length = 0

    for file in files:
        data = np.load(file)
        ground_truth_labels = data['ground_truth_labels']
        ground_truth_phrase_labels = syllable_to_phrase_labels(ground_truth_labels, silence=0)
        
        for label in np.unique(ground_truth_phrase_labels):
            if label != 0:  # Exclude silence
                entropy = calculate_average_phrase_entropy(ground_truth_phrase_labels, label)
                phrase_length = calculate_average_phrase_length(ground_truth_phrase_labels, label)
                max_entropy = max(max_entropy, entropy)
                max_phrase_length = max(max_phrase_length, phrase_length)

    return max_entropy, max_phrase_length

# Calculate max values
max_entropy, max_phrase_length = calculate_max_values()

# Process files with multiple window sizes
pearson_values_entropy, pearson_values_phrase_length, pearson_values_combined = process_multiple_window_sizes()

# Plot R² values
plt.figure(figsize=(12, 8))
window_sizes, pearson_entropy = zip(*pearson_values_entropy)
_, pearson_phrase_length = zip(*pearson_values_phrase_length)
_, pearson_combined = zip(*pearson_values_combined)

plt.scatter(window_sizes, pearson_entropy, label='Entropy', alpha=0.7)
plt.scatter(window_sizes, pearson_phrase_length, label='Phrase Duration', alpha=0.7)
plt.scatter(window_sizes, pearson_combined, label='Combined Average', alpha=0.7)

plt.xlabel('Smoothing Window Size', fontsize=14)
plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
plt.title('Pearson Correlation for Different Smoothing Window Sizes', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig('pearson_correlation_by_window_size.png', dpi=300)
plt.close()

print("Processing complete. Check the generated plots for results.")
