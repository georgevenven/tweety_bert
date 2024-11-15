import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from collections import Counter
from typing import Dict, List, Tuple
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

    if window_size == 0:
        return labels

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
    def create_shared_area_matrix(ground_truth: np.ndarray, 
                                  predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create and normalize shared area matrix between ground truth and predicted labels."""
        unique_ground_truth = np.unique(ground_truth)
        unique_predicted = np.unique(predicted)
        
        # Create shared area matrix
        shared_matrix = np.zeros((len(unique_ground_truth), len(unique_predicted)))
        
        for i, gt_label in enumerate(unique_ground_truth):
            for j, pred_label in enumerate(unique_predicted):
                shared_matrix[i, j] = np.sum((ground_truth == gt_label) & 
                                             (predicted == pred_label))
        
        # Normalize columns (since we are mapping ground truth to HDBSCAN labels)
        col_sums = shared_matrix.sum(axis=0, keepdims=True)
        normalized_matrix = np.divide(shared_matrix, col_sums, where=col_sums != 0)
        
        return normalized_matrix, unique_ground_truth, unique_predicted

def create_merged_label_mapping(normalized_matrix, unique_ground_truth_labels, unique_hdbscan_labels, ground_truth_labels):
    """
    Creates a mapping that allows multiple ground truth labels to map to a single HDBSCAN label
    when that HDBSCAN label is the best match for each ground truth label.
    """
    row_sums = normalized_matrix.sum(axis=1, keepdims=True)
    row_normalized_matrix = np.divide(normalized_matrix, row_sums, 
                                      where=row_sums != 0)  # Avoid division by zero
    
    label_mapping = {}
    
    # Map each ground truth label to the HDBSCAN label with the highest normalized value
    for gt_idx, gt_label in enumerate(unique_ground_truth_labels):
        if row_sums[gt_idx, 0] > 0:  # Only map if there are any matches
            best_match_idx = np.argmax(normalized_matrix[gt_idx, :])
            label_mapping[gt_label] = unique_hdbscan_labels[best_match_idx]
        else:
            label_mapping[gt_label] = -1  # Map to noise if no matches
    
    # Ensure all ground truth labels are in the mapping
    all_gt_labels = np.unique(ground_truth_labels)
    for label in all_gt_labels:
        if label not in label_mapping:
            label_mapping[label] = -1  # Map any missing labels to noise
    
    return label_mapping

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

def calculate_weighted_pearson(y_true, y_pred, weights):
    """Calculate weighted Pearson correlation coefficient."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)

    mean_y_true = np.average(y_true, weights=weights)
    mean_y_pred = np.average(y_pred, weights=weights)

    cov = np.average((y_true - mean_y_true) * (y_pred - mean_y_pred), weights=weights)
    var_y_true = np.average((y_true - mean_y_true) ** 2, weights=weights)
    var_y_pred = np.average((y_pred - mean_y_pred) ** 2, weights=weights)

    correlation = cov / np.sqrt(var_y_true * var_y_pred)
    return correlation

# Function to process files with a given smoothing window
def process_files(smoothing_window):
    ground_truth_entropies = []
    hdbscan_entropies = []
    ground_truth_avg_phrase_lengths = []
    hdbscan_avg_phrase_lengths = []
    file_ids = []
    phrase_counts = []

    analyzer = SequenceAnalyzer()

    for file_id, file in enumerate(files):
        # Load data
        data = np.load(file)
        ground_truth_labels = data['ground_truth_labels']
        hdbscan_labels = data['hdbscan_labels']

        # Compute mapping before smoothing
        ground_truth_phrase_labels = syllable_to_phrase_labels(ground_truth_labels, silence=0)

        # Fill in -1 labels in hdbscan_labels by smoothing with window_size=0
        hdbscan_labels_filled = smooth_labels(hdbscan_labels, window_size=0)

        # Create shared area matrix using filled HDBSCAN labels
        normalized_matrix, unique_ground_truth, unique_predicted = analyzer.create_shared_area_matrix(
            ground_truth_phrase_labels, hdbscan_labels_filled)

        # Use merged labels mapping (ground truth labels to HDBSCAN labels)
        mapping = create_merged_label_mapping(
            normalized_matrix, unique_ground_truth, unique_predicted, ground_truth_phrase_labels)

        # Map ground truth labels before smoothing
        mapped_ground_truth_phrase_labels = np.array([mapping[label] for label in ground_truth_phrase_labels])

        # Apply smoothing to the original HDBSCAN labels
        smoothed_hdbscan_labels = smooth_labels(hdbscan_labels, window_size=smoothing_window)

        # Process each mapped HDBSCAN label
        unique_mapped_labels = np.unique(mapped_ground_truth_phrase_labels)
        for mapped_label in unique_mapped_labels:
            if mapped_label == -1:
                continue  # Skip noise

            gt_entropy = calculate_average_phrase_entropy(mapped_ground_truth_phrase_labels, mapped_label)
            hdbscan_entropy = calculate_average_phrase_entropy(smoothed_hdbscan_labels, mapped_label)

            gt_phrase_length = calculate_average_phrase_length(mapped_ground_truth_phrase_labels, mapped_label)
            hdbscan_phrase_length = calculate_average_phrase_length(smoothed_hdbscan_labels, mapped_label)

            # Collect counts
            phrase_count = np.sum(mapped_ground_truth_phrase_labels == mapped_label)

            ground_truth_entropies.append(gt_entropy)
            hdbscan_entropies.append(hdbscan_entropy)
            ground_truth_avg_phrase_lengths.append(gt_phrase_length)
            hdbscan_avg_phrase_lengths.append(hdbscan_phrase_length)
            file_ids.append(file_id)
            phrase_counts.append(phrase_count)

    return (ground_truth_entropies, hdbscan_entropies, 
            ground_truth_avg_phrase_lengths, hdbscan_avg_phrase_lengths, 
            file_ids, phrase_counts)

def plot_correlation_comparisons(results, smoothing_window, max_entropy, max_phrase_length):
    ground_truth_entropies, hdbscan_entropies, ground_truth_avg_phrase_lengths, hdbscan_avg_phrase_lengths, file_ids, phrase_counts = results

    # Normalize phrase_counts to sizes between min_size and max_size
    min_size = 50
    max_size = 300
    min_count = min(phrase_counts)
    max_count = max(phrase_counts)
    if max_count > min_count:
        sizes = [min_size + (count - min_count) / (max_count - min_count) * (max_size - min_size) for count in phrase_counts]
    else:
        sizes = [min_size for _ in phrase_counts]

    # Plot transition entropy
    plt.figure(figsize=(8, 8), dpi=300)

    # Plot y=x line first (without label)
    x_range = np.linspace(0, max_entropy, 100)
    plt.plot(x_range, x_range, color='red', linestyle='--', zorder=1)

    # Plot scatter points
    scatter = sns.scatterplot(x=ground_truth_entropies, y=hdbscan_entropies, 
                              hue=file_ids, 
                              size=sizes,
                              sizes=(min_size, max_size), palette='viridis', 
                              alpha=0.7, edgecolor='w', linewidth=0.5,
                              zorder=2)

    # Calculate and display Pearson correlation
    pearson = calculate_weighted_pearson(np.array(ground_truth_entropies), np.array(hdbscan_entropies), np.array(phrase_counts))
    plt.text(0.05, 0.95, f'Pearson r = {pearson:.3f}', 
             transform=plt.gca().transAxes, fontsize=18, 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Ground Truth Average Phrase Entropy', fontsize=24)
    plt.ylabel('HDBSCAN Average Phrase Entropy', fontsize=24)
    plt.title(f'Average Phrase Entropy Comparison\n(Smoothing Window: {smoothing_window})', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, max_entropy)
    plt.ylim(0, max_entropy)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    # Create legend with only Bird ID
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:len(set(file_ids))], [f'Bird {id+1}' for id in range(len(set(file_ids)))], 
              title='Bird ID', fontsize=18, title_fontsize=18)

    plt.savefig(f'phrase_entropy_correlation_{smoothing_window}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'phrase_entropy_correlation_{smoothing_window}.svg', format='svg', bbox_inches='tight')
    plt.close()

    # Plot average phrase length
    plt.figure(figsize=(8, 8), dpi=300)

    # Plot y=x line
    x_range = np.linspace(0, max_phrase_length, 100)
    plt.plot(x_range, x_range, color='red', linestyle='--', zorder=1)

    # Plot scatter points
    scatter = sns.scatterplot(x=ground_truth_avg_phrase_lengths, y=hdbscan_avg_phrase_lengths, 
                              hue=file_ids, 
                              size=sizes,
                              sizes=(min_size, max_size), palette='viridis', 
                              alpha=0.7, edgecolor='w', linewidth=0.5,
                              zorder=2)

    # Calculate and display Pearson correlation
    pearson = calculate_weighted_pearson(np.array(ground_truth_avg_phrase_lengths), np.array(hdbscan_avg_phrase_lengths), np.array(phrase_counts))
    plt.text(0.05, 0.95, f'Pearson r = {pearson:.3f}', 
             transform=plt.gca().transAxes, fontsize=18, 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Ground Truth Phrase Length', fontsize=24)
    plt.ylabel('HDBSCAN Average Phrase Length', fontsize=24)
    plt.title(f'Average Phrase Length Comparison\n(Smoothing Window: {smoothing_window})', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, max_phrase_length)
    plt.ylim(0, max_phrase_length)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    # Create legend with only Bird ID in bottom right
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:len(set(file_ids))], [f'Bird {id+1}' for id in range(len(set(file_ids)))], 
              title='Bird ID', fontsize=18, title_fontsize=18,
              loc='lower right')

    plt.savefig(f'phrase_length_correlation_{smoothing_window}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'phrase_length_correlation_{smoothing_window}.svg', format='svg', bbox_inches='tight')
    plt.close()

def process_multiple_window_sizes():
    window_sizes = list(range(0, 1000, 25))
    pearson_values_entropy = []
    pearson_values_phrase_length = []
    pearson_values_combined = []

    for window_size in tqdm(window_sizes, desc="Processing window sizes"):
        results = process_files(smoothing_window=window_size)
        ground_truth_entropies, hdbscan_entropies, ground_truth_avg_phrase_lengths, hdbscan_avg_phrase_lengths, file_ids, phrase_counts = results

        pearson_entropy = calculate_weighted_pearson(np.array(ground_truth_entropies), np.array(hdbscan_entropies), np.array(phrase_counts))
        pearson_phrase_length = calculate_weighted_pearson(np.array(ground_truth_avg_phrase_lengths), np.array(hdbscan_avg_phrase_lengths), np.array(phrase_counts))
        avg_pearson = (pearson_entropy + pearson_phrase_length) / 2
        pearson_values_entropy.append((window_size, pearson_entropy))
        pearson_values_phrase_length.append((window_size, pearson_phrase_length))
        pearson_values_combined.append((window_size, avg_pearson))

    return pearson_values_entropy, pearson_values_phrase_length, pearson_values_combined

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

def visualize_diagonalized_merged_mapping(normalized_matrix, unique_ground_truth, unique_hdbscan_labels, file_name, ground_truth_phrase_labels, hdbscan_labels, mapping):
    """Visualize the diagonalized normalized matrix with boxes showing merged ground truth labels."""
    # First get optimal arrangement using linear sum assignment
    cost_matrix = -normalized_matrix.T  # Transpose to align with ground truth to HDBSCAN mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reorder the matrix and labels (swap row_ind and col_ind)
    optimal_matrix = normalized_matrix[np.ix_(col_ind, row_ind)]
    optimal_ground_truth = unique_ground_truth[col_ind]
    optimal_hdbscan = unique_hdbscan_labels[row_ind]

    # Calculate frame error rate excluding silences and noise labels
    mapped_ground_truth = np.array([mapping[label] for label in ground_truth_phrase_labels])
    non_silence_mask = (mapped_ground_truth != -1) & (hdbscan_labels != -1)
    filtered_mapped_ground_truth = mapped_ground_truth[non_silence_mask]
    filtered_hdbscan = hdbscan_labels[non_silence_mask]

    frame_errors = np.sum(filtered_mapped_ground_truth != filtered_hdbscan)
    error_rate = (frame_errors / len(filtered_mapped_ground_truth)) * 100

    # Save mapped labels information to text file
    with open(f'mapped_labels_{file_name.replace(" ", "_").replace(":", "_")}.txt', 'w') as f:
        f.write(f"Frame Error Rate: {error_rate:.2f}%\n\n")
        f.write("Mapped Labels:\n")
        merged_groups = {}
        for gt_label, hdbscan_label in mapping.items():
            if hdbscan_label not in merged_groups:
                merged_groups[hdbscan_label] = []
            merged_groups[hdbscan_label].append(gt_label)
        for hdbscan_label, gt_labels_list in merged_groups.items():
            if hdbscan_label != -1:
                f.write(f"HDBSCAN Label {hdbscan_label} <- Ground Truth Labels {sorted(gt_labels_list)}\n")

    print(f"Frame Error Rate for {file_name}: {error_rate:.2f}%")

    plt.figure(figsize=(8, 8), dpi=300)
    ax = sns.heatmap(optimal_matrix, annot=False, cmap="viridis",
                     xticklabels=optimal_hdbscan,
                     yticklabels=optimal_ground_truth)

    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Normalized Shared Area', fontsize=14)

    # Add boxes of matching colors around ground truth labels merged into a single HDBSCAN label
    hdbscan_label_colors = {}
    color_palette = sns.color_palette("husl", len(optimal_hdbscan))  # Changed to husl for brighter colors
    
    for idx, hdbscan_label in enumerate(optimal_hdbscan):
        hdbscan_label_colors[hdbscan_label] = color_palette[idx]

    # Draw rectangles with increased linewidth
    for hdbscan_label in optimal_hdbscan:
        gt_labels_mapped = [gt_label for gt_label in optimal_ground_truth if mapping[gt_label] == hdbscan_label]
        if len(gt_labels_mapped) > 1:
            gt_indices = [list(optimal_ground_truth).index(gt_label) for gt_label in gt_labels_mapped]
            gt_indices.sort()
            
            groups = []
            current_group = [gt_indices[0]]
            
            for i in range(1, len(gt_indices)):
                if gt_indices[i] == gt_indices[i-1] + 1:
                    current_group.append(gt_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [gt_indices[i]]
            groups.append(current_group)
            
            j = list(optimal_hdbscan).index(hdbscan_label)
            for group in groups:
                i_min = min(group)
                i_max = max(group)
                rect = plt.Rectangle((j, i_min), 1, i_max - i_min + 1, 
                                   fill=False, 
                                   edgecolor=hdbscan_label_colors[hdbscan_label], 
                                   linewidth=3,  # Increased linewidth
                                   alpha=1.0)    # Full opacity
                ax.add_patch(rect)

    plt.xlabel('HDBSCAN Labels', fontsize=24)
    plt.ylabel('Ground Truth Phrase Labels', fontsize=24)
    plt.title(f'Diagonalized Mapping Matrix\nFrame Error Rate: {error_rate:.2f}%', fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig(f'diagonalized_mapping_matrix_{file_name.replace(" ", "_").replace(":", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'diagonalized_mapping_matrix_{file_name.replace(" ", "_").replace(":", "_")}.svg', 
                format='svg', bbox_inches='tight')
    plt.close()

# Files to process
files = [
    "/media/george-vengrovski/flash-drive/llb3_for_paper.npz",
    "/media/george-vengrovski/flash-drive/llb11_for_paper.npz",
    "/media/george-vengrovski/flash-drive/llb16_for_paper.npz"
]

# Calculate max values
max_entropy, max_phrase_length = calculate_max_values()

# Process files with multiple window sizes
pearson_values_entropy, pearson_values_phrase_length, pearson_values_combined = process_multiple_window_sizes()

# Plot Pearson correlation values
plt.figure(figsize=(8, 8))
window_sizes, pearson_entropy = zip(*pearson_values_entropy)
_, pearson_phrase_length = zip(*pearson_values_phrase_length)
_, pearson_combined = zip(*pearson_values_combined)

# Convert to numpy arrays for easier manipulation
window_sizes = np.array(window_sizes)
pearson_entropy = np.array(pearson_entropy)
pearson_phrase_length = np.array(pearson_phrase_length)
pearson_combined = np.array(pearson_combined)

# Create smooth line using more points for combined average only
x_smooth = np.linspace(min(window_sizes), max(window_sizes), 100)
y_combined_smooth = np.interp(x_smooth, window_sizes, pearson_combined)

# Plot points for all metrics
plt.scatter(window_sizes, pearson_entropy, label='Entropy', alpha=0.7, zorder=2, s=100)
plt.scatter(window_sizes, pearson_phrase_length, label='Phrase Duration', alpha=0.7, zorder=2, s=100)
plt.scatter(window_sizes, pearson_combined, label='Combined Average', alpha=0.7, zorder=2, s=100)

# Plot smoothed line only for combined average with dashed style
plt.plot(x_smooth, y_combined_smooth, alpha=0.5, zorder=1, linestyle='--')

plt.xlabel('Smoothing Window Size', fontsize=24)
plt.ylabel('Pearson Correlation Coefficient', fontsize=24)
plt.title('Pearson Correlation vs Window Sizes', fontsize=24)
plt.legend(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

plt.savefig('pearson_by_window_size.png', dpi=300)
plt.savefig('pearson_by_window_size.svg', format='svg')
plt.close()

print("Processing complete. Check the generated plots for results.")

# Find the best window size based on combined Pearson values
window_sizes, combined_pearson = zip(*pearson_values_combined)
best_window = window_sizes[np.argmax(combined_pearson)]
print(f"Best smoothing window size: {best_window}")

# Process files with the best window and create high-res correlation plots
results = process_files(smoothing_window=best_window)
plot_correlation_comparisons(results, best_window, max_entropy, max_phrase_length)

analyzer = SequenceAnalyzer()
for file in files:
    # Load data
    data = np.load(file)
    ground_truth_labels = data['ground_truth_labels']
    hdbscan_labels = data['hdbscan_labels']
    
    # Process labels using best window size
    ground_truth_phrase_labels = syllable_to_phrase_labels(ground_truth_labels, silence=0)

    # Fill in -1 labels in hdbscan_labels by smoothing with window_size=0
    hdbscan_labels_filled = smooth_labels(hdbscan_labels, window_size=0)

    # Create shared area matrix using filled HDBSCAN labels
    normalized_matrix, unique_ground_truth, unique_predicted = analyzer.create_shared_area_matrix(
        ground_truth_phrase_labels, hdbscan_labels_filled)

    # Use merged labels mapping (ground truth labels to HDBSCAN labels)
    mapping = create_merged_label_mapping(
        normalized_matrix, unique_ground_truth, unique_predicted, ground_truth_phrase_labels)

    # Map ground truth labels before smoothing
    mapped_ground_truth_phrase_labels = np.array([mapping[label] for label in ground_truth_phrase_labels])

    # Apply smoothing to the original HDBSCAN labels
    smoothed_hdbscan_labels = smooth_labels(hdbscan_labels, window_size=best_window)

    # Visualize the diagonalized mapping matrix
    visualize_diagonalized_merged_mapping(
        normalized_matrix,
        unique_ground_truth,
        unique_predicted,
        f"File: {file.split('/')[-1]}, Window: {best_window}",
        ground_truth_phrase_labels,
        hdbscan_labels,
        mapping
    )
