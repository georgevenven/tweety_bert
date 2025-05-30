import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def syllable_to_phrase_labels(arr, silence=-1):
    """
    convert a sequence of syllable labels into a sequence of phrase labels,
    merging silence bins with their nearest adjacent syllables.

    for each contiguous block of silence:
    - if it's bounded by the same label on both sides, assign that label to all.
    - if it's bounded by two different labels, assign each time-bin to the closer label;
      ties go to the left.
    - if it's at the beginning or end (missing one side), assign to the available side.

    parameters
    ----------
    arr : np.ndarray
        array of integer labels, where `silence` frames are indicated by `silence`.
    silence : int, optional
        integer value representing silence (default: -1).

    returns
    -------
    np.ndarray
        array of phrase-level labels with silence frames appropriately merged.
    """
    new_arr = np.array(arr, dtype=int)
    length = len(new_arr)
    if length == 0:
        return new_arr  # edge case: empty input

    def find_silence_runs(labels):
        runs = []
        in_silence = False
        start = None
        for i, val in enumerate(labels):
            if val == silence and not in_silence:
                in_silence = True
                start = i
            elif val != silence and in_silence:
                runs.append((start, i - 1))
                in_silence = False
        if in_silence:
            runs.append((start, length - 1))
        return runs

    silence_runs = find_silence_runs(new_arr)
    for start_idx, end_idx in silence_runs:
        left_label = new_arr[start_idx - 1] if start_idx > 0 else None
        right_label = new_arr[end_idx + 1] if end_idx < length - 1 else None

        if left_label is None and right_label is None:
            continue  # entire array is silence; no change
        elif left_label is None:
            new_arr[start_idx:end_idx+1] = right_label
        elif right_label is None:
            new_arr[start_idx:end_idx+1] = left_label
        elif left_label == right_label:
            new_arr[start_idx:end_idx+1] = left_label
        else:
            # assign each bin to the nearer label; ties go left
            for i in range(start_idx, end_idx + 1):
                dist_left = i - (start_idx - 1)
                dist_right = (end_idx + 1) - i
                if dist_left <= dist_right:
                    new_arr[i] = left_label
                else:
                    new_arr[i] = right_label
    return new_arr

# Get the project root directory (tweety_bert_paper)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# paths and file selection
data_folder = "/media/george-vengrovski/George-SSD/llb_stuff/llb3_test"
npz_files = [f for f in os.listdir(data_folder) if f.endswith('.npz')]
num_samples = min(500, len(npz_files))
sample_files = random.sample(npz_files, num_samples)

# output folder: create in project_root/imgs/verifying_ground_truth_labels
out_dir = os.path.join(project_root, "imgs", "verifying_ground_truth_labels")
print(f"Attempting to create directory at: {os.path.abspath(out_dir)}")
os.makedirs(out_dir, exist_ok=True)

# Create a consistent color map for all spectrograms
# Using a combination of tab10, tab20, and tab20b for more distinct colors
cmap1 = plt.get_cmap('tab10')
cmap2 = plt.get_cmap('tab20')
cmap3 = plt.get_cmap('tab20b')

# Create a fixed color map for labels 0-30
fixed_colors = []
# First 10 colors from tab10
fixed_colors.extend([cmap1(i) for i in range(10)])
# Next 10 colors from tab20 (skipping duplicates with tab10)
fixed_colors.extend([cmap2(i) for i in range(10, 20)])
# Additional colors from tab20b if needed
fixed_colors.extend([cmap3(i) for i in range(10)])

# Make sure we have at least 31 colors (for labels 0-30)
while len(fixed_colors) < 31:
    fixed_colors.append((random.random(), random.random(), random.random(), 1))

# Make silence black (label 0)
fixed_colors[0] = (0, 0, 0, 1)  # Black color for silence

# Debug: Print number of files to process
print(f"Found {len(npz_files)} total files")
print(f"Processing {num_samples} sample files")

for file in tqdm(sample_files, desc="Processing files"):
    file_path = os.path.join(data_folder, file)
    # Debug: Print file being processed
    print(f"\nProcessing: {file}")
    data = np.load(file_path, allow_pickle=True)
    spectrogram = data['s']
    labels = data['labels']

    # obtain phrase labels via the provided function
    phrase_labels = syllable_to_phrase_labels(labels, silence=0)

    # create color arrays for both label versions using the fixed color map
    orig_colors = [fixed_colors[min(label, 30)] for label in labels]
    phrase_colors = [fixed_colors[min(label, 30)] for label in phrase_labels]

    # create a figure with three vertically stacked subplots using gridspec for custom heights
    fig = plt.figure(figsize=(18, 9))  # Increased height by 1 inch (8→9)
    gs = fig.add_gridspec(3, 1, height_ratios=[8, 0.67, 0.67])  # Reduced label bar heights by ~1/3
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # spectrogram
    axes[0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_ylabel("Frequency (bins)", fontsize=24)
    axes[0].set_title("Spectrogram", fontsize=28, pad=10)  # Removed filename from title
    axes[0].set_xticks([])  # Remove x-axis ticks from spectrogram

    # original labels row
    axes[1].imshow([orig_colors], aspect='auto')
    axes[1].set_yticks([])
    axes[1].set_title("Original Syllable Labels", fontsize=24)
    axes[1].set_xticks([])  # Remove x-axis ticks from original labels

    # phrase labels row
    axes[2].imshow([phrase_colors], aspect='auto')
    axes[2].set_yticks([])
    axes[2].set_xlabel("Time (bins)", fontsize=24)
    axes[2].set_title("Converted Phrase Labels", fontsize=24)

    # Increase tick label sizes
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=20)

    # Ensure all subplots share the same x-axis range
    for i in range(1, 3):
        axes[i].set_xlim(axes[0].get_xlim())

    plt.tight_layout()

    # save the figure
    base_name = os.path.splitext(file)[0]
    out_path = os.path.join(out_dir, f"{base_name}.png")
    print(f"Saving figure to: {os.path.abspath(out_path)}")
    plt.savefig(out_path)
    plt.close(fig)

print("Processing complete.")
