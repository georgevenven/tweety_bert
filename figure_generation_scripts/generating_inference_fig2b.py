"""
1) Train decoder with several songs 
2) Load npz file with those several songs

"""


import numpy as np
import matplotlib.pyplot as plt
import os
import umap
import matplotlib.colors as mcolors

# Load data
f = np.load("files/llb3_predictions_for_fig2b.npz")
predictions = f["predictions"]
spec = f["s"]
file_indices = f["file_indices"]

print(file_indices.shape)

# Replace the first file selection with longest file selection
unique_indices = np.unique(file_indices)
longest_file_idx = None
max_length = 0

for idx in unique_indices:
    current_length = np.sum(file_indices == idx)
    if current_length > max_length:
        max_length = current_length
        longest_file_idx = idx

# Get data for longest file
longest_file_mask = file_indices == longest_file_idx
predictions = predictions[longest_file_mask]
spec = spec[longest_file_mask]

print(predictions.shape)
print(spec.shape)

# Save full spectrogram first
plt.figure(figsize=(30, 6))
plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time Frame')
plt.ylabel('Frequency Bin')
plt.title('Full Spectrogram')
plt.savefig('full_spectrogram.svg', format='svg', bbox_inches='tight')
plt.close()

# Take first 3000 points and create 3 segments of 1000 each
n_samples = 3000
n_segments = 3
segment_size = 1000
predictions_subset = predictions[:n_samples]
spec_subset = spec[:n_samples]

# Create UMAP embedding
reducer = umap.UMAP(n_neighbors=200, min_dist=0, n_components=2, metric='cosine')
embedding = reducer.fit_transform(predictions_subset)

# Create segment-based colors
primary_colors = ['#FF0000', '#0000FF', '#008000']
colors = []
for i in range(n_segments):
    colors.extend([primary_colors[i]] * segment_size)

# Plot UMAP projection
plt.figure(figsize=(10, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, alpha=0.6)
plt.title('UMAP Projection', fontsize=14)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Make the plot square by setting equal limits
xlim = plt.xlim()
ylim = plt.ylim()
min_lim = min(xlim[0], ylim[0])
max_lim = max(xlim[1], ylim[1])
plt.xlim(min_lim, max_lim)
plt.ylim(min_lim, max_lim)

plt.gca().set_aspect('equal')
plt.savefig('umap_projection.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()

# Plot neural activations and spectrograms for each segment
for i in range(n_segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size
    
    # Neural activations
    plt.figure(figsize=(30, 6))
    plt.imshow(predictions_subset[start_idx:end_idx].T, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title(f'Neural Activations - Segment {i+1}')
    plt.savefig(f'neural_activations_segment_{i+1}.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    # Spectrograms
    plt.figure(figsize=(30, 6))
    plt.imshow(spec_subset[start_idx:end_idx].T, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title(f'Spectrogram - Segment {i+1}')
    plt.savefig(f'spectrogram_segment_{i+1}.svg', format='svg', bbox_inches='tight')
    plt.close()