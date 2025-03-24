import numpy as np
import matplotlib.pyplot as plt

# Load the spectrogram data from npz file
data = np.load('temp/umap_files/1741203573-USA5483_segment_10.npz')
spectrogram = data['s']  # Access the 's' array

# Crop the spectrogram as specified
spectrogram = spectrogram[10:216, :]

# Create the figure and plot without colorbar
plt.figure(figsize=(30, 6))
plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
plt.xlabel('Time Frame')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram')

# Save the figure as SVG
plt.savefig('spectrogram.svg', format='svg', bbox_inches='tight')
plt.close()
