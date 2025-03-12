import json
import numpy as np
import matplotlib.pyplot as plt

# Load training loss from JSON file
with open('/media/george-vengrovski/Desk SSD/TweetyBERT/models/TweetyBERT_Paper_Yarden_Model/training_statistics.json', 'r') as file:
    data = json.load(file)

training_loss = data['training_loss']
validation_loss = data['validation_loss']
steps = np.arange(len(training_loss)) / 1000  # Convert to thousands

# Define simple moving average function
def moving_average(values, window):
    box = np.ones(window)/window
    return np.convolve(values, box, mode='valid')

# Calculate moving average (window=1000 steps)
window = 1000
training_loss_ma = moving_average(training_loss, window)
validation_loss_ma = moving_average(validation_loss, window)
steps_ma = steps[window - 1:]

# Plotting
plt.figure(figsize=(24, 12))

# Plot moving averages only
plt.plot(steps_ma, training_loss_ma, label='Training Loss', linewidth=4, color='blue')
plt.plot(steps_ma, validation_loss_ma, label='Validation Loss', linewidth=4, color='red')

# Plot formatting for publication with increased font sizes (1.25x)
plt.xlabel('Training Steps (Thousands)', fontsize=34)  # 27 * 1.25 ≈ 34
plt.ylabel('Loss (MSE)', fontsize=34)  # 27 * 1.25 ≈ 34
plt.xticks(np.arange(0, max(steps_ma)+1, 5), fontsize=30)  # 24 * 1.25 = 30
plt.yticks(fontsize=30)  # 24 * 1.25 = 30
# Grid removed
plt.legend(fontsize=30)  # 24 * 1.25 = 30
plt.tight_layout()

# Set x-axis limits to match data
plt.xlim(0, max(steps_ma))

# Save or display
plt.savefig('yarden_training_loss_plot.png', dpi=300) 