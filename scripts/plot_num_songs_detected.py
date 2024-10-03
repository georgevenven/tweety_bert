import json
import matplotlib.pyplot as plt

# Load the JSON data from a file
with open('/media/george-vengrovski/flash-drive/new_season/onset_offset_results.json', 'r') as file:
    data = json.load(file)

# Initialize a dictionary to store the results
bird_stats = {}
song_lengths = {}

# Process each entry in the JSON data
for entry in data:
    bird_id = entry['filename'].split('_')[0]
    if bird_id not in bird_stats:
        bird_stats[bird_id] = {'num_songs': 0, 'total_duration_ms': 0}
        song_lengths[bird_id] = []
    
    if entry['song_present']:
        bird_stats[bird_id]['num_songs'] += 1
        for segment in entry['segments']:
            duration = segment['offset_ms'] - segment['onset_ms']
            bird_stats[bird_id]['total_duration_ms'] += duration
            song_lengths[bird_id].append(duration / 1000)  # Convert to seconds

# Prepare data for plotting
bird_ids = list(bird_stats.keys())
num_songs = [stats['num_songs'] for stats in bird_stats.values()]
total_duration_s = [stats['total_duration_ms'] / 1000 for stats in bird_stats.values()]

# Plot the number of songs
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.bar(bird_ids, num_songs, color='skyblue')
plt.title('Number of Songs per Bird')
plt.xlabel('Bird ID')
plt.ylabel('Number of Songs')
plt.xticks(rotation=90)

# Plot the total duration
plt.subplot(1, 3, 2)
plt.bar(bird_ids, total_duration_s, color='lightgreen')
plt.title('Total Duration of Songs per Bird (s)')
plt.xlabel('Bird ID')
plt.ylabel('Total Duration (s)')
plt.xticks(rotation=90)

# Plot song length distributions
plt.subplot(1, 3, 3)
for bird_id, lengths in song_lengths.items():
    plt.hist(lengths, bins=30, alpha=0.5, label=bird_id, histtype='step', linewidth=2)

plt.title('Song Length Distributions')
plt.xlabel('Song Length (s)')
plt.ylabel('Frequency')
plt.legend(title='Bird ID', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
