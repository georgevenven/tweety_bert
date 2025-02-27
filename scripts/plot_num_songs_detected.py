import json
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration variables
JSON_PATH = '/media/george-vengrovski/Desk SSD/TweetyBERT/song_detecton_database.json'
SINGLE_BIRD = "USA5508"  # Set to None for all birds, or a string like "USA5336" for a specific bird

class FileParser:
    def parse_date_time(self, file_path, format="standard"):
        """
        Parse the date and time from the file name.
        Args:
        - file_path (str): The file name.
        - format (str): The format of the file name. "standard" or "yarden".
        Returns:
        - tuple: (datetime object, file_base_name)
        """
        parts = file_path.split('_')
        # Remove .wav or .npz at the end of the last part
        parts[-1] = parts[-1].replace('.wav', '').replace('.npz', '')
        
        try:
            if format == "standard":
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = int(parts[6])
                file_date = datetime(2024, month, day, hour, minute, second)
                file_name = parts[0]
            else:
                print(f"Unknown format: {format}")
                return None, None
        except (ValueError, IndexError) as e:
            print(f"Invalid date format in file path: {file_path}\nError: {e}")
            return None, None
               
        return file_date, file_name

# Load the JSON data from a file
with open(JSON_PATH, 'r') as file:
    data = json.load(file)

# Initialize dictionaries to store the results
bird_stats = {}
time_data = {}  # Dictionary for temporal data
parser = FileParser()  # Create an instance of your class

if __name__ == "__main__":
    # Process each entry in the JSON data
    for entry in data:
        timestamp, bird_id = parser.parse_date_time(entry['filename'], format="standard")
        
        if timestamp is None or bird_id is None:
            continue
            
        # Skip if we're looking for a specific bird and this isn't it
        if SINGLE_BIRD and bird_id != SINGLE_BIRD:
            continue

        if bird_id not in bird_stats:
            bird_stats[bird_id] = {'num_songs': 0}
            time_data[bird_id] = {'timestamps': []}
        
        if entry['song_present']:
            num_segments = len(entry['segments'])
            bird_stats[bird_id]['num_songs'] += num_segments
            time_data[bird_id]['timestamps'].extend([timestamp] * num_segments)

    # Plotting logic
    if SINGLE_BIRD:
        plt.figure(figsize=(8, 6))  # Single plot for one bird
        timestamps = time_data[SINGLE_BIRD]['timestamps']
        if timestamps:
            min_date = min(timestamps)
            max_date = max(timestamps)
            total_days = max((max_date - min_date).days, 1)  # Ensure at least 1 day
            
            plt.hist(timestamps, 
                    bins=total_days,
                    alpha=0.5,
                    label=SINGLE_BIRD,
                    histtype='step',
                    linewidth=2)
            
            plt.title(f'Number of Songs per Day - {SINGLE_BIRD}')
            plt.xlabel('Time')
            plt.ylabel('Number of Songs')
            plt.legend(title='Bird ID')
            plt.gcf().autofmt_xdate()
    else:
        # Prepare data for plotting
        bird_ids = list(bird_stats.keys())
        num_songs = [stats['num_songs'] for stats in bird_stats.values()]

        # Update the figure layout
        plt.figure(figsize=(15, 6))

        # Plot the number of songs
        plt.subplot(1, 2, 1)
        plt.bar(bird_ids, num_songs, color='skyblue')
        plt.title('Number of Songs per Bird')
        plt.xlabel('Bird ID')
        plt.ylabel('Number of Songs')
        plt.xticks(rotation=90)

        # Plot songs per day
        plt.subplot(1, 2, 2)
        for bird_id in time_data:
            timestamps = time_data[bird_id]['timestamps']
            if not timestamps:  # Skip if no songs for this bird
                continue
                
            min_date = min(timestamps)
            max_date = max(timestamps)
            total_days = (max_date - min_date).days
            
            plt.hist(timestamps, 
                     bins=total_days,  # One bin per day
                     alpha=0.5,
                     label=bird_id,
                     histtype='step',
                     linewidth=2)

        plt.title('Number of Songs per Day')
        plt.xlabel('Time')
        plt.ylabel('Number of Songs')
        plt.legend(title='Bird ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()
