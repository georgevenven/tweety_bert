import os
import json
import ast
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
from scipy.stats import chi2_contingency
import pandas as pd


class StateSwitchingAnalysis:
    def __init__(self, json_file, visualize=False, trial_date_str="2024-04-09 00:00:00", group_mode=None):
        self.trial_date_str = trial_date_str
        # Read the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)['results']
        
        # Convert to DataFrame
        data = pd.DataFrame(data)
        
        # Apply the parsing function to the relevant columns
        data['syllable_onsets_offsets_ms'] = data['syllable_onsets_offsets_ms'].apply(self.parse_json_safe)
        data['syllable_onsets_offsets_timebins'] = data['syllable_onsets_offsets_timebins'].apply(self.parse_json_safe)
        
        # Filter out rows where song_present is False
        initial_count = len(data)
        data = data[data['song_present']].reset_index(drop=True)
        filtered_count = len(data)
        
        # Parse date and file name, and extract labels
        data[['date_time', 'file_base_name']] = data['file_name'].apply(
            lambda x: pd.Series(self.parse_date_time(x, format="standard"))
        )
        
        # Check for any None values in date_time
        missing_dates = data['date_time'].isna().sum()
        if missing_dates > 0:
            print(f"Warning: {missing_dates} entries have invalid 'date_time' and will be excluded.")
            data = data.dropna(subset=['date_time']).reset_index(drop=True)
            print(f"Rows after removing entries with invalid 'date_time': {len(data)}")
        
        # Extract labels sequence from syllable_onsets_offsets_timebins
        data['labels'] = data['syllable_onsets_offsets_timebins'].apply(self.extract_labels_from_onsets)
                
        self.visualize = visualize
        self.file_names = data['file_name']
        self.song_present = data['song_present']
        self.syllable_onsets_offsets_ms = data['syllable_onsets_offsets_ms']
        self.syllable_onsets_offsets_timebins = data['syllable_onsets_offsets_timebins']
        self.date_times = data['date_time']
        self.file_base_names = data['file_base_name']
        self.labels = data['labels']
        
        # Set up results directory before any methods that use it
        base_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        json_base_name = os.path.splitext(os.path.basename(json_file))[0]
        self.results_dir = os.path.join(base_results_dir, f'decoder_analysis_{json_base_name}')
        os.makedirs(self.results_dir, exist_ok=True)        

        # Extract the base name from the JSON file path
        self.base_name = json_base_name
        
        # Create the song database
        self.database = self.create_song_database(data)
        if self.database is not None:
            if group_mode is not None:
                if group_mode == "time_of_day":
                    self.database = self.group_by_time_of_day(self.database)
                elif group_mode == "before_after_trial":
                    self.database = self.group_before_after_trial(self.database)
                else:
                    print(f"Unknown group mode: {group_mode}")
                    return
            self.database_to_csv(self.database, filename="temp.csv")  # Save CSV for debugging
        else:
            print("Error: Database creation failed.")
        
        # Initialize attributes
        self.transition_matrices = {}
        self.transition_matrices_norm = {}
        self.switching_times = {}
        self.transition_entropies = {}
        self.total_entropy = {}
        self.group_labels = {}  # New attribute to store labels for each group
        self.song_entropy = {}   # Stores entropy per song
        self.song_average_duration = {}  # Stores average duration per song
    
        # Initialize fixed_positions as None; it will be set after transition matrices are calculated
        self.fixed_positions = None

    def parse_json_safe(self, s):
        """
        Safely parse a string representation of a JSON object.
        Handles extra quotes and converts single quotes to double quotes.
        """
        if isinstance(s, dict):
            return s  # If it's already a dictionary, return it as is
        
        if pd.isna(s):
            return {}
        
        # Remove surrounding single quotes
        s = s.strip()
        if s.startswith("''") and s.endswith("''"):
            s = s[2:-2]
        elif s.startswith("'") and s.endswith("'"):
            s = s[1:-1]
        
        if not s:
            return {}
        
        try:
            # First, attempt to parse using json
            s_json = s.replace("'", '"')
            return json.loads(s_json)
        except json.JSONDecodeError:
            try:
                # If json fails, attempt using ast.literal_eval
                return ast.literal_eval(s)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing string: {s}\nError: {e}")
                return {}

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
        # Remove .wav at the end of the last part
        parts[-1] = parts[-1].replace('.wav', '')
        try:
            if format == "yarden":
                year = int(parts[2])
                month = int(parts[3])
                day = int(parts[4])
                hour = int(parts[5])
                minute = int(parts[6])
                second = int(parts[7])
                file_date = datetime(year, month, day, hour, minute, second)
                file_name = parts[0] + parts[1]
            elif format == "standard":
                # Example format adjustment as per user's requirement
                # Placeholder implementation:
                # Assuming parts: ['USA5288', '45355.32428022', '3', '4', '9', '0', '28']
                # parts[2] = month, parts[3] = day, parts[4] = hour, parts[5] = minute, parts[6] = second
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = int(parts[6])
                # Assuming year is 2024 as per the original date
                file_date = datetime(2024, month, day, hour, minute, second)
                file_name = parts[0]
            else:
                print(f"Unknown format: {format}")
                return None, None
        except (ValueError, IndexError) as e:
            print("parts:", *[f" {part}" for part in parts])
            print(f"Invalid date format in file path: {file_path}\nError: {e}")
            return None, None
               
        return file_date, file_name

    def extract_labels_from_onsets(self, syllable_dict):
        """
        Reconstruct a list of labels as one label per timebin in a giant array.

        Args:
        - syllable_dict (dict): Dictionary with syllable keys and their onset-offset times.

        Returns:
        - list: Array of labels, one per timebin.
        """
        if not syllable_dict:
            return []
        
        # Initialize an empty list to hold the labels for each timebin
        timebin_labels = []
        
        for label, intervals in syllable_dict.items():
            for interval in intervals:
                start_time = round(interval[0])
                end_time = round(interval[1])
                # Extend the list with the label repeated for each timebin in the interval
                timebin_labels.extend([int(label)] * (end_time - start_time))
        
        # Convert to numpy array for easier manipulation
        timebin_labels = np.array(timebin_labels)
        window_size = 149  # Number of timebins on each side
        total_window_size = 2 * window_size + 1
        
        # Initialize the majority labels list
        majority_labels = []
        
        # Initialize a dictionary to count occurrences of each label in the current window
        label_count = Counter(timebin_labels[:total_window_size])
        
        # Find the most common label in the initial window
        most_common_label = label_count.most_common(1)[0][0]
        majority_labels.append(most_common_label)
        
        # Slide the window across the timebin_labels
        for i in range(1, len(timebin_labels)):
            # Remove the label that is sliding out of the window
            if i - window_size - 1 >= 0:
                outgoing_label = timebin_labels[i - window_size - 1]
                label_count[outgoing_label] -= 1
                if label_count[outgoing_label] == 0:
                    del label_count[outgoing_label]
            
            # Add the label that is sliding into the window
            if i + window_size < len(timebin_labels):
                incoming_label = timebin_labels[i + window_size]
                label_count[incoming_label] += 1
            
            # Find the most common label in the current window
            most_common_label = label_count.most_common(1)[0][0]
            majority_labels.append(most_common_label)
        
        majority_labels = np.array(majority_labels)
        collapsed_labels = majority_labels[np.insert(np.diff(majority_labels) != 0, 0, True)]

        return collapsed_labels

    def create_song_database(self, data):
        """
        Create a song database DataFrame with necessary information.

        Args:
        - data (pd.DataFrame): The filtered and parsed DataFrame.

        Returns:
        - pd.DataFrame: The song database.
        """
        # Check required columns
        required_columns = ['file_name', 'song_present', 'syllable_onsets_offsets_ms',
                            'syllable_onsets_offsets_timebins', 'date_time', 'file_base_name', 'labels']
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing required column: {col}")
                return None
        
        # Select and rename columns as needed
        database = data[['file_name', 'date_time', 'file_base_name', 'labels', 'syllable_onsets_offsets_ms']].copy()
        
        return database

    def group_before_after_trial(self, db):
        """
        Group the database entries into 'before_trial' and 'after_trial' based on a trial date.

        Args:
        - db (pd.DataFrame): The song database.

        Returns:
        - pd.DataFrame: The grouped database with a new 'group_id' column.
        """
        # Convert trial_date string to datetime
        try:
            trial_date = datetime.strptime(self.trial_date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"Invalid trial_date format: {self.trial_date_str}\nError: {e}")
            return db
        
        # Create a new column 'group_id'
        db['group_id'] = db['date_time'].apply(
            lambda x: 'morning' if x < trial_date else 'afternoon'
        )
        
        # Debug: Count songs in each group
        group_counts = db['group_id'].value_counts()

        return db

    def group_by_time_of_day(self, db):
        """
        Group the database entries into 'morning' and 'afternoon' based on the time of day.

        Args:
        - db (pd.DataFrame): The song database.

        Returns:
        - pd.DataFrame: The grouped database with a new 'group_id' column.
        """
        # Create a new column 'group_id' based on the hour of the 'date_time'
        db['group_id'] = db['date_time'].apply(
            lambda x: 'morning' if 6 <= x.hour < 8 else ('afternoon' if 16 <= x.hour < 18 else 'other')
        )
        
        return db

    def database_to_csv(self, db, filename=None):
        """
        Save the database to a CSV file.

        Args:
        - db (pd.DataFrame): The song database.
        - filename (str, optional): The filename to save. Defaults to None.

        Returns:
        - None
        """
        if filename is None:
            filename = f"{self.base_name}_database.csv"
        filepath = os.path.join(self.results_dir, filename)
        db.to_csv(filepath, index=False)

    def calculate_transition_matrix(self, group):
        """
        Calculate the transition matrix for a specified group.

        Args:
        - group (str): The group identifier ('before_trial' or 'after_trial').

        Returns:
        - None
        """
        group_songs = self.database[self.database['group_id'] == group]['labels']
        
        if group_songs.empty:
            print(f"No songs found for group: {group}")
            return
        
        try:
            # Get all unique labels in the group
            group_labels = np.unique(np.concatenate(group_songs.values))
            print(f"Unique labels in group '{group}': {group_labels}")
        except Exception as e:
            print(f"Error concatenating labels for group {group}: {e}")
            return
        
        self.group_labels[group] = group_labels
        n_labels = len(group_labels)
        
        if n_labels == 0:
            print(f"No unique labels found for group: {group}")
            return
        
        transition_matrix = np.zeros((n_labels, n_labels))
        
        for song_idx, song in enumerate(group_songs, start=1):
            for i in range(len(song) - 1):
                from_label = song[i]
                to_label = song[i+1]
                if from_label != to_label:  # Exclude self-transitions
                    from_idx = np.where(group_labels == from_label)[0][0]
                    to_idx = np.where(group_labels == to_label)[0][0]
                    transition_matrix[from_idx, to_idx] += 1
            if song_idx % 1000 == 0:
                print(f"Processed {song_idx} songs for group '{group}'")
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix_norm = transition_matrix / row_sums[:, np.newaxis]
        transition_matrix_norm[transition_matrix_norm < 0.1] = 0  # Thresholding
        
        self.transition_matrices[group] = transition_matrix
        self.transition_matrices_norm[group] = transition_matrix_norm
        
    def create_fixed_positions(self):
        """
        Create fixed positions for nodes in the network graph.

        Returns:
        - dict: A dictionary with node positions.
        """
        # Aggregate all unique labels across all groups
        all_labels = set()
        for labels in self.group_labels.values():
            all_labels.update(labels)
        
        if not all_labels:
            print("No labels found to create fixed positions.")
            return {}
        
        G = nx.Graph()
        G.add_nodes_from(all_labels)
        fixed_pos = nx.spring_layout(G, k=0.5, iterations=50)
        return fixed_pos

    def create_transition_graph(self, group):
        """
        Create a transition graph for a specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - networkx.DiGraph: The transition graph.
        """
        if group not in self.group_labels:
            print(f"Group '{group}' not found in group_labels.")
            return None
        
        G = nx.DiGraph()
        for label in self.group_labels[group]:
            G.add_node(label)
        for i, from_label in enumerate(self.group_labels[group]):
            for j, to_label in enumerate(self.group_labels[group]):
                if i != j:  # Exclude self-edges
                    weight = self.transition_matrices_norm[group][i, j]
                    if weight > 0:
                        G.add_edge(from_label, to_label, weight=weight)
        
        return G

    def plot_transition_graph_and_matrix(self, group):
        """
        Plot the transition graph and transition matrix heatmap for a specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - None
        """
        print(f"Plotting transition graph and matrix for group: {group}")
        group_data = self.collect_statistics(group)
        if group_data is None:
            print(f"No data to plot for group: {group}")
            return
        
        G = self.create_transition_graph(group)
        if G is None:
            return
        
        # Ensure fixed_positions are created
        if self.fixed_positions is None:
            print("Creating fixed positions for network graph...")
            self.fixed_positions = self.create_fixed_positions()
            if not self.fixed_positions:
                print("Fixed positions could not be created. Skipping plotting.")
                return
        else:
            print("Using existing fixed positions for network graph.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # Plot graph
        node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        normalized_weights = [w / max_weight for w in edge_weights]
        edge_colors = plt.cm.YlOrRd(normalized_weights)

        try:
            nx.draw_networkx_nodes(G, self.fixed_positions, node_size=node_sizes, node_color='lightblue', ax=ax1)
            nx.draw_networkx_labels(G, self.fixed_positions, font_size=10, font_weight='bold', ax=ax1)
            nx.draw_networkx_edges(G, self.fixed_positions, width=[w * 3 for w in normalized_weights],
                                   edge_color=edge_colors, arrows=True,
                                   arrowsize=20, ax=ax1, connectionstyle="arc3,rad=0.1")
        except KeyError as e:
            print(f"Error drawing network nodes: {e}")
            return

        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=max_weight))
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, label='Transition Probability')

        ax1.set_title(f"Transition Graph for Group '{group}'\n(Excluding or Group '{group}'\n(Excluding Self-Transitions and Transitions < 0.1)", fontsize=16)
        ax1.axis('off')

        # Plot transition matrix as heatmap
        sns.heatmap(group_data["transition_matrix_norm"], cmap='YlGnBu', ax=ax2, 
                    xticklabels=group_data["unique_labels"], 
                    yticklabels=group_data["unique_labels"],
                    cbar_kws={'label': 'Transition Probability'})
        ax2.set_title(f"Transition Matrix for Group '{group}'", fontsize=16)
        ax2.set_xlabel("To State")
        ax2.set_ylabel("From State")

        plt.tight_layout()
        plot_filename = os.path.join(self.results_dir, f'transition_graph_matrix_group_{group}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    def calculate_switching_times(self, group):
        """
        Calculate the durations of each syllable in the group of songs.

        This function creates a dictionary where each unique syllable label is a key,
        and the value is a list containing the durations of that syllable across all songs.

        Args:
        - group (str): The group identifier.

        Returns:
        - None
        """
        switching_times_dict = {}
        group_songs = self.database[self.database['group_id'] == group]['syllable_onsets_offsets_ms']
        
        for song_idx, song_syllables in enumerate(group_songs, start=1):
            if not song_syllables:
                continue
            
            for label, intervals in song_syllables.items():
                for interval in intervals:
                    onset = interval[0]
                    offset = interval[1]
                    duration = offset - onset  # Duration in milliseconds
                    if label not in switching_times_dict:
                        switching_times_dict[label] = []
                    switching_times_dict[label].append(duration)
            
            if song_idx % 100 == 0:
                print(f"Processed {song_idx} songs for switching times in group '{group}'")
        
        self.switching_times[group] = switching_times_dict
        
    
    def plot_switching_times_histogram(self, group, all_durations_all_groups, xlim=None, ylim=None):
        """
        Plot a combined histogram of switching times for all syllables in the group.

        Args:
        - group (str): The group identifier.
        - all_durations_all_groups (list): List of all durations from all groups for consistent x-axis.
        - xlim (tuple, optional): The x-axis limits for the plot.
        - ylim (tuple, optional): The y-axis limits for the plot.

        Returns:
        - None
        """
        if group not in self.switching_times:
            print(f"No switching times found for group: {group}")
            return
        
        plt.figure(figsize=(14, 8))

        # Combine all durations from all syllables
        all_durations = [duration for durations in self.switching_times[group].values() for duration in durations]

        if not all_durations:
            print(f"No durations to plot for group: {group}")
            return

        plt.hist(all_durations, bins=50, color='blue', alpha=0.7, range=(min(all_durations_all_groups), max(all_durations_all_groups)))
        plt.title(f"Histogram of Switching Times for Group '{group}'")
        plt.xlabel("Duration (ms)")
        plt.ylabel("Frequency")
        plt.grid(True)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.tight_layout()
        plot_filename = os.path.join(self.results_dir, f'switching_times_histogram_group_{group}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        # Print statistics
        avg_duration = np.mean(all_durations)
        median_duration = np.median(all_durations)
        duration_range = (min(all_durations), max(all_durations))
        print(f"Group '{group}' - Average Duration: {avg_duration:.2f} ms, Median Duration: {median_duration:.2f} ms, Range: {duration_range}")

        print(f"Switching times histogram plotted and saved to {plot_filename}")

    def plot_switching_times_violin(self, group, max_duration):
        """
        Plot a violin plot with dot plots of the duration distributions for each syllable in the group,
        sorted in ascending order of the syllable labels. Overlay the standard deviation as a black "I" shape
        (horizontal bars for ± standard deviation) and display mean and std labels.

        Args:
        - group: str, the group identifier for which the violin plots are plotted.
        - max_duration: float, the maximum duration across all groups for consistent y-axis.

        Returns:
        - None
        """
        if group not in self.switching_times:
            print(f"No switching times found for group: {group}")
            return

        plt.figure(figsize=(24, 10))

        # Sort labels based on their numeric value
        sorted_labels = sorted(self.switching_times[group].keys(), key=int)
        
        # Sort data according to sorted labels
        data = [self.switching_times[group][label] for label in sorted_labels]
        labels = [f'Label {label}' for label in sorted_labels]

        # Plot the violin plot
        sns.violinplot(data=data, bw=0.2, cut=0, inner=None, orient='v', linewidth=1.2)

        # Overlay dot plots and standard deviation lines (as black "I" shaped bars)
        for i, durations in enumerate(data):
            if not durations:
                continue
            # Adding scatter plot on the same plot
            plt.scatter(np.random.normal(i, 0.04, size=len(durations)), durations, color='black', alpha=0.6, s=10)

            # Calculate statistics for std deviation lines
            mean = np.mean(durations)
            std_dev = np.std(durations)

            # Add standard deviation "I" line (black with horizontal bars at ends)
            plt.plot([i, i], [mean - std_dev, mean + std_dev], color='black', lw=2)  # Vertical line
            plt.plot([i - 0.1, i + 0.1], [mean - std_dev, mean - std_dev], color='black', lw=2)  # Lower horizontal bar
            plt.plot([i - 0.1, i + 0.1], [mean + std_dev, mean + std_dev], color='black', lw=2)  # Upper horizontal bar

            # Annotate the plot with mean and std deviation
            plt.text(i, max(durations)*1.05, f'Mean: {mean:.0f} ms\nStd: {std_dev:.0f} ms', 
                    ha='center', va='bottom', fontsize=10)

        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=16)
        plt.yticks(fontsize=16)  # Set y-axis ticks size to 16
        plt.ylim(0, max_duration * 1.1)  # Set y-axis limit for consistency
        plt.title(f'Distribution of Durations for Each Syllable in Group {group}', fontsize=24)
        plt.xlabel('Syllable', fontsize=24)
        plt.ylabel('Duration (ms)', fontsize=24)
        plt.tight_layout()
        plot_filename = os.path.join(self.results_dir, f'switching_times_violin_group_{group}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    def collect_statistics(self, group):
        """
        Collect statistics for a given group, including unique labels and transition matrices.

        Args:
        - group (str): The group identifier.

        Returns:
        - dict: A dictionary containing the unique labels and transition matrices.
        """
        group_songs = self.database[self.database['group_id'] == group]['labels']
        if group_songs.empty:
            print(f"No songs found for group: {group}")
            return None

        try:
            unique_labels = np.unique(np.concatenate(group_songs.values))
            print(f"Unique labels for statistics in group '{group}': {unique_labels}")
        except Exception as e:
            print(f"Error collecting unique labels for group {group}: {e}")
            return None

        transition_matrix = self.transition_matrices.get(group, None)
        transition_matrix_norm = self.transition_matrices_norm.get(group, None)

        if transition_matrix is None or transition_matrix_norm is None:
            print(f"Transition matrices not calculated for group: {group}")
            return None

        return {
            "unique_labels": unique_labels,
            "transition_matrix": transition_matrix,
            "transition_matrix_norm": transition_matrix_norm
        }

    def visualize_group_data(self):
        """
        Visualize the distribution of data across different groups, including number of songs,
        unique labels, and total transitions.

        Returns:
        - None
        """
        group_data = []
        for group in self.database['group_id'].unique():
            stats = self.collect_statistics(group)
            if stats is None:
                continue
            group_data.append({
                'group': group,
                'num_songs': int(len(self.database[self.database['group_id'] == group])),
                'num_labels': int(len(stats['unique_labels'])),
                'total_transitions': int(np.sum(self.transition_matrices.get(group, np.zeros_like(self.transition_matrices[group]))))
            })
        
        if not group_data:
            print("No group data available for visualization.")
            return

        groups = [d['group'] for d in group_data]
        num_songs = [d['num_songs'] for d in group_data]
        total_transitions = [d['total_transitions'] for d in group_data]
        num_labels = [d['num_labels'] for d in group_data]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        ax1.bar(groups, num_songs, color='skyblue')
        ax1.set_title('Number of Songs per Group')
        ax1.set_ylabel('Number of Songs')
        
        ax2.bar(groups, num_labels, color='lightgreen')
        ax2.set_title('Number of Unique Syllables per Group')
        ax2.set_ylabel('Number of Unique Syllables')
        
        ax3.bar(groups, total_transitions, color='salmon')
        ax3.set_title('Total Transitions per Group')
        ax3.set_xlabel('Group')
        ax3.set_ylabel('Number of Transitions')
        
        plt.tight_layout()
        plot_filename = os.path.join(self.results_dir, 'group_data_distribution.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()


    def calculate_average_durations(self, group):
        """
        Calculate the average duration of each syllable in the specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - dict: A dictionary with syllable labels as keys and their average durations as values.
        """
        if group not in self.switching_times:
            print(f"No switching times found for group: {group}")
            return {}
        
        average_durations = {}
        for label, durations in self.switching_times[group].items():
            if durations:
                average_durations[label] = np.mean(durations)
            else:
                average_durations[label] = 0.0
        
        for label, avg in average_durations.items():
            print(f"  Syllable {label}: {avg:.2f} ms")
        return average_durations

    def calculate_average_duration_per_song(self):
        """
        Calculate the average phrase duration for each song in the database.

        Returns:
        - None
        """
        for song_idx, song in enumerate(self.database.itertuples(), start=1):
            syllable_dict = song.syllable_onsets_offsets_ms
            durations = []
            
            for label, intervals in syllable_dict.items():
                for interval in intervals:
                    duration = interval[1] - interval[0]
                    durations.append(duration)
            
            if durations:
                avg_duration = np.mean(durations)
            else:
                avg_duration = 0.0
            
            self.song_average_duration[song.file_name] = avg_duration

            # Debug: Print average duration for each song
            # print(f"  Song {song_idx}: {song.file_name} - Average Duration: {avg_duration:.2f} ms")
        
        print("Average phrase durations calculated for each song.")

    def calculate_transition_entropy(self, group_by):
        """
        Calculate the total transition entropy based on a specified time grouping.

        Args:
        - group_by (str): The column name to group by ('date' or 'hour').

        Returns:
        - pd.DataFrame: A DataFrame with the group_by column and 'total_transition_entropy' column.
        """
        self.database[group_by] = self.database['date_time'].dt.__getattribute__(group_by)
        entropy_per_group = []

        for group_value, group in self.database.groupby(group_by):
            # Initialize counters for transitions and state occurrences
            transition_counts = Counter()
            state_counts = Counter()

            # Tally up transitions within each song
            for labels in group['labels']:
                for i in range(len(labels) - 1):
                    from_state = labels[i]
                    to_state = labels[i + 1]
                    if from_state != to_state:  # Exclude self-transitions
                        transition = (from_state, to_state)
                        transition_counts[transition] += 1
                        state_counts[from_state] += 1

            # Calculate branch entropy (H_a) for each state
            branch_entropies = {}
            for from_state in state_counts:
                entropy = 0.0
                for to_state in set(label for _, label in transition_counts if _ == from_state):
                    transition_probability = transition_counts[(from_state, to_state)] / state_counts[from_state]
                    entropy -= transition_probability * np.log(transition_probability)
                branch_entropies[from_state] = entropy

            # Calculate total transition entropy (TE)
            total_transitions = sum(state_counts.values())
            total_entropy = 0.0
            for state, count in state_counts.items():
                state_probability = count / total_transitions
                total_entropy += branch_entropies[state] * state_probability

            entropy_per_group.append({group_by: group_value, 'total_transition_entropy': total_entropy})

        return pd.DataFrame(entropy_per_group)

    def plot_summary_over_days(self, trial_date_str=None):
        """
        Plot total song entropy, total transition entropy, and songs sung across days on one canvas.
        Optionally draws a red vertical bar where the experiment occurred if a trial date is provided.

        Args:
        - trial_date_str (str, optional): The date of the experiment in "YYYY-MM-DD HH:MM:SS" format.

        Returns:
        - None
        """
        trial_date = None
        if trial_date_str:
            try:
                trial_date = datetime.strptime(trial_date_str, "%Y-%m-%d %H:%M:%S").date()
            except ValueError as e:
                print(f"Invalid trial_date format: {trial_date_str}\nError: {e}")
                return

        # Extract date from 'date_time' and add as a new column
        self.database['date'] = self.database['date_time'].dt.date

        # Calculate total transition entropy per day
        entropy_per_day = self.calculate_transition_entropy('date')

        # Count songs sung per day
        songs_per_day = self.database.groupby('date').size().reset_index(name='songs_sung')

        # Merge all metrics into a single DataFrame
        summary_df = entropy_per_day.merge(songs_per_day, on='date')

        # Sort by date
        summary_df = summary_df.sort_values('date')

        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=(20, 20), sharex=True)

        # Plot Average Entropy
        axes[0].plot(summary_df['date'], summary_df['total_transition_entropy'], 'o-', color='blue', label='Total Transition Entropy')
        axes[0].set_ylabel('Total Transition Entropy', color='blue', fontsize=18)
        axes[0].tick_params(axis='y', labelcolor='blue', labelsize=16)
        axes[0].set_title('Total Transition Entropy per Day', fontsize=20)
        if trial_date:
            axes[0].axvline(x=trial_date, color='red', linestyle='--', linewidth=2, label='Experiment Occurred')
        axes[0].legend(fontsize=16)

        # Plot Songs Sung
        axes[1].plot(summary_df['date'], summary_df['songs_sung'], 'o-', color='red', label='Songs Sung')
        axes[1].set_ylabel('Songs Sung', color='red', fontsize=18)
        axes[1].tick_params(axis='y', labelcolor='red', labelsize=16)
        axes[1].set_title('Songs Sung per Day', fontsize=20)
        axes[1].set_xlabel('Date', fontsize=18)
        if trial_date:
            axes[1].axvline(x=trial_date, color='red', linestyle='--', linewidth=2, label='Experiment Occurred')
        axes[1].legend(fontsize=16)

        # Improve x-axis formatting
        plt.xticks(rotation=45, fontsize=16)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(self.results_dir, 'summary_over_days.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Summary over days plotted and saved to {plot_filename}")

    def plot_summary_over_time_of_day(self):
        """
        Plot total transition entropy and songs sung across hours of the day on one canvas.

        Returns:
        - None
        """

        # Calculate total transition entropy per hour
        total_transition_entropy_per_hour = self.calculate_transition_entropy('hour')
        # Count songs sung per hour
        songs_per_hour = self.database.groupby('hour').size().reset_index(name='songs_sung')

        # Merge all metrics into a single DataFrame
        summary_df = total_transition_entropy_per_hour.merge(songs_per_hour, on='hour')

        # Sort by hour
        summary_df = summary_df.sort_values('hour')

        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

        # Plot Total Transition Entropy
        axes[0].plot(summary_df['hour'], summary_df['total_transition_entropy'], 'o-', color='green', label='Total Transition Entropy')
        axes[0].set_ylabel('Total Transition Entropy', color='green', fontsize=18)
        axes[0].tick_params(axis='y', labelcolor='green', labelsize=16)
        axes[0].set_title('Total Transition Entropy per Hour', fontsize=20)
        axes[0].legend(fontsize=16)

        # Plot Songs Sung
        axes[1].plot(summary_df['hour'], summary_df['songs_sung'], 'o-', color='red', label='Songs Sung')
        axes[1].set_ylabel('Songs Sung', color='red', fontsize=18)
        axes[1].tick_params(axis='y', labelcolor='red', labelsize=16)
        axes[1].set_title('Songs Sung per Hour', fontsize=20)
        axes[1].set_xlabel('Hour of Day', fontsize=18)
        axes[1].legend(fontsize=16)

        # Improve x-axis formatting
        plt.xticks(rotation=45, fontsize=16)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(self.results_dir, 'summary_over_time_of_day.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

    def calculate_and_plot_histograms(self, groups):
        """
        Calculate global x and y limits and plot histograms for each group.
 
        Args:
        - groups (list): List of group identifiers.
 
        Returns:
        - None
        """
        all_durations_all_groups = []
        max_frequency = 0
 
        # Collect all durations and determine max frequency
        for group in groups:
            if group in self.switching_times:
                all_durations = [duration for durations in self.switching_times[group].values() for duration in durations]
                all_durations_all_groups.extend(all_durations)
                # Calculate histogram to find max frequency
                hist, _ = np.histogram(all_durations, bins=50, range=(min(all_durations_all_groups), max(all_durations_all_groups)))
                max_frequency = max(max_frequency, max(hist))
 
        # Calculate global x and y limits
        xlim = (min(all_durations_all_groups), max(all_durations_all_groups))
        ylim = (0, max_frequency * 1.1)  # Add a little padding to the y-axis
 
        # Plot each group with the same limits
        for group in groups:
            self.plot_switching_times_histogram(group, all_durations_all_groups, xlim, ylim)

    def plot_combined_switching_times_histogram(self, groups):
        """
        Plot a combined histogram of switching times for all syllables in the groups.

        Args:
        - groups (list): List of group identifiers.

        Returns:
        - None
        """
        
        plt.figure(figsize=(14, 8))

        colors = ['red', 'blue']
        all_durations_all_groups = []

        # Collect all durations and determine x limits
        for group in groups:
            if group in self.switching_times:
                all_durations = [duration for durations in self.switching_times[group].values() for duration in durations]
                all_durations_all_groups.extend(all_durations)

        # Calculate global x limits
        xlim = (min(all_durations_all_groups), max(all_durations_all_groups))

        for idx, group in enumerate(groups):
            if group not in self.switching_times:
                print(f"No switching times found for group: {group}")
                continue

            # Combine all durations from all syllables
            all_durations = [duration for durations in self.switching_times[group].values() for duration in durations]

            if not all_durations:
                print(f"No durations to plot for group: {group}")
                continue

            plt.hist(all_durations, bins=50, color=colors[idx], alpha=0.5, range=xlim, density=True, label=f'Group {group}')

        plt.title("Combined Histogram of Switching Times for Groups")
        plt.xlabel("Duration (ms)")
        plt.ylabel("Probability Density")
        plt.grid(True)
        plt.xlim(xlim)
        plt.tight_layout()
        plt.legend()
        plot_filename = os.path.join(self.results_dir, 'combined_switching_times_histogram.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Combined switching times histogram plotted and saved to {plot_filename}")

    def plot_combined_switching_times_kde(self, groups):
        """
        Plot a combined KDE of switching times for all syllables in the groups.

        Args:
        - groups (list): List of group identifiers.

        Returns:
        - None
        """        
        plt.figure(figsize=(14, 8))

        colors = ['red', 'blue']
        all_durations_all_groups = []

        for idx, group in enumerate(groups):
            if group not in self.switching_times:
                print(f"No switching times found for group: {group}")
                continue

            # Combine all durations from all syllables
            all_durations = [duration for durations in self.switching_times[group].values() for duration in durations]

            if not all_durations:
                print(f"No durations to plot for group: {group}")
                continue

            # Plot KDE for the group
            sns.kdeplot(all_durations, color=colors[idx], label=f'Group {group}', fill=True, alpha=0.5)

        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plot_filename = os.path.join(self.results_dir, 'combined_switching_times_kde.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()

# Example usage:
# Ensure that this script is run in an environment where '__file__' is defined,
# such as a Python script, not in interactive environments like Jupyter notebooks.

if __name__ == "__main__":
    json_files_with_dates = [
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/llb3_songs_decoded_database.json', '2024-03-07 00:00:00'),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/llb11_songs_decoded_database.json', '2024-03-07 00:00:00'),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5177_variable_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5203_static_decoded_database.json', None),
        ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5336_decoded_database.json', None)
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5337_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5371_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5443_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20240930T185033Z-001/USA5271_decoded.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20240930T185033Z-001/USA5283_decoded.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5505_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5347_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5468_RC3_decoded_database.json', None),
        # ('/home/george-vengrovski/Downloads/drive-download-20241007T224715Z-001/USA5509_decoded_database.json', None)
    ]
    
    for json_file_path, trial_date_str in json_files_with_dates:
        analysis = StateSwitchingAnalysis(json_file_path, visualize=True, trial_date_str=trial_date_str, group_mode="time_of_day")
        analysis.calculate_average_duration_per_song()
        analysis.plot_summary_over_days(trial_date_str=trial_date_str)
        analysis.plot_summary_over_time_of_day()


#### OLD ####

        
        # # # Calculate transition matrices for both groups
        # # for group in ['before_trial', 'after_trial']:
            
        
        # #     # Now create fixed positions after all transition matrices are calculated
        # #     analysis.fixed_positions = analysis.create_fixed_positions()
            
        # #     # Proceed to plot only if fixed_positions were successfully created
        # #     if analysis.fixed_positions:
        # #         all_durations_all_groups = []
        # #         max_duration = 0
        # #         for group in ['before_trial', 'after_trial']:
        # #             analysis.calculate_switching_times(group)
        # #             analysis.calculate_entropy_per_song(group)
        # #             analysis.calculate_average_duration_per_song(group)
        # #             all_durations = [duration for durations in analysis.switching_times[group].values() for duration in durations]
        # #             all_durations_all_groups.extend(all_durations)
        # #             if all_durations:  # Ensure there are durations
        # #                 max_duration = max(max_duration, max(all_durations))
                
        # #         for group in ['before_trial', 'after_trial']:
        # #             analysis.plot_transition_graph_and_matrix(group)
        # #             analysis.plot_switching_times_histogram(group, all_durations_all_groups)
        # #             analysis.plot_switching_times_violin(group, max_duration)  # Updated Method
        # #             analysis.calculate_transition_entropy(group)
        # #             average_durations = analysis.calculate_average_durations(group)
        # #             print(f"Average Durations for Group '{group}': {average_durations}")
                
        # #         analysis.plot_summary_over_days(trial_date_str=trial_date_str)
        # #         analysis.visualize_group_data()
        # #     else:
        # #         print("Fixed positions were not created. Skipping plotting.")
        
        # # Integrate with time of day
        # for time_group in ['morning', 'afternoon']:

        #     all_durations = []
        #     max_duration = 0
        #     analysis.calculate_transition_matrix(time_group)
        #     analysis.fixed_positions = analysis.create_fixed_positions()

        #     time_grouped_db = analysis.group_by_time_of_day(analysis.database)
        #     analysis.database = time_grouped_db  # Update the database with time groupings
            
        #     time_group_id = f"{time_group}"

        #     analysis.calculate_switching_times(time_group_id)
        #     analysis.calculate_entropy_per_song(time_group_id)
        #     analysis.calculate_average_duration_per_song(time_group_id)
        #     all_durations = [duration for durations in analysis.switching_times[time_group_id].values() for duration in durations]
        #     all_durations.extend(all_durations)

        #     if all_durations:  # Ensure there are durations
        #         max_duration = max(max_duration, max(all_durations))

        #     analysis.calculate_transition_matrix(time_group_id)
        #     analysis.calculate_entropy_per_song(time_group_id)
        #     analysis.calculate_average_duration_per_song(time_group_id)
            
        #     analysis.plot_transition_graph_and_matrix(time_group_id)
        #     analysis.plot_switching_times_histogram(time_group_id, all_durations)
        #     analysis.plot_switching_times_violin(time_group_id, max_duration)
        #     # analysis.calculate_transition_entropy(time_group_id)
        #     average_durations = analysis.calculate_average_durations(time_group_id)

        #     print(f"Average Durations for Group '{time_group_id}': {average_durations}")
            
        #     analysis.plot_summary_over_time_of_day()
        #     analysis.visualize_group_data()

        # analysis.plot_summary_over_days(trial_date_str=None)

        # analysis.plot_combined_switching_times_histogram(['morning', 'afternoon'])
