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
    def __init__(self, csv_file, visualize=False, trial_date_str="2024-04-09 00:00:00"):
        self.trial_date_str = trial_date_str
        # Read the CSV file
        print(f"Reading CSV file: {csv_file}")
        data = pd.read_csv(csv_file)
        print(f"Total rows in CSV: {len(data)}")
        
        # Apply the parsing function to the relevant columns
        print("Parsing 'syllable_onsets_offsets_ms' and 'syllable_onsets_offsets_timebins' columns...")
        data['syllable_onsets_offsets_ms'] = data['syllable_onsets_offsets_ms'].apply(self.parse_json_safe)
        data['syllable_onsets_offsets_timebins'] = data['syllable_onsets_offsets_timebins'].apply(self.parse_json_safe)
        
        # Filter out rows where song_present is False
        initial_count = len(data)
        data = data[data['song_present']].reset_index(drop=True)
        filtered_count = len(data)
        print(f"Filtered out {initial_count - filtered_count} rows where 'song_present' is False.")
        
        # Parse date and file name, and extract labels
        print("Parsing 'date_time' and 'file_base_name' from 'file_name'...")
        data[['date_time', 'file_base_name']] = data['file_name'].apply(
            lambda x: pd.Series(self.parse_date_time(x, format="standard"))
        )
        
        # Check for any None values in date_time
        missing_dates = data['date_time'].isna().sum()
        if missing_dates > 0:
            print(f"Warning: {missing_dates} entries have invalid 'date_time' and will be excluded.")
            data = data.dropna(subset=['date_time']).reset_index(drop=True)
            print(f"Rows after removing entries with invalid 'date_time': {len(data)}")
        
        # Extract labels sequence from syllable_onsets_offsets_ms
        print("Extracting labels sequence from 'syllable_onsets_offsets_ms'...")
        data['labels'] = data['syllable_onsets_offsets_ms'].apply(self.extract_labels_from_onsets)
        
        print("Filtered and Parsed Data Head:")
        print(data.head())
        
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
        csv_base_name = os.path.splitext(os.path.basename(csv_file))[0]
        self.results_dir = os.path.join(base_results_dir, f'decoder_analysis_{csv_base_name}')
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results directory set at: {self.results_dir}")
        
        # Set up image directory for debugging visualizations
        self.img_dir = os.path.join(self.results_dir, "imgs/vis_smoothed")
        os.makedirs(self.img_dir, exist_ok=True)  # Create the directory if it doesn't exist
        print(f"Image directory set at: {self.img_dir}")
        
        # Extract the base name from the CSV file path
        self.base_name = csv_base_name
        print(f"Base name extracted: {self.base_name}")
        
        # Create the song database
        print("Creating song database...")
        self.database = self.create_song_database(data)
        if self.database is not None:
            print(f"Total songs in database: {len(self.database)}")
            self.database = self.group_before_after_trial(self.database)
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
                # Example format: USA5288_45355.32428022_3_4_9_0_28.wav
                # Assuming parts: ['USA5288', '45355.32428022', '3', '4', '9', '0', '28']
                # Mapping to year, month, day, hour, minute, second
                # Adjust as per actual format
                # Placeholder example:
                year = 2024  # Define how to extract the year if it's part of the filename
                month = int(parts[2])
                day = int(parts[3])
                hour = int(parts[4])
                minute = int(parts[5])
                second = int(parts[6])
                file_date = datetime(year, month, day, hour, minute, second)
                file_name = parts[0]
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
        Extract a sorted list of labels based on syllable onset times.

        Args:
        - syllable_dict (dict): Dictionary with syllable keys and their onset-offset times.

        Returns:
        - list: Ordered list of labels based on onset times.
        """
        if not syllable_dict:
            return []
        
        # Create a list of tuples (onset_time, label)
        onset_label_list = []
        for label, intervals in syllable_dict.items():
            for interval in intervals:
                onset_time = interval[0]
                onset_label_list.append((onset_time, label))
        
        # Sort the list based on onset_time
        onset_label_list.sort(key=lambda x: x[0])
        
        # Extract the labels in order
        sorted_labels = [int(label) for onset, label in onset_label_list]
        
        return sorted_labels

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
        
        # Additional processing can be done here if needed
        print(f"Created song database with {len(database)} songs.")
        
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
            lambda x: 'before_trial' if x < trial_date else 'after_trial'
        )
        
        # Debug: Count songs in each group
        group_counts = db['group_id'].value_counts()
        print("Grouping Results:")
        for group, count in group_counts.items():
            print(f"  {group}: {count} songs")
        
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
        print(f"Database saved to {filepath}")

    def calculate_transition_matrix(self, group):
        """
        Calculate the transition matrix for a specified group.

        Args:
        - group (str): The group identifier ('before_trial' or 'after_trial').

        Returns:
        - None
        """
        print(f"Calculating transition matrix for group: {group}")
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
        print(f"Number of unique labels in group '{group}': {n_labels}")
        
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
        
        # Debug: Print transition matrix shape and a snippet
        print(f"Transition matrix for group '{group}':\nShape: {transition_matrix.shape}")
        print(f"Transition matrix snippet:\n{transition_matrix[:5, :5]}")
        print(f"Normalized transition matrix snippet:\n{transition_matrix_norm[:5, :5]}")
        
        print(f"Transition matrices calculated for group: {group}")

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
        print("Fixed positions for network graph created.")
        return fixed_pos

    def create_transition_graph(self, group):
        """
        Create a transition graph for a specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - networkx.DiGraph: The transition graph.
        """
        print(f"Creating transition graph for group: {group}")
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
        
        print(f"Transition graph for group '{group}' created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
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
        print(f"Transition graph and matrix plotted and saved to {plot_filename}")

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
        print(f"Calculating switching times for group: {group}")
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
        
        # Debug: Print number of durations per syllable
        print(f"Switching times calculated for group: {group}")
        for label, durations in switching_times_dict.items():
            print(f"  Syllable {label}: {len(durations)} durations recorded.")
    
    def plot_switching_times_histogram(self, group, all_durations_all_groups):
        """
        Plot a combined histogram of switching times for all syllables in the group.

        Args:
        - group (str): The group identifier.
        - all_durations_all_groups (list): List of all durations from all groups for consistent x-axis.

        Returns:
        - None
        """
        print(f"Plotting switching times histogram for group: {group}")
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
        print(f"Plotting switching times violin plot for group: {group}")
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
        print(f"Switching times violin plot plotted and saved to {plot_filename}")

    def calculate_transition_entropy(self, group):
        """
        Calculate the entropy of transitions for each syllable in the specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - None
        """
        print(f"Calculating transition entropy for group: {group}")
        if group not in self.transition_matrices_norm:
            print(f"No transition matrix found for group: {group}")
            self.transition_entropies[group] = {}
            self.total_entropy[group] = 0
            return

        # Initialize total entropy for the group
        self.transition_entropies[group] = {}
        self.total_entropy[group] = 0

        group_labels = self.group_labels[group]
        transition_matrix_norm = self.transition_matrices_norm[group]

        for i, state in enumerate(group_labels):
            probabilities = transition_matrix_norm[i]
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
            if probabilities.size > 0:
                entropy = -np.sum(probabilities * np.log2(probabilities))
            else:
                entropy = 0
            self.transition_entropies[group][state] = entropy

            # Calculate state frequency for weighting
            state_count = np.sum(self.transition_matrices[group][i])
            self.total_entropy[group] += entropy * state_count

        print(f"Transition entropies calculated for group: {group}")
        print(f"Total entropy for group '{group}': {self.total_entropy[group]:.4f}")

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
        print("Visualizing group data distribution...")
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

        print(f"Group data distribution visualization saved to {plot_filename}")

    def calculate_average_durations(self, group):
        """
        Calculate the average duration of each syllable in the specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - dict: A dictionary with syllable labels as keys and their average durations as values.
        """
        print(f"Calculating average durations for group: {group}")
        if group not in self.switching_times:
            print(f"No switching times found for group: {group}")
            return {}
        
        average_durations = {}
        for label, durations in self.switching_times[group].items():
            if durations:
                average_durations[label] = np.mean(durations)
            else:
                average_durations[label] = 0.0
        
        print(f"Average durations calculated for group: {group}")
        for label, avg in average_durations.items():
            print(f"  Syllable {label}: {avg:.2f} ms")
        return average_durations

    def calculate_entropy_per_song(self, group):
        """
        Calculate the entropy of transitions for each song in the specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - None
        """
        print(f"Calculating entropy per song for group: {group}")
        group_songs = self.database[self.database['group_id'] == group]
        
        for song_idx, song in enumerate(group_songs.itertuples(), start=1):
            labels = song.labels
            transitions = [(labels[i], labels[i+1]) for i in range(len(labels)-1) if labels[i] != labels[i+1]]
            
            if not transitions:
                entropy = 0
            else:
                transition_counts = Counter(transitions)
                total_transitions = sum(transition_counts.values())
                transition_probs = {k: v / total_transitions for k, v in transition_counts.items()}
                entropy = -sum(p * np.log2(p) for p in transition_probs.values() if p > 0)
            
            self.song_entropy[song.file_name] = entropy

            # Debug: Print entropy for each song
            #print(f"  Song {song_idx}: {song.file_name} - Entropy: {entropy:.4f}")
        
        print(f"Entropy calculated for each song in group: {group}")

    def calculate_average_duration_per_song(self, group):
        """
        Calculate the average phrase duration for each song in the specified group.

        Args:
        - group (str): The group identifier.

        Returns:
        - None
        """
        print(f"Calculating average phrase durations per song for group: {group}")
        group_songs = self.database[self.database['group_id'] == group]
        
        for song_idx, song in enumerate(group_songs.itertuples(), start=1):
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
        
        print(f"Average phrase durations calculated for each song in group: {group}")


    def plot_summary_over_days(self, trial_date_str="2024-04-08 00:00:00"):
        """
        Plot total song entropy, average phrase duration, and songs sung across days on one canvas.
        Draws a red vertical bar where the experiment occurred.

        Args:
        - trial_date_str (str): The date of the experiment in "YYYY-MM-DD HH:MM:SS" format.

        Returns:
        - None
        """
        print(f"Plotting summary over days with trial date: {trial_date_str}")
        # Convert trial_date_str to datetime
        try:
            trial_date = datetime.strptime(trial_date_str, "%Y-%m-%d %H:%M:%S").date()
        except ValueError as e:
            print(f"Invalid trial_date format: {trial_date_str}\nError: {e}")
            return

        # Extract date from 'date_time' and add as a new column
        self.database['date'] = self.database['date_time'].dt.date

        # Aggregate total entropy per day
        entropy_per_day = self.database.groupby('date')['file_name'].apply(
            lambda names: np.mean([self.song_entropy.get(name, 0) for name in names])
        ).reset_index(name='average_entropy')
        print("Average entropy per day:")
        print(entropy_per_day.head())

        # Aggregate average phrase duration per day
        avg_duration_per_day = self.database.groupby('date')['file_name'].apply(
            lambda names: np.mean([self.song_average_duration.get(name, 0) for name in names]) if len(names) > 0 else 0
        ).reset_index(name='average_phrase_duration')
        print("Average phrase duration per day:")
        print(avg_duration_per_day.head())

        # Count songs sung per day
        songs_per_day = self.database.groupby('date').size().reset_index(name='songs_sung')
        print("Songs sung per day:")
        print(songs_per_day.head())

        # Merge all metrics into a single DataFrame
        summary_df = entropy_per_day.merge(avg_duration_per_day, on='date').merge(songs_per_day, on='date')

        # Sort by date
        summary_df = summary_df.sort_values('date')

        # Debug: Print the merged summary DataFrame
        print("Merged summary DataFrame:")
        print(summary_df.head())

        # Create the plot
        fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

        # Plot Average Entropy
        axes[0].plot(summary_df['date'], summary_df['average_entropy'], 'o-', color='blue', label='Average Entropy')
        axes[0].set_ylabel('Average Entropy', color='blue', fontsize=14)
        axes[0].tick_params(axis='y', labelcolor='blue')
        axes[0].set_title('Average Entropy per Day', fontsize=16)
        axes[0].axvline(x=trial_date, color='red', linestyle='--', linewidth=2, label='Experiment Occurred')
        axes[0].legend()

        # Plot Average Phrase Duration
        axes[1].plot(summary_df['date'], summary_df['average_phrase_duration'], 'o-', color='green', label='Average Phrase Duration (ms)')
        axes[1].set_ylabel('Average Phrase Duration (ms)', color='green', fontsize=14)
        axes[1].tick_params(axis='y', labelcolor='green')
        axes[1].set_title('Average Phrase Duration per Day', fontsize=16)
        axes[1].axvline(x=trial_date, color='red', linestyle='--', linewidth=2, label='Experiment Occurred')
        axes[1].legend()

        # Plot Songs Sung
        axes[2].plot(summary_df['date'], summary_df['songs_sung'], 'o-', color='red', label='Songs Sung')
        axes[2].set_ylabel('Songs Sung', color='red', fontsize=14)
        axes[2].tick_params(axis='y', labelcolor='red')
        axes[2].set_title('Songs Sung per Day', fontsize=16)
        axes[2].set_xlabel('Date', fontsize=14)
        axes[2].axvline(x=trial_date, color='red', linestyle='--', linewidth=2, label='Experiment Occurred')
        axes[2].legend()

        # Improve x-axis formatting
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(self.results_dir, 'summary_over_days.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Summary over days plotted and saved to {plot_filename}")

# Example usage:
# Ensure that this script is run in an environment where '__file__' is defined,
# such as a Python script, not in interactive environments like Jupyter notebooks.

if __name__ == "__main__":
    csv_files_with_dates = [
        ('USA5468_RC3.csv', '2024-06-28 00:00:00'),
        ('USA5347.csv', '2024-01-20 00:00:00'),
        ('USA5337.csv', '2024-04-10 00:00:00'),
        ('USA5336.csv', '2024-01-23 00:00:00'),
        ('USA5505.csv', '2024-07-02 00:00:00'),
        ('USA5326.csv', '2024-02-20 00:00:00'),
        ('USA5325.csv', '2024-02-13 00:00:00'),
        ('USA5443.csv', '2024-04-30 00:00:00'),
        ('5371_decoded.csv', '2024-04-16 00:00:00'),
        ('5509_decoded.csv', '2024-05-01 00:00:00'),
        ('5288_decoded.csv', '2024-04-09 00:00:00')
    ]
    
    for csv_file_path, trial_date_str in csv_files_with_dates:
        analysis = StateSwitchingAnalysis(csv_file_path, visualize=True, trial_date_str=trial_date_str)
        
        # Calculate transition matrices for both groups
        for group in ['before_trial', 'after_trial']:
            analysis.calculate_transition_matrix(group)
        
        # Now create fixed positions after all transition matrices are calculated
        analysis.fixed_positions = analysis.create_fixed_positions()
        
        # Proceed to plot only if fixed_positions were successfully created
        if analysis.fixed_positions:
            all_durations_all_groups = []
            max_duration = 0
            for group in ['before_trial', 'after_trial']:
                analysis.calculate_switching_times(group)
                analysis.calculate_entropy_per_song(group)
                analysis.calculate_average_duration_per_song(group)
                all_durations = [duration for durations in analysis.switching_times[group].values() for duration in durations]
                all_durations_all_groups.extend(all_durations)
                if all_durations:  # Ensure there are durations
                    max_duration = max(max_duration, max(all_durations))
            
            for group in ['before_trial', 'after_trial']:
                analysis.plot_transition_graph_and_matrix(group)
                analysis.plot_switching_times_histogram(group, all_durations_all_groups)
                analysis.plot_switching_times_violin(group, max_duration)  # Updated Method
                analysis.calculate_transition_entropy(group)
                average_durations = analysis.calculate_average_durations(group)
                print(f"Average Durations for Group '{group}': {average_durations}")
            
            analysis.plot_summary_over_days(trial_date_str=trial_date_str)
            analysis.visualize_group_data()
        else:
            print("Fixed positions were not created. Skipping plotting.")