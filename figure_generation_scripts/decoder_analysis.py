import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
from scipy.stats import chi2_contingency
import json
import os
from datetime import datetime
import pandas as pd
import re
import ast

class StateSwitchingAnalysis:
    def __init__(self, csv_file, visualize=False):
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Apply the parsing function to the relevant columns
        data['syllable_onsets_offsets_ms'] = data['syllable_onsets_offsets_ms'].apply(self.parse_json_safe)
        data['syllable_onsets_offsets_timebins'] = data['syllable_onsets_offsets_timebins'].apply(self.parse_json_safe)
        
        # Filter out rows where song_present is False
        data = data[data['song_present']].reset_index(drop=True)
        
        # Parse date and file name, and extract labels
        data[['date_time', 'file_base_name']] = data['file_name'].apply(
            lambda x: pd.Series(self.parse_date_time(x, format="standard"))
        )
        
        # Extract labels sequence from syllable_onsets_offsets_ms
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
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up image directory for debugging visualizations
        self.img_dir = "imgs/vis_smoothed"
        os.makedirs(self.img_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Extract the base name from the CSV file path
        self.base_name = os.path.splitext(os.path.basename(csv_file))[0]

        # Create the song database
        self.database = self.create_song_database(data)
        if self.database is not None:
            self.database = self.group_before_after_trial(self.database)
            self.database_to_csv(self.database)
        else:
            print("Error: Database creation failed.")

        # Initialize attributes
        self.transition_matrices = {}
        self.transition_matrices_norm = {}
        self.switching_times = {}
        self.transition_entropies = {}
        self.total_entropy = {}
        self.group_labels = {}  # New attribute to store labels for each group

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
        
        return database

    def group_before_after_trial(self, db, trial_date_str="2024-04-08 00:00:00"):
        """
        Group the database entries into 'before_trial' and 'after_trial' based on a trial date.

        Args:
        - db (pd.DataFrame): The song database.
        - trial_date_str (str): The trial date in "%Y-%m-%d %H:%M:%S" format.

        Returns:
        - pd.DataFrame: The grouped database with a new 'group_id' column.
        """
        # Convert trial_date string to datetime
        try:
            trial_date = datetime.strptime(trial_date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            print(f"Invalid trial_date format: {trial_date_str}\nError: {e}")
            return db
        
        # Create a new column 'group_id'
        db['group_id'] = db['date_time'].apply(
            lambda x: 'before_trial' if x < trial_date else 'after_trial'
        )
        
        print(f"Database grouped into 'before_trial' and 'after_trial' based on {trial_date_str}")
        
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
        group_songs = self.database[self.database['group_id'] == group]['labels']
        
        if group_songs.empty:
            print(f"No songs found for group: {group}")
            return
        
        try:
            # Get all unique labels in the group
            group_labels = np.unique(np.concatenate(group_songs.values))
        except Exception as e:
            print(f"Error concatenating labels for group {group}: {e}")
            return
        
        self.group_labels[group] = group_labels
        n_labels = len(group_labels)
        
        if n_labels == 0:
            print(f"No unique labels found for group: {group}")
            return
        
        transition_matrix = np.zeros((n_labels, n_labels))
        
        for song in group_songs:
            for i in range(len(song) - 1):
                from_label = song[i]
                to_label = song[i+1]
                if from_label != to_label:  # Exclude self-transitions
                    from_idx = np.where(group_labels == from_label)[0][0]
                    to_idx = np.where(group_labels == to_label)[0][0]
                    transition_matrix[from_idx, to_idx] += 1
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix_norm = transition_matrix / row_sums[:, np.newaxis]
        transition_matrix_norm[transition_matrix_norm < 0.1] = 0  # Thresholding
        
        self.transition_matrices[group] = transition_matrix
        self.transition_matrices_norm[group] = transition_matrix_norm
        
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
        return nx.spring_layout(G, k=0.5, iterations=50)

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
        group_data = self.collect_statistics(group)
        if group_data is None:
            print(f"No data to plot for group: {group}")
            return
        
        G = self.create_transition_graph(group)
        if G is None:
            return
        
        # Ensure fixed_positions are created
        if self.fixed_positions is None:
            print("Creating fixed positions...")
            self.fixed_positions = self.create_fixed_positions()
            if not self.fixed_positions:
                print("Fixed positions could not be created. Skipping plotting.")
                return
        
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

        ax1.set_title(f"Transition Graph for Group '{group}'\n(Excluding Self-Transitions and Transitions < 0.1)", fontsize=16)
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

        This function creates
