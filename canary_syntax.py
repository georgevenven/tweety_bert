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

class StateSwitchingAnalysis:
    def __init__(self, dir):
        data = np.load(dir, allow_pickle=True)

        self.hdbscan_labels = data['hdbscan_labels']
        self.file_info = data['file_indices']
        self.file_map = data['file_map']

        self.database = self.create_song_database()
        if self.database is not None:
            self.database = self.group_time_of_day(self.database)
        else:
            print("Error: Database creation failed.")

        # Set up results directory
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize attributes
        self.transition_matrices = {}
        self.transition_matrices_norm = {}
        self.switching_times = {}
        self.transition_entropies = {}
        self.total_entropy = {}
        self.group_labels = {}  # New attribute to store labels for each group

    def parse_date_time(self, file_path):
        parts = file_path.split('_')
        # remove .npz at the end of the last part
        parts[-1] = parts[-1].replace('.npz', '')
        try:
            file_name, month, day, hour, minute, second = parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            file_date = datetime(2024, month, day, hour, minute, second)
        except ValueError:
            print(f"Invalid date format in file path: {file_path}")
            return None
               
        return file_date, file_name

    def create_song_database(self):
        db = pd.DataFrame(columns=["song_id", "group_id", "file_name", "date_time", "labels"])

        if isinstance(self.file_map, np.ndarray) and self.file_map.ndim == 0:
            # If file_map is a 0-d array, convert it to a dict
            self.file_map = self.file_map.item()
        
        if isinstance(self.file_map, dict):
            for file_id, file_info in self.file_map.items():
                # Assuming file_info is a tuple with one element
                file_path = file_info[0] if isinstance(file_info, tuple) else file_info
                date_time, file_name = self.parse_date_time(file_path)

                index = np.where(self.file_info == file_id)

                db.loc[len(db)] = [file_name, None, file_path, date_time, self.hdbscan_labels[index].tolist()]

        else:
            print(f"Unexpected file_map type: {type(self.file_map)}")
            print(f"File map content: {self.file_map}")
        
        return db

    def database_to_csv(self, db):
        db.to_csv("song_database.csv", index=False)

    def group_time_of_day(self, db):
        # Create a new empty DataFrame to store the filtered results
        filtered_db = pd.DataFrame(columns=db.columns)

        for index, row in db.iterrows():
            hour = row['date_time'].hour
            if 6 <= hour < 9:
                new_row = row.copy()
                new_row['group_id'] = 'early_morning'
                filtered_db = pd.concat([filtered_db, pd.DataFrame([new_row])], ignore_index=True)
            elif 16 <= hour < 18:
                new_row = row.copy()
                new_row['group_id'] = 'late_night'
                filtered_db = pd.concat([filtered_db, pd.DataFrame([new_row])], ignore_index=True)

        # Assign the filtered DataFrame back to the class attribute
        self.database = filtered_db
        return filtered_db

    def calculate_transition_matrix(self, group):
        group_songs = self.database[self.database['group_id'] == group]['labels']
        
        if group_songs.empty:
            return
        
        try:
            group_labels = np.unique(np.concatenate(group_songs.values))
        except Exception as e:
            return
        
        self.group_labels[group] = group_labels
        n_labels = len(group_labels)
        
        if n_labels == 0:
            return
        
        transition_matrix = np.zeros((n_labels, n_labels))
        
        for song in group_songs:
            for i in range(len(song) - 1):
                from_state = np.where(group_labels == song[i])[0][0]
                to_state = np.where(group_labels == song[i+1])[0][0]
                if from_state != to_state:  # Exclude self-transitions
                    transition_matrix[from_state, to_state] += 1

        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix_norm = transition_matrix / row_sums[:, np.newaxis]
        transition_matrix_norm[transition_matrix_norm < 0.1] = 0

        self.transition_matrices[group] = transition_matrix
        self.transition_matrices_norm[group] = transition_matrix_norm

    def create_fixed_positions(self):
        # Create a fixed layout for all nodes
        G = nx.Graph()
        G.add_nodes_from(self.unique_labels)
        return nx.spring_layout(G, k=0.5, iterations=50)

    def create_transition_graph(self, group):
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
        group_data = self.collect_statistics(group)
        G = nx.DiGraph()
        
        for label in group_data["unique_labels"]:
            G.add_node(label)
        
        for i, from_label in enumerate(group_data["unique_labels"]):
            for j, to_label in enumerate(group_data["unique_labels"]):
                if i != j:  # Exclude self-edges
                    weight = group_data["transition_matrix_norm"][i][j]
                    if weight > 0:
                        G.add_edge(from_label, to_label, weight=weight)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # Plot graph
        node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        normalized_weights = [w / max_weight for w in edge_weights]
        edge_colors = plt.cm.YlOrRd(normalized_weights)

        nx.draw_networkx_nodes(G, self.fixed_positions, node_size=node_sizes, node_color='lightblue', ax=ax1)
        nx.draw_networkx_labels(G, self.fixed_positions, font_size=8, font_weight='bold', ax=ax1)
        nx.draw_networkx_edges(G, self.fixed_positions, width=[w * 3 for w in normalized_weights],
                               edge_color=edge_colors, arrows=True,
                               arrowsize=20, ax=ax1, connectionstyle="arc3,rad=0.1")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=max_weight))
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, label='Transition Probability')

        ax1.set_title(f"Transition Graph for Group {group}\n(Excluding Noise State, Self-Transitions, and Transitions < 0.1)", fontsize=16)
        ax1.axis('off')

        # Plot transition matrix as heatmap
        sns.heatmap(group_data["transition_matrix_norm"], cmap='YlGnBu', ax=ax2, 
                    xticklabels=group_data["unique_labels"], 
                    yticklabels=group_data["unique_labels"],
                    cbar_kws={'label': 'Transition Probability'})
        ax2.set_title(f"Transition Matrix for Group {group}", fontsize=16)
        ax2.set_xlabel("To State")
        ax2.set_ylabel("From State")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'transition_graph_matrix_group_{group}.png'), dpi=300)
        plt.close()

    def calculate_switching_times(self, group):
        switching_times = []
        group_songs = self.database[self.database['group_id'] == group]['labels']
        for song in group_songs:
            for i in range(len(song) - 1):
                current_state = song[i]
                current_duration = 1
                for label in song[i+1:]:
                    if label == current_state:
                        current_duration += 1
                    else:
                        if current_duration < 1000:
                            switching_times.append(current_duration)
                        current_state = label
                        current_duration = 1
                if current_duration < 1000:
                    switching_times.append(current_duration)
        self.switching_times[group] = np.array(switching_times)

    def plot_switching_times_histogram(self, group):
        plt.figure(figsize=(14, 8))
        bins = np.linspace(0, 100, 501)
        counts, bins, _ = plt.hist(self.switching_times[group], bins=bins, edgecolor='black', alpha=0.7)

        plt.title(f'Histogram of State Switching Times for Group {group}\n(Excluding Noise State, Times < 1000)', fontsize=16)
        plt.xlabel('Duration (number of time steps)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        mean_time = np.mean(self.switching_times[group])
        median_time = np.median(self.switching_times[group])
        plt.axvline(mean_time, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_time:.2f}')
        plt.axvline(median_time, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_time:.2f}')

        plt.legend(fontsize=10)

        stats_text = (f'Mean: {mean_time:.2f}\n'
                      f'Median: {median_time:.2f}\n'
                      f'Max: {np.max(self.switching_times[group])}\n'
                      f'Min: {np.min(self.switching_times[group])}\n'
                      f'Std Dev: {np.std(self.switching_times[group]):.2f}')
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'switching_times_histogram_group_{group}.png'), dpi=300)
        plt.close()

    def calculate_transition_entropy(self, group):
        if self.transition_matrices[group] is None or self.transition_matrices[group].size == 0:
            self.transition_entropies[group] = {}
            self.total_entropy[group] = 0
            return

        transition_matrix = self.transition_matrices_norm[group]
        state_frequencies = np.sum(self.transition_matrices[group], axis=1)
        state_frequencies = state_frequencies / np.sum(state_frequencies)

        self.transition_entropies[group] = {}
        self.total_entropy[group] = 0

        for i, state in enumerate(self.group_labels[group]):
            probabilities = transition_matrix[i]
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
            if probabilities.size > 0:
                entropy = -np.sum(probabilities * np.log2(probabilities))
            else:
                entropy = 0
            self.transition_entropies[group][state] = entropy

            # Contribute to total entropy
            self.total_entropy[group] += entropy * state_frequencies[i]

    def collect_statistics(self, group):
        group_songs = self.database[self.database['group_id'] == group]['labels']
        
        all_labels = []
        transitions = []
        
        for song_labels in group_songs:
            # Remove -1 labels and convert to numpy array
            song_labels = np.array([label for label in song_labels if label != -1])
            
            if len(song_labels) > 1:  # Ensure there's at least one transition
                all_labels.extend(song_labels)
                
                # Calculate transitions for this song, excluding self-transitions
                song_transitions = []
                for i in range(len(song_labels) - 1):
                    if song_labels[i] != song_labels[i+1]:
                        song_transitions.append((song_labels[i], song_labels[i+1]))
                
                transitions.extend(song_transitions)
        
        all_labels = np.array(all_labels)
        transitions = np.array(transitions)
        
        unique_labels = np.unique(all_labels)
        num_labels = len(unique_labels)
        
        # Create transition matrix
        transition_matrix = np.zeros((num_labels, num_labels), dtype=int)
        for from_label, to_label in transitions:
            i = np.where(unique_labels == from_label)[0][0]
            j = np.where(unique_labels == to_label)[0][0]
            transition_matrix[i, j] += 1
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix_norm = np.divide(transition_matrix, row_sums[:, np.newaxis], 
                                           where=row_sums[:, np.newaxis] != 0)
        
        # Calculate label frequencies
        label_counts = np.bincount(all_labels)
        label_frequencies = label_counts / len(all_labels)
        
        return {
            "group": group,  # Keep group as string
            "num_songs": int(len(group_songs)),
            "num_labels": int(len(unique_labels)),
            "total_transitions": int(np.sum(transition_matrix)),
            "label_frequencies": {int(k): float(v) for k, v in zip(unique_labels, label_frequencies)},
            "transition_matrix": transition_matrix.tolist(),
            "transition_matrix_norm": transition_matrix_norm.tolist(),
            "unique_labels": [int(label) for label in unique_labels.tolist()]
        }

    def visualize_group_data(self):
        group_data = []
        for group in self.database['group_id'].unique():
            stats = self.collect_statistics(group)
            group_data.append({
                'group': stats['group'],
                'num_songs': stats['num_songs'],
                'total_transitions': stats['total_transitions'],
                'num_labels': stats['num_labels']
            })
        
        groups = [d['group'] for d in group_data]
        num_songs = [d['num_songs'] for d in group_data]
        total_transitions = [d['total_transitions'] for d in group_data]
        num_labels = [d['num_labels'] for d in group_data]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        ax1.bar(groups, num_songs, color='skyblue')
        ax1.set_title('Number of Songs per Group')
        ax1.set_ylabel('Number of Songs')
        
        ax2.bar(groups, num_labels, color='lightgreen')
        ax2.set_title('Number of Unique Labels per Group')
        ax2.set_ylabel('Number of Unique Labels')
        
        ax3.bar(groups, total_transitions, color='salmon')
        ax3.set_title('Total Transitions per Group')
        ax3.set_xlabel('Group')
        ax3.set_ylabel('Number of Transitions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'group_data_distribution.png'))
        plt.close()

        print(f"Group data distribution visualization saved to {os.path.join(self.results_dir, 'group_data_distribution.png')}")

    def plot_transition_matrix_difference(self, group1, group2):
        if group1 not in self.transition_matrices_norm or group2 not in self.transition_matrices_norm:
            print(f"One or both groups ({group1}, {group2}) do not have transition matrices.")
            return

        matrix1 = self.transition_matrices_norm[group1]
        matrix2 = self.transition_matrices_norm[group2]

        if matrix1.shape != matrix2.shape:
            print(f"Transition matrices for groups {group1} and {group2} have different shapes.")
            return

        difference_matrix = matrix1 - matrix2

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(difference_matrix, cmap='coolwarm', center=0, ax=ax,
                    xticklabels=self.group_labels[group1], yticklabels=self.group_labels[group1],
                    cbar_kws={'label': 'Difference in Transition Probability'})
        ax.set_title(f"Difference in Transition Matrices: {group1} - {group2}", fontsize=16)
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'transition_matrix_difference_{group1}_{group2}.png'), dpi=300)
        plt.close()

        print(f"Transition matrix difference plot saved to {os.path.join(self.results_dir, f'transition_matrix_difference_{group1}_{group2}.png')}")

    def run_analysis(self):
        results = {"groups": []}

        self.unique_labels = np.unique(np.concatenate(self.database['labels']))
        self.n_labels = len(self.unique_labels)
        self.num_groups = len(self.database['group_id'].unique())

        self.fixed_positions = self.create_fixed_positions()

        unique_groups = self.database['group_id'].unique()

        for group in unique_groups:
            group_songs = self.database[self.database['group_id'] == group]['labels']
            if not group_songs.empty:
                try:
                    self.calculate_transition_matrix(group)
                    self.plot_transition_graph_and_matrix(group)
                    self.calculate_switching_times(group)
                    self.plot_switching_times_histogram(group)
                    self.calculate_transition_entropy(group)
                    group_stats = self.collect_statistics(group)
                    group_stats["phrase_entropy"] = self.transition_entropies[group]
                    group_stats["total_song_entropy"] = self.total_entropy[group]
                    results["groups"].append(group_stats)
                except Exception as e:
                    print(f"Error processing group {group}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Group {group} has no songs. Skipping analysis.")

        self.visualize_group_data()
        self.plot_entropy_values(results)

        # Example usage for plotting the difference between two groups
        if self.num_groups == 2:
            unique_groups = self.database['group_id'].unique()
            self.plot_transition_matrix_difference(unique_groups[0], unique_groups[1])

        # Convert all keys to strings
        def convert_keys_to_str(d):
            if isinstance(d, dict):
                return {str(k): convert_keys_to_str(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_keys_to_str(i) for i in d]
            else:
                return d

        results = convert_keys_to_str(results)

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(self.results_dir, "state_switching_analysis.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Analysis complete. Results saved to {os.path.join(self.results_dir, 'state_switching_analysis.json')}")

    def plot_entropy_values(self, results):
        groups = [group["group"] for group in results["groups"]]
        total_entropies = [group["total_song_entropy"] for group in results["groups"]]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(groups, total_entropies, color='purple')
        ax.set_title('Total Song Entropy per Group')
        ax.set_xlabel('Group')
        ax.set_ylabel('Total Song Entropy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'total_song_entropy_per_group.png'))
        plt.close()

        print(f"Total song entropy visualization saved to {os.path.join(self.results_dir, 'total_song_entropy_per_group.png')}")

# Usage
analysis = StateSwitchingAnalysis(dir="/media/george-vengrovski/flash-drive/labels_SHAM_5271.npz")
analysis.run_analysis()
