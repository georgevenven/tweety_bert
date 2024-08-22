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

        print(f"file info {self.file_info}")
        print(f"file map {self.file_map}")

        self.database = self.create_song_database()

        # save database to csv
        self.database.to_csv("song_database.csv", index=False)

        # # Remove noise cluster (-1) and update related arrays
        # non_noise_mask = self.hdbscan_labels != -1
        # self.hdbscan_labels = self.hdbscan_labels[non_noise_mask]
        # self.song_ids = self.song_ids[non_noise_mask]
        # self.group_ids = self.group_ids[non_noise_mask]

        # self.unique_labels = np.unique(self.hdbscan_labels)
        # self.n_labels = len(self.unique_labels)
        # self.num_groups = len(np.unique(self.group_ids))

        # self.songs = self.group_songs()

        # self.transition_matrices = {group: None for group in range(self.num_groups)}
        # self.transition_matrices_norm = {group: None for group in range(self.num_groups)}
        # self.switching_times = {group: None for group in range(self.num_groups)}
        # self.transition_entropies = {group: {} for group in range(self.num_groups)}
        # self.total_entropy = {group: 0 for group in range(self.num_groups)}

        # # Set up results directory
        # self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        # os.makedirs(self.results_dir, exist_ok=True)

        # # Create fixed positions for graph nodes
        # self.fixed_positions = self.create_fixed_positions()

    def parse_date_time(self, file_path):
        parts = file_path.split('_')
        # remove .npz at the end of the last part
        parts[-1] = parts[-1].replace('.npz', '')
        try:
            print(f"parts {parts}")
            month, day, hour, minute, second = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
            file_date = datetime(2024, month, day, hour, minute, second)
        except ValueError:
            print(f"Invalid date format in file path: {file_path}")
            return None
               
        return file_date


    def create_song_database(self):
        db = pd.DataFrame(columns=["song_id", "file_name", "date_time", "labels"])

        if isinstance(self.file_map, np.ndarray) and self.file_map.ndim == 0:
            # If file_map is a 0-d array, convert it to a dict
            self.file_map = self.file_map.item()
        
        if isinstance(self.file_map, dict):
            for file_id, file_info in self.file_map.items():
                # Assuming file_info is a tuple with one element
                file_path = file_info[0] if isinstance(file_info, tuple) else file_info
                print(f"File ID: {file_id}, File Path: {file_path}")

                date_time = self.parse_date_time(file_path)

                index = np.where(self.file_info == file_id)

                db.loc[len(db)] = [file_id, file_path, date_time, self.hdbscan_labels[index]]
                
                # Here you can add the file information to your database
                # For example:
                # self.song_database[file_id] = {'file_path': file_path, 'other_info': ...}

        else:
            print(f"Unexpected file_map type: {type(self.file_map)}")
            print(f"File map content: {self.file_map}")

        
        return db
    

    def calculate_transition_matrix(self, group):
        transition_matrix = np.zeros((self.n_labels, self.n_labels))
        
        for song in self.songs[group]:
            for i in range(len(song) - 1):
                from_state = np.where(self.unique_labels == song[i])[0][0]
                to_state = np.where(self.unique_labels == song[i+1])[0][0]
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
        for label in self.unique_labels:
            G.add_node(label)
        for i, from_label in enumerate(self.unique_labels):
            for j, to_label in enumerate(self.unique_labels):
                if i != j:  # Exclude self-edges
                    weight = self.transition_matrices_norm[group][i, j]
                    if weight > 0:
                        G.add_edge(from_label, to_label, weight=weight)
        return G

    def plot_transition_graph_and_matrix(self, group):
        G = self.create_transition_graph(group)
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
        sns.heatmap(self.transition_matrices_norm[group], cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Transition Probability'})
        ax2.set_title(f"Normalized Transition Matrix Heatmap for Group {group}\n(Transitions < 0.1 set to zero)", fontsize=16)
        ax2.set_xlabel("To State")
        ax2.set_ylabel("From State")
        ax2.set_xticklabels(self.unique_labels)
        ax2.set_yticklabels(self.unique_labels)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'transition_graph_and_matrix_group_{group}.png'), dpi=300)
        plt.close()

    def calculate_switching_times(self, group):
        switching_times = []
        for song in self.songs[group]:
            current_state = song[0]
            current_duration = 1
            for label in song[1:]:
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
            print(f"Warning: Transition matrix for group {group} is empty or not initialized.")
            self.transition_entropies[group] = {}
            self.total_entropy[group] = 0
            return

        transition_matrix = self.transition_matrices_norm[group]
        state_frequencies = np.sum(self.transition_matrices[group], axis=1)
        state_frequencies = state_frequencies / np.sum(state_frequencies)

        self.transition_entropies[group] = {}
        self.total_entropy[group] = 0

        for i, state in enumerate(self.unique_labels):
            probabilities = transition_matrix[i]
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
            if probabilities.size > 0:
                entropy = -np.sum(probabilities * np.log2(probabilities))
            else:
                entropy = 0
            self.transition_entropies[group][state] = entropy

            # Contribute to total entropy
            self.total_entropy[group] += entropy * state_frequencies[i]

    def calculate_chi_square(self, group1, group2):
        matrix1 = self.transition_matrices[group1]
        matrix2 = self.transition_matrices[group2]

        # Ensure matrices have the same shape
        max_shape = max(matrix1.shape[0], matrix2.shape[0])
        if matrix1.shape[0] < max_shape:
            matrix1 = np.pad(matrix1, ((0, max_shape - matrix1.shape[0]), (0, max_shape - matrix1.shape[1])))
        if matrix2.shape[0] < max_shape:
            matrix2 = np.pad(matrix2, ((0, max_shape - matrix2.shape[0]), (0, max_shape - matrix2.shape[1])))

        # Perform Chi-square test
        chi2, p_value, dof, expected = chi2_contingency([matrix1, matrix2])

        return chi2, p_value

    def collect_statistics(self, group):
        labels = np.concatenate(self.songs[group])
        transitions = [(int(labels[i]), int(labels[i+1])) for i in range(len(labels)-1)
                       if labels[i] != labels[i+1]]
        top_transitions = Counter(transitions).most_common(5)

        stats = {
            "group": int(group),
            "unique_labels": int(self.n_labels),
            "top_5_transitions": [{"from": int(t[0][0]), "to": int(t[0][1]), "count": int(t[1])} for t in top_transitions],
            "transition_matrix": self.transition_matrices[group].tolist(),
            "normalized_transition_matrix": self.transition_matrices_norm[group].tolist(),
            "switching_times": {
                "mean": float(np.mean(self.switching_times[group])),
                "median": float(np.median(self.switching_times[group])),
                "max": int(np.max(self.switching_times[group])),
                "min": int(np.min(self.switching_times[group])),
                "std_dev": float(np.std(self.switching_times[group])),
                "total_switches": int(len(self.switching_times[group]))
            },
            "total_transition_entropy": float(self.total_entropy[group]),
            "transition_entropies": {str(state): float(entropy) for state, entropy in self.transition_entropies[group].items()}
        }
        return stats

    def run_analysis(self):
        results = {"groups": [], "chi_square_tests": []}

        for group in range(self.num_groups):
            if self.songs[group]:  # Only analyze groups with songs
                try:
                    self.calculate_transition_matrix(group)
                    self.plot_transition_graph_and_matrix(group)
                    self.calculate_switching_times(group)
                    self.plot_switching_times_histogram(group)
                    self.calculate_transition_entropy(group)
                    results["groups"].append(self.collect_statistics(group))
                except Exception as e:
                    print(f"Error processing group {group}: {str(e)}")
            else:
                print(f"Group {group} has no songs after removing noise cluster. Skipping analysis.")

        # Compare transition matrices between groups using Chi-square test
        if self.num_groups > 1:
            for i in range(self.num_groups):
                for j in range(i+1, self.num_groups):
                    if self.songs[i] and self.songs[j]:
                        try:
                            chi2, p_value = self.calculate_chi_square(i, j)
                            results["chi_square_tests"].append({
                                "group1": int(i),
                                "group2": int(j),
                                "chi_square_statistic": float(chi2),
                                "p_value": float(p_value)
                            })
                        except Exception as e:
                            print(f"Error calculating Chi-square test for groups {i} and {j}: {str(e)}")

        # Ensure the results directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Save results as JSON
        with open(os.path.join(self.results_dir, "state_switching_analysis.json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"Analysis complete. Results saved to {os.path.join(self.results_dir, 'state_switching_analysis.json')}")

# Usage
analysis = StateSwitchingAnalysis(dir="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/labels_TEST_NEW_APPROACH.npz")
# analysis.run_analysis()