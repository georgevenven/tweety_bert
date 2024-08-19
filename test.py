import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter

class StateSwitchingAnalysis:
    def __init__(self, smoothed_labels):
        self.smoothed_labels = smoothed_labels
        self.unique_labels = np.unique(smoothed_labels)
        self.unique_labels = self.unique_labels[self.unique_labels != 0]
        self.n_labels = len(self.unique_labels)
        self.transition_matrix = None
        self.transition_matrix_norm = None
        self.switching_times = None

    def calculate_transition_matrix(self):
        self.transition_matrix = np.zeros((self.n_labels, self.n_labels))
        for i in range(len(self.smoothed_labels) - 1):
            if self.smoothed_labels[i] != 0 and self.smoothed_labels[i+1] != 0:
                from_state = np.where(self.unique_labels == self.smoothed_labels[i])[0][0]
                to_state = np.where(self.unique_labels == self.smoothed_labels[i+1])[0][0]
                if from_state != to_state:  # Exclude self-transitions
                    self.transition_matrix[from_state, to_state] += 1

        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.transition_matrix_norm = self.transition_matrix / row_sums[:, np.newaxis]
        self.transition_matrix_norm[self.transition_matrix_norm < 0.1] = 0

    def create_transition_graph(self):
        G = nx.DiGraph()
        for label in self.unique_labels:
            G.add_node(label)
        for i, from_label in enumerate(self.unique_labels):
            for j, to_label in enumerate(self.unique_labels):
                if i != j:  # Exclude self-edges
                    weight = self.transition_matrix_norm[i, j]
                    if weight > 0:
                        G.add_edge(from_label, to_label, weight=weight)
        return G

    def plot_transition_graph_and_matrix(self):
        G = self.create_transition_graph()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # Plot graph
        pos = nx.spring_layout(G, k=0.01, iterations=50)
        node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights)
        normalized_weights = [w / max_weight for w in edge_weights]
        edge_colors = plt.cm.YlOrRd(normalized_weights)

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax1)
        nx.draw_networkx_edges(G, pos, width=[w * 3 for w in normalized_weights],
                               edge_color=edge_colors, arrows=True,
                               arrowsize=20, ax=ax1, connectionstyle="arc3,rad=0.1")

        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=max_weight))
        sm.set_array([])
        plt.colorbar(sm, ax=ax1, label='Transition Probability')

        ax1.set_title("Improved Transition Graph from HDBSCAN Labels\n(Excluding Noise State, Self-Transitions, and Transitions < 0.1)", fontsize=16)
        ax1.axis('off')

        # Plot transition matrix as heatmap
        sns.heatmap(self.transition_matrix_norm, cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Transition Probability'})
        ax2.set_title("Normalized Transition Matrix Heatmap\n(Transitions < 0.1 set to zero)", fontsize=16)
        ax2.set_xlabel("To State")
        ax2.set_ylabel("From State")
        ax2.set_xticklabels(self.unique_labels)
        ax2.set_yticklabels(self.unique_labels)

        plt.tight_layout()
        plt.show()

    def calculate_switching_times(self):
        self.switching_times = []
        current_state = self.smoothed_labels[0]
        current_duration = 1

        for label in self.smoothed_labels[1:]:
            if label == current_state:
                current_duration += 1
            else:
                if current_state != 0 and current_duration < 1000:
                    self.switching_times.append(current_duration)
                current_state = label
                current_duration = 1

        # Add the last state duration if it hasn't switched at the end
        if current_state != 0 and current_duration < 1000:
            self.switching_times.append(current_duration)

        self.switching_times = np.array(self.switching_times)

    def plot_switching_times_histogram(self):
        plt.figure(figsize=(14, 8))
        bins = np.linspace(0, 100, 501)
        counts, bins, _ = plt.hist(self.switching_times, bins=bins, edgecolor='black', alpha=0.7)

        plt.title('Histogram of State Switching Times (Excluding State 0, Times < 1000)', fontsize=16)
        plt.xlabel('Duration (number of time steps)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        mean_time = np.mean(self.switching_times)
        median_time = np.median(self.switching_times)
        plt.axvline(mean_time, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_time:.2f}')
        plt.axvline(median_time, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_time:.2f}')

        plt.legend(fontsize=10)

        stats_text = (f'Mean: {mean_time:.2f}\n'
                      f'Median: {median_time:.2f}\n'
                      f'Max: {np.max(self.switching_times)}\n'
                      f'Min: {np.min(self.switching_times)}\n'
                      f'Std Dev: {np.std(self.switching_times):.2f}')
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)

        plt.tight_layout()
        plt.savefig('switching_times_histogram_filtered.png', dpi=300)
        plt.show()

    def print_statistics(self):
        print(f"Number of unique labels (excluding zero state): {self.n_labels}")
        print("\nTop 5 most common transitions (excluding zero state and self-transitions):")
        transitions = [(self.smoothed_labels[i], self.smoothed_labels[i+1]) for i in range(len(self.smoothed_labels)-1)
                       if self.smoothed_labels[i] != 0 and self.smoothed_labels[i+1] != 0 and self.smoothed_labels[i] != self.smoothed_labels[i+1]]
        for transition, count in Counter(transitions).most_common(5):
            print(f"From {transition[0]} to {transition[1]}: {count} times")

        print("\nTransition matrix (excluding zero state and self-transitions):")
        print(self.transition_matrix)

        print("\nNormalized transition matrix (excluding zero state and self-transitions):")
        print(self.transition_matrix_norm)

        zero_percentage = (self.smoothed_labels == 0).mean() * 100
        print(f"\nPercentage of data points classified as zero: {zero_percentage:.2f}%")

        print(f"\nMean switching time: {np.mean(self.switching_times):.2f}")
        print(f"Median switching time: {np.median(self.switching_times):.2f}")
        print(f"Maximum switching time: {np.max(self.switching_times)}")
        print(f"Minimum switching time: {np.min(self.switching_times)}")
        print(f"Standard deviation of switching times: {np.std(self.switching_times):.2f}")
        print(f"Total number of switches (excluding state 0 and times >= 1000): {len(self.switching_times)}")

    def run_analysis(self):
        self.calculate_transition_matrix()
        self.plot_transition_graph_and_matrix()
        self.calculate_switching_times()
        self.plot_switching_times_histogram()
        self.print_statistics()

smoothed_labels = np.load('smoothed_labels.npz')['smoothed_labels']
analysis = StateSwitchingAnalysis(smoothed_labels)
analysis.run_analysis()