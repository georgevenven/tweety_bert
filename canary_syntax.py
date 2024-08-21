import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter


# we will compare the syntactic structure of the song between baseline and DOI conditions by
# measuring syllable variability, phrase transition entropy, phrase duration distributions, and other syntax
# measures described previously by the lab including prediction suffix trees (Koparkar et al.; Markowitz et
# al.). We will also determine the overall variability of the song by calculating the total transition entropy over all
# phrases, which considers the transition entropies at each phrase and weights them by their frequency of
# occurrence. Additionally, we will calculate the percentage of transitions from phrase to phrase and create
# transition matrices for the baseline and DOI conditions. We will then perform a Chi-square test of
# independence on these matrices to determine if new transitions were observed or certain transitions have
# disappeared. The Chi-square test will compare the observed percentages of each transition in the baseline and
# DOI conditions to the expected percentages assuming independence.Transition entropy
# Transition entropy is a measure of uncertainty of sequence at a given syllable (Sakata and Brainard, 2006). With c different outputs from the given syllable ‘a’ and P(i) the probability of the ith outcome, we calculate the entropy Ha as
# Ha=−∑ci=1P(i)logP(i)��=-∑�=1��������
# We call this value ‘transition entropy per branchpoint’ in Figure 3A.
# To determine the overall variability of song before and after mMAN lesions, we calculated total transition entropy TE, over all syllables ‘b’ as:
# TE=−∑nb=1HbP(b)��=-∑�=1�����
# where Hb is the transition entropy at ‘b’ and P(b) is the frequency of syllable ‘b’ (Chatfield and Lemon, 1970; Katahira et al., 2013).
# History dependence
# History dependence is a previously established metric that measures the extent to which the most common transition at a given syllable is influenced by the transition at the last occurrence of this syllable (Warren et al., 2012). It has been used to characterize instances of apparent sequence variability, where seemingly variable transitions are always strictly alternating. For example, if the possible transitions from syllable ‘a’ are ‘ab’ or ‘ac’ but these strictly alternated (‘ab … ac … ab … ac’ and so on), then the seemingly variable branchpoint ‘a’ is perfectly predictable based on its history (Warren et al., 2012). Such apparent variability should be largely eliminated in our sequence analysis by the introduction of context-dependent states (i.e., in this example, the ‘a’ would be re-labeled as ‘a1’ or ‘a2’ depending on the context in which it occurs) and identification of chunks. However, if higher-order dependencies in the song determine the order of chunks, we might still expect some variable transitions to be governed by history dependence. If ‘ab’ is the most frequently occurring transition from ‘a’, and ‘ac’ is the collection of all other transitions from ‘a’, we define history dependence D of ‘a’ as:
# D=P(abn|abn−1)−P(abn|acn−1)∨�=�������-1-�������-1∨
# where P(abn|abn−1)�������-1 is the conditional probability of ‘ab’ transition given that ‘ab’ transition occurred at the previous instance of ‘a’ and P(abn|acn−1)�������-1 is the conditional probability of ‘ab’ transition given that ‘ac’ transition occurred at the previous instance of ‘a’.
# Chunk consistency
# As defined above, a chunk is defined by a single dominant sequence, but may have a small amount of variability across different instances. To quantify the stereotypy of chunks, we used a measure based on sequence consistency previously defined for relatively stereotyped zebra finch songs (Scharff and Nottebohm, 1991). Across all instances of a given chunk, we identified the syllable sequence that occurred most often as the ‘dominant sequence’. We then defined ‘n_dominant’ as the number of instances of the dominant sequence, and ‘n_other’ as the number of instances of other sequence variants for the chunk.
# We quantified chunk consistency C as the proportion of total instances that were the dominant sequence:
# C=ndominantndominant+nother�=������������������+���ℎ��
# To compare a chunk before and after mMAN lesions, the dominant sequence for the pre-lesion chunk was used as a reference, regardless of whether the same sequence qualified as a chunk post lesion. To quantify chunk consistency post lesion, the most dominant sequence post lesion was used (even if that was not the same as the most dominant sequence pre lesion).
# Repeat number variability
# To study the influence of mMAN on repeat phrases, we examined the distribution of repeat numbers before and after lesions. We quantified the variability of these distributions as their coefficient of variation (CV = standard deviation/mean). ~~~~ Integrate these caclulations into the syntax code ... do so carefully and double check your code and work!

class StateSwitchingAnalysis:
    def __init__(self, dir):
        data = np.load(dir, allow_pickle=True)

        self.hdbscan_labels = data['hdbscan_labels']
        self.song_ids = data['combined_sample_ids']
        self.group_ids = data['combined_group_ids']
        
        # Remove noise cluster (-1) and update related arrays
        non_noise_mask = self.hdbscan_labels != -1
        self.hdbscan_labels = self.hdbscan_labels[non_noise_mask]
        self.song_ids = self.song_ids[non_noise_mask]
        self.group_ids = self.group_ids[non_noise_mask]

        self.unique_labels = np.unique(self.hdbscan_labels)
        self.n_labels = len(self.unique_labels)
        self.num_groups = len(np.unique(self.group_ids))

        self.songs = self.group_songs()

        self.transition_matrices = {group: None for group in range(self.num_groups)}
        self.transition_matrices_norm = {group: None for group in range(self.num_groups)}
        self.switching_times = {group: None for group in range(self.num_groups)}

    def group_songs(self):
        grouped_songs = {group: [] for group in range(self.num_groups)}
        unique_song_ids = np.unique(self.song_ids)
        
        for id in unique_song_ids:
            song_mask = self.song_ids == id
            hdbscan_labels_masked = self.hdbscan_labels[song_mask]
            if len(hdbscan_labels_masked) > 0:
                group_id = self.group_ids[song_mask][0]
                grouped_songs[group_id].append(hdbscan_labels_masked)
        
        return grouped_songs

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
        pos = nx.spring_layout(G, k=0.01, iterations=50)
        node_sizes = [300 * (1 + G.degree(node)) for node in G.nodes()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
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
        plt.savefig(f'transition_graph_and_matrix_group_{group}.png', dpi=300)
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
        plt.savefig(f'switching_times_histogram_group_{group}.png', dpi=300)
        plt.close()

    def print_statistics(self, group):
        labels = np.concatenate(self.songs[group])
        print(f"\nStatistics for Group {group}:")
        print(f"Number of unique labels (excluding noise state): {self.n_labels}")
        print("\nTop 5 most common transitions:")
        transitions = [(labels[i], labels[i+1]) for i in range(len(labels)-1)
                       if labels[i] != labels[i+1]]
        for transition, count in Counter(transitions).most_common(5):
            print(f"From {transition[0]} to {transition[1]}: {count} times")

        print("\nTransition matrix:")
        print(self.transition_matrices[group])

        print("\nNormalized transition matrix:")
        print(self.transition_matrices_norm[group])

        print(f"\nMean switching time: {np.mean(self.switching_times[group]):.2f}")
        print(f"Median switching time: {np.median(self.switching_times[group]):.2f}")
        print(f"Maximum switching time: {np.max(self.switching_times[group])}")
        print(f"Minimum switching time: {np.min(self.switching_times[group])}")
        print(f"Standard deviation of switching times: {np.std(self.switching_times[group]):.2f}")
        print(f"Total number of switches: {len(self.switching_times[group])}")

        print(f"\nTotal Transition Entropy: {self.total_entropy[group]}")
        print("\nTransition Entropies per state:")
        for state, entropy in self.transition_entropies[group].items():
            print(f"State {state}: {entropy}")

        print("\nHistory Dependence per state:")
        for state, dependence in self.history_dependence[group].items():
            print(f"State {state}: {dependence}")

        print("\nChunk Consistency:")
        for chunk, consistency in self.chunk_consistency[group].items():
            print(f"Chunk {chunk}: {consistency}")

        print("\nRepeat Number Variability (CV):")
        for phrase, cv in self.repeat_variability[group].items():
            print(f"Phrase {phrase}: {cv}")

    def run_analysis(self):
        for group in range(self.num_groups):
            if self.songs[group]:  # Only analyze groups with songs
                self.calculate_transition_matrix(group)
                self.plot_transition_graph_and_matrix(group)
                self.calculate_switching_times(group)
                self.plot_switching_times_histogram(group)
                self.print_statistics(group)
            else:
                print(f"Group {group} has no songs after removing noise cluster. Skipping analysis.")

# Usage
analysis = StateSwitchingAnalysis(dir="/home/george-vengrovski/Documents/tweety_bert/files/labels_Yarden_LLB3_Whisperseg.npz")
analysis.run_analysis()