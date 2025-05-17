import os
import itertools
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from sklearn.metrics import v_measure_score, silhouette_score
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed

# --- Configuration Parameters (Hardcoded as requested) ---
# Data and Paths
# TODO: Replace with your actual data loading mechanism
# This function should return:
#   embeddings_100k (np.array): Subsampled high-dimensional embeddings (e.g., 100k x D)
#   gt_labels_100k (np.array): Corresponding ground truth labels (1D array)
#   gt_colors_map (dict or list): Mapping for GT labels to colors as used in UMAP_plots_from_npz.py
#                                Example: {0: [r,g,b,a], 1: [r,g,b,a], ...} or list of [r,g,b,a]
#   embeddings_full (np.array): Full high-dimensional embeddings for the fold
#   gt_labels_full (np.array): Corresponding full ground truth labels
def load_data_for_fold(fold_id, base_data_path="path/to/your/data", subsample_size=100000):
    """
    Placeholder for data loading.
    Replace with your actual data loading logic.
    This function should load subsampled data for sweeps and potentially full data for final runs.
    It should also load the ground truth color mapping.
    """
    print(f"Loading data for fold {fold_id} (Placeholder)...")
    # Example: Load a dummy dataset
    # Replace with your actual data loading
    rng = np.random.RandomState(42)
    original_dim = 196 # As per TweetyBERT
    n_classes = 10
    n_points_subsample = subsample_size
    n_points_full = subsample_size * 5 # Example full size

    # Dummy subsampled data
    embeddings_100k = rng.rand(n_points_subsample, original_dim)
    gt_labels_100k = rng.randint(0, n_classes, n_points_subsample)

    # Dummy full data (can be the same as subsample for this script's initial run)
    embeddings_full = rng.rand(n_points_full, original_dim) # Or load actual full data
    gt_labels_full = rng.randint(0, n_classes, n_points_full) # Or load actual full labels

    # Dummy GT color map (list of RGBA tuples/lists, or a dict)
    # Ensure this matches the structure expected by the plotting function
    # Example: using a standard colormap
    cmap_tab20 = plt.cm.get_cmap('tab20', n_classes)
    gt_colors_map_list = [cmap_tab20(i) for i in range(n_classes)]
    # Your UMAP_plots_from_npz.py seems to handle a list where index = label
    # And black for label 1 (potentially silence or unclassified in GT)
    # gt_colors_map_list[1] = (0,0,0,1) # Example: making label 1 black if needed

    # Convert to a dictionary if your plotting function prefers that for GT
    gt_colors_dict = {i: gt_colors_map_list[i] for i in range(n_classes)}


    print(f"Loaded dummy data: {embeddings_100k.shape[0]} subsampled points, {embeddings_full.shape[0]} full points.")
    return embeddings_100k, gt_labels_100k, gt_colors_dict, embeddings_full, gt_labels_full

OUTPUT_BASE_DIR = Path("./umbra_sweep_results")
PLOTS_DIR_NAME = "plots"
CURRENT_FOLD_ID = "fold1" # For single fold run; adapt for multi-fold loop

# UMAP Parameters
UMAP_METRICS = ['euclidean', 'cosine', 'braycurtis', 'canberra', 'correlation']
UMAP_N_NEIGHBORS_COARSE = [30, 100]
UMAP_MIN_DIST_COARSE = [0.0]
UMAP_N_COMPONENTS_COARSE = [2, 4]
UMAP_N_EPOCHS_COARSE = 100
UMAP_N_EPOCHS_REFINED = 400
UMAP_RANDOM_STATE = 42

# PCA Parameters
PCA_N_COMPONENTS = 20

# HDBSCAN Parameters
# Adjust min_cluster_size based on your data/expectations (e.g., ~1/2 rarest GT class size)
HDBSCAN_MIN_CLUSTER_SIZES = [50, 200, 500, 1000] # Example, adjust
HDBSCAN_MIN_SAMPLES_ABS = [1, 5, 10]
# HDBSCAN_MIN_SAMPLES_REL_FACTOR = [0.1] # Relative to min_cluster_size
HDBSCAN_CLUSTER_SELECTION_METHODS = ['eom', 'leaf']
HDBSCAN_ALLOW_SINGLE_CLUSTER = False
HDBSCAN_ALPHAS = [1.0, 1.5]

# Fixed HDBSCAN for UMAP Coarse Sweep evaluation
FAST_HDBSCAN_PARAMS = {
    'min_cluster_size': 100, # Small for speed
    'min_samples': 5,
    'cluster_selection_method': 'eom',
    'allow_single_cluster': False,
    'alpha': 1.0
}

# Joblib parallelization
N_JOBS = -2  # Use all cores except one

# Early rejection thresholds
EARLY_REJECT_UMAP_FER_THRESHOLD = 0.80
EARLY_REJECT_UMAP_NOISE_THRESHOLD = 0.50 # 50%
EARLY_REJECT_HDBSCAN_NOISE_THRESHOLD = 0.40 # 40%

# --- Helper Functions ---

def ensure_dir(path: Path):
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def get_gt_color_array(gt_labels, gt_colors_map_dict):
    """Generate a color array for plotting based on GT labels and a color map dict."""
    default_color = to_rgba('gray') # Color for labels not in map
    colors = [gt_colors_map_dict.get(label, default_color) for label in gt_labels]
    return np.array(colors)

def get_hdbscan_color_array(hdb_labels, hdbscan_cmap_name='viridis'):
    """Generate a color array for HDBSCAN labels. Noise (-1) is gray."""
    unique_labels = np.unique(hdb_labels)
    core_labels = sorted([l for l in unique_labels if l != -1])
    
    if not core_labels: # All noise
        return np.array([to_rgba('gray')] * len(hdb_labels))

    cmap = plt.cm.get_cmap(hdbscan_cmap_name, len(core_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(core_labels)}
    label_to_color[-1] = to_rgba('gray') # Noise color

    colors = [label_to_color[label] for label in hdb_labels]
    return np.array(colors)

def plot_umap_embedding(embedding, labels, title, save_path,
                        is_gt_labels=True,
                        gt_colors_map_dict=None, hdbscan_cmap_name='viridis',
                        point_size=1, alpha=0.1): # Adjusted alpha for better visibility
    """
    Plots UMAP embedding, styled similarly to UMAP_plots_from_npz.py.
    Uses 'dark_background' style.
    """
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if is_gt_labels:
            if gt_colors_map_dict is None:
                raise ValueError("gt_colors_map_dict must be provided for GT labels.")
            # Ensure labels are integers for dict lookup if they aren't already
            processed_labels = labels.astype(int)
            color_array = get_gt_color_array(processed_labels, gt_colors_map_dict)
        else:
            color_array = get_hdbscan_color_array(labels, hdbscan_cmap_name)

        ax.scatter(embedding[:, 0], embedding[:, 1], c=color_array, s=point_size, alpha=alpha, edgecolors='none')
        ax.set_title(title, fontsize=14, color='white')
        ax.axis('off') # Turn off axis numbers and ticks

        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

class UmapEvalMetrics:
    """
    Adapted from umap_eval.py to calculate FER and V-measure.
    Focuses on Macro-Averaged Matched FER with 100% penalty for unmapped GT classes.
    """
    def __init__(self, ground_truth_labels: np.ndarray, predicted_labels: np.ndarray):
        self.gt_labels = ground_truth_labels
        self.pred_labels = predicted_labels
        self.gt_unique_labels = np.unique(self.gt_labels)
        self.pred_unique_labels = np.unique(self.pred_labels)
        self.contingency_matrix = self._build_contingency_matrix()

    def _build_contingency_matrix(self) -> np.ndarray:
        gt_map = {label: i for i, label in enumerate(self.gt_unique_labels)}
        pred_map = {label: i for i, label in enumerate(self.pred_unique_labels)}
        
        matrix = np.zeros((len(self.gt_unique_labels), len(self.pred_unique_labels)), dtype=int)
        for gt, pred in zip(self.gt_labels, self.pred_labels):
            if pred == -1: # Skip noise points from predicted for initial matrix build for mapping
                continue
            matrix[gt_map[gt], pred_map[pred]] += 1
        return matrix

    def map_labels_optimal(self) -> Dict[int, int]:
        """Maps predicted cluster labels to ground truth labels using Hungarian algorithm."""
        if self.contingency_matrix.shape[1] == 0: # No non-noise predicted clusters
            return {}

        # Cost matrix: negative of overlaps to maximize overlap
        cost_matrix = -self.contingency_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping: gt_label_original -> pred_label_original
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            # Check if this assignment has any overlap
            if self.contingency_matrix[r, c] > 0:
                 # map original gt label to original pred label
                gt_label_original = self.gt_unique_labels[r]
                pred_label_original = self.pred_unique_labels[self.pred_unique_labels != -1][c] # Ensure we map to actual cluster labels
                mapping[gt_label_original] = pred_label_original
        return mapping


    def calculate_macro_avg_matched_fer(self) -> float:
        """
        Calculates Macro-Averaged Matched FER.
        GT classes not mapped to any HDBSCAN cluster get 100% FER.
        """
        optimal_mapping_gt_to_pred = self.map_labels_optimal() # gt_original -> pred_original
        
        per_gt_class_fer = {}
        gt_counts = Counter(self.gt_labels)

        for gt_class_val in self.gt_unique_labels:
            if gt_class_val not in gt_counts or gt_counts[gt_class_val] == 0:
                per_gt_class_fer[gt_class_val] = 0.0 # Or skip if class not in current data subset
                continue

            if gt_class_val not in optimal_mapping_gt_to_pred:
                per_gt_class_fer[gt_class_val] = 1.0  # 100% error for unmapped GT classes
            else:
                mapped_pred_class_val = optimal_mapping_gt_to_pred[gt_class_val]
                
                # Count how many items of this GT class were correctly assigned to the mapped_pred_class
                correct_assignments = 0
                # Count how many items of this GT class were assigned to *any* non-noise pred_class
                # (for denominator of matched FER)
                # Or, more simply, total items in this GT class
                total_gt_class_items = gt_counts[gt_class_val]

                for i in range(len(self.gt_labels)):
                    if self.gt_labels[i] == gt_class_val and self.pred_labels[i] == mapped_pred_class_val:
                        correct_assignments += 1
                
                if total_gt_class_items == 0: # Should not happen if gt_class_val in gt_counts
                     per_gt_class_fer[gt_class_val] = 0.0
                else:
                    fer_for_class = 1.0 - (correct_assignments / total_gt_class_items)
                    per_gt_class_fer[gt_class_val] = fer_for_class
        
        if not per_gt_class_fer:
            return 1.0 # All unmapped or no GT classes
            
        macro_avg_fer = np.mean(list(per_gt_class_fer.values()))
        return macro_avg_fer

    def calculate_v_measure(self) -> float:
        # V-measure handles noise points (-1) appropriately by default if they are in both true and pred.
        # If GT doesn't have -1, it might affect. For proxy generation, GT is clean.
        # We should probably filter out noise from pred_labels before v_measure if GT has no noise.
        # However, if we want to see how well clusters (excluding noise) match GT:
        non_noise_indices = self.pred_labels != -1
        if np.sum(non_noise_indices) == 0: # All noise
            return 0.0 
        return v_measure_score(self.gt_labels[non_noise_indices], self.pred_labels[non_noise_indices])

def calculate_silhouette(embedding, labels):
    """Calculates silhouette score. Returns 0 if only one cluster or all noise."""
    unique_labels = np.unique(labels)
    num_clusters = len([l for l in unique_labels if l != -1])
    if num_clusters < 2:
        return 0.0  # Silhouette score is not defined for < 2 clusters
    
    # Filter out noise for silhouette calculation
    core_point_indices = labels != -1
    if np.sum(core_point_indices) < 2: # Not enough core points
        return 0.0

    core_embeddings = embedding[core_point_indices]
    core_labels = labels[core_point_indices]
    
    # Check again if after filtering noise, we still have enough clusters
    if len(np.unique(core_labels)) < 2:
        return 0.0

    try:
        return silhouette_score(core_embeddings, core_labels)
    except ValueError:
        return 0.0 # Handles cases like all points in one cluster after noise removal

def calculate_noise_percentage(labels):
    """Calculates the percentage of points labeled as noise (-1)."""
    if len(labels) == 0:
        return 0.0
    noise_count = np.sum(labels == -1)
    return noise_count / len(labels)


def run_single_umap_config(config_tuple, embeddings_data, gt_labels_data, gt_colors_map_dict, fold_id, base_plot_path_str, stage_name):
    """
    Runs UMAP for a single configuration, then a fast HDBSCAN, calculates metrics, and plots.
    Used in Stage 1a.
    """
    metric, n_neighbors, min_dist, n_components, n_epochs = config_tuple
    
    # Create a unique identifier for this UMAP config for filenames
    umap_config_name = f"umap_metric-{metric}_k-{n_neighbors}_md-{min_dist}_nc-{n_components}_ep-{n_epochs}"
    
    # --- UMAP Embedding ---
    try:
        reducer = umap.UMAP(
            n_neighbors=int(n_neighbors), # Ensure int
            min_dist=float(min_dist),     # Ensure float
            n_components=int(n_components),
            metric=metric,
            n_epochs=int(n_epochs),
            random_state=UMAP_RANDOM_STATE,
            verbose=False
        )
        t0 = time.time()
        embedding = reducer.fit_transform(embeddings_data)
        umap_time = time.time() - t0
    except Exception as e:
        print(f"ERROR during UMAP for {umap_config_name} in {stage_name}: {e}")
        return {**dict(zip(['metric', 'n_neighbors', 'min_dist', 'n_components', 'n_epochs_umap'], config_tuple[0:5])),
                'fold_id': fold_id, 'stage': stage_name, 'umap_config_name': umap_config_name,
                'error': str(e), 'macro_fer': 1.0, 'v_measure': 0.0, 'silhouette': 0.0, 'noise_pct': 1.0,
                'umap_time': 0, 'hdbscan_time':0, 'plot_gt_path': '', 'plot_hdb_path': ''}

    # --- Fast HDBSCAN for Coarse UMAP Evaluation ---
    try:
        clusterer_temp = hdbscan.HDBSCAN(**FAST_HDBSCAN_PARAMS)
        t0_hdb = time.time()
        temp_hdb_labels = clusterer_temp.fit_predict(embedding)
        hdbscan_time = time.time() - t0_hdb
    except Exception as e:
        print(f"ERROR during Fast HDBSCAN for {umap_config_name} in {stage_name}: {e}")
        # Log UMAP success but HDBSCAN failure
        # Plot GT for UMAP
        plot_gt_path = Path(base_plot_path_str) / fold_id / stage_name / f"{umap_config_name}_gt.png"
        plot_umap_embedding(embedding, gt_labels_data, f"{umap_config_name} (GT Labels)", plot_gt_path, is_gt_labels=True, gt_colors_map_dict=gt_colors_map_dict)
        return {**dict(zip(['metric', 'n_neighbors', 'min_dist', 'n_components', 'n_epochs_umap'], config_tuple[0:5])),
                'fold_id': fold_id, 'stage': stage_name, 'umap_config_name': umap_config_name,
                'error': f"HDBSCAN_fast_error: {e}", 'macro_fer': 1.0, 'v_measure': 0.0, 'silhouette': 0.0, 'noise_pct': 1.0,
                'umap_time': umap_time, 'hdbscan_time':0, 'plot_gt_path': str(plot_gt_path), 'plot_hdb_path': ''}


    # --- Metrics ---
    metrics_calculator = UmapEvalMetrics(gt_labels_data, temp_hdb_labels)
    macro_fer = metrics_calculator.calculate_macro_avg_matched_fer()
    v_measure = metrics_calculator.calculate_v_measure()
    silhouette = calculate_silhouette(embedding, temp_hdb_labels)
    noise_pct = calculate_noise_percentage(temp_hdb_labels)

    # --- Plotting ---
    plot_gt_path = Path(base_plot_path_str) / fold_id / stage_name / f"{umap_config_name}_gt.png"
    plot_hdb_path = Path(base_plot_path_str) / fold_id / stage_name / f"{umap_config_name}_hdbscan_temp.png"
    
    plot_umap_embedding(embedding, gt_labels_data, f"{umap_config_name} (GT Labels)\nFold: {fold_id}", plot_gt_path, is_gt_labels=True, gt_colors_map_dict=gt_colors_map_dict)
    plot_umap_embedding(embedding, temp_hdb_labels, f"{umap_config_name} (Fast HDBSCAN)\nFER:{macro_fer:.3f} V-M:{v_measure:.3f} Noise:{noise_pct:.2%}", plot_hdb_path, is_gt_labels=False)
    
    result = {
        'fold_id': fold_id,
        'stage': stage_name,
        'umap_metric': metric,
        'umap_n_neighbors': n_neighbors,
        'umap_min_dist': min_dist,
        'umap_n_components': n_components,
        'umap_n_epochs': n_epochs,
        'hdbscan_params_temp': str(FAST_HDBSCAN_PARAMS),
        'macro_fer': macro_fer,
        'v_measure': v_measure,
        'silhouette': silhouette,
        'noise_pct': noise_pct,
        'umap_time': umap_time,
        'hdbscan_time': hdbscan_time,
        'plot_gt_path': str(plot_gt_path),
        'plot_hdb_path': str(plot_hdb_path),
        'error': None,
        'embedding_ref': embedding # Keep reference for selection if needed, or save path
    }
    return result

def run_single_hdbscan_config(config_tuple, embedding_data, embedding_name, umap_source_params_str,
                               gt_labels_data, gt_colors_map_dict, fold_id, base_plot_path_str, stage_name):
    """
    Runs HDBSCAN for a single configuration on a given embedding, calculates metrics, and plots.
    Used in Stage 1c.
    """
    min_cluster_size, min_samples_val, cluster_selection_method, allow_single_cluster, alpha = config_tuple
    
    hdb_params = {
        'min_cluster_size': int(min_cluster_size),
        'min_samples': int(min_samples_val),
        'cluster_selection_method': cluster_selection_method,
        'allow_single_cluster': allow_single_cluster,
        'alpha': float(alpha)
    }
    hdb_config_name = f"hdb_mcs-{hdb_params['min_cluster_size']}_ms-{hdb_params['min_samples']}_csm-{hdb_params['cluster_selection_method']}_alpha-{hdb_params['alpha']}"
    full_config_name = f"{embedding_name}_{hdb_config_name}"

    try:
        clusterer = hdbscan.HDBSCAN(**hdb_params)
        t0 = time.time()
        hdb_labels = clusterer.fit_predict(embedding_data)
        hdbscan_time = time.time() - t0
    except Exception as e:
        print(f"ERROR during HDBSCAN for {full_config_name} in {stage_name}: {e}")
        return {'fold_id': fold_id, 'stage': stage_name, 'embedding_source': embedding_name, 
                'umap_source_params': umap_source_params_str, 'hdbscan_params': str(hdb_params),
                'error': str(e), 'macro_fer': 1.0, 'v_measure': 0.0, 'silhouette': 0.0, 'noise_pct': 1.0,
                'hdbscan_time':0, 'plot_hdb_path': ''}

    metrics_calculator = UmapEvalMetrics(gt_labels_data, hdb_labels)
    macro_fer = metrics_calculator.calculate_macro_avg_matched_fer()
    v_measure = metrics_calculator.calculate_v_measure()
    silhouette = calculate_silhouette(embedding_data, hdb_labels) # Silhouette on original embedding
    noise_pct = calculate_noise_percentage(hdb_labels)

    plot_hdb_path = Path(base_plot_path_str) / fold_id / stage_name / embedding_name / f"{hdb_config_name}.png"
    plot_title = (f"{embedding_name} ({umap_source_params_str})\n"
                  f"HDBSCAN: {hdb_config_name}\n"
                  f"FER:{macro_fer:.3f} V-M:{v_measure:.3f} Noise:{noise_pct:.2%}")
    
    # Need to plot the original embedding (e.g., U2_refined) colored by these new hdb_labels
    # This requires the UMAP embedding to be passed or re-loaded if only paths are stored.
    # For now, assume embedding_data is the one to plot.
    plot_umap_embedding(embedding_data, hdb_labels, plot_title, plot_hdb_path, is_gt_labels=False)
    
    result = {
        'fold_id': fold_id,
        'stage': stage_name,
        'embedding_source': embedding_name,
        'umap_source_params': umap_source_params_str,
        'hdbscan_params': str(hdb_params),
        'macro_fer': macro_fer,
        'v_measure': v_measure,
        'silhouette': silhouette,
        'noise_pct': noise_pct,
        'hdbscan_time': hdbscan_time,
        'plot_hdb_path': str(plot_hdb_path),
        'error': None,
        'hdbscan_labels_ref': hdb_labels # For final selection
    }
    return result


# --- Main Script ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='umap') # UMAP can be verbose
    warnings.filterwarnings("ignore", category=RuntimeWarning) # Often from matplotlib/seaborn or metrics with no clusters

    # --- Setup ---
    output_dir_fold = OUTPUT_BASE_DIR / CURRENT_FOLD_ID
    plots_dir_fold = output_dir_fold / PLOTS_DIR_NAME
    ensure_dir(output_dir_fold)
    ensure_dir(plots_dir_fold)

    master_log_file = output_dir_fold / "umbra_sweep_results_log.tsv"
    all_results_df = pd.DataFrame()
    
    print(f"Starting Umbra Sweep for Fold: {CURRENT_FOLD_ID}")
    print(f"Output will be saved in: {output_dir_fold}")
    print(f"Master log: {master_log_file}")

    # --- Load Data (Subsampled for sweeps) ---
    # gt_colors_map should be a dictionary: {label_int: [r,g,b,a_float], ...}
    embeddings_100k, gt_labels_100k, gt_colors_map, embeddings_full, gt_labels_full = load_data_for_fold(CURRENT_FOLD_ID)
    
    # Convert gt_labels to int if they are not, for dict keys and consistency
    gt_labels_100k = gt_labels_100k.astype(int)
    gt_labels_full = gt_labels_full.astype(int)


    # === Stage 1a: UMAP Coarse Sweep ===
    print("\n--- Stage 1a: UMAP Coarse Sweep ---")
    stage1a_results_list = []
    umap_coarse_configs = list(itertools.product(
        UMAP_METRICS, UMAP_N_NEIGHBORS_COARSE, UMAP_MIN_DIST_COARSE,
        UMAP_N_COMPONENTS_COARSE, [UMAP_N_EPOCHS_COARSE]
    ))
    print(f"Total UMAP coarse configurations: {len(umap_coarse_configs)}")

    stage1a_results_list = Parallel(n_jobs=N_JOBS)(
        delayed(run_single_umap_config)(
            config, embeddings_100k, gt_labels_100k, gt_colors_map,
            CURRENT_FOLD_ID, str(plots_dir_fold), "Stage1a_UMAP_Coarse"
        ) for config in tqdm(umap_coarse_configs, desc="Stage 1a UMAPs")
    )
    
    stage1a_df = pd.DataFrame([res for res in stage1a_results_list if res is not None])
    all_results_df = pd.concat([all_results_df, stage1a_df], ignore_index=True)
    all_results_df.to_csv(master_log_file, sep='\t', index=False)
    print(f"Stage 1a results logged. Found {len(stage1a_df)} valid results.")

    # Selection for UMAP Refinement
    # Select top 1-2 UMAP (metric, n_neighbors) per n_component based on Macro FER
    # Filter out early rejected UMAPs first
    stage1a_df_valid = stage1a_df[
        (stage1a_df['macro_fer'] <= EARLY_REJECT_UMAP_FER_THRESHOLD) &
        (stage1a_df['noise_pct'] <= EARLY_REJECT_UMAP_NOISE_THRESHOLD) &
        (stage1a_df['error'].isna())
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Store the actual embedding with the result for easy access, or save path
    # For simplicity here, we'll re-generate for refinement or assume it's small enough in memory via 'embedding_ref'
    # In a large scale run, you'd save embeddings and load paths.

    refined_umap_params_to_run = []
    if not stage1a_df_valid.empty:
        for nc in UMAP_N_COMPONENTS_COARSE:
            top_for_nc = stage1a_df_valid[stage1a_df_valid['umap_n_components'] == nc].nsmallest(2, 'macro_fer') # Top 2
            for _, row in top_for_nc.iterrows():
                refined_umap_params_to_run.append({
                    'metric': row['umap_metric'],
                    'n_neighbors': int(row['umap_n_neighbors']),
                    'min_dist': float(row['umap_min_dist']), # Should be 0.0
                    'n_components': int(row['umap_n_components']),
                    'n_epochs': UMAP_N_EPOCHS_REFINED # Use refined epochs
                    # 'coarse_embedding_ref': row['embedding_ref'] # If we carried it over
                })
        # Deduplicate (in case same params were top for different nc, though unlikely here)
        refined_umap_params_to_run = [dict(t) for t in {tuple(d.items()) for d in refined_umap_params_to_run}]
    print(f"Selected {len(refined_umap_params_to_run)} UMAP configurations for refinement.")


    # === Stage 1b: UMAP Refinement ===
    print("\n--- Stage 1b: UMAP Refinement ---")
    stage1b_refined_embeddings = {} # Store as {config_name: embedding_array}
    
    for umap_params_dict in tqdm(refined_umap_params_to_run, desc="Stage 1b UMAP Refinement"):
        metric = umap_params_dict['metric']
        n_neighbors = umap_params_dict['n_neighbors']
        min_dist = umap_params_dict['min_dist']
        n_components = umap_params_dict['n_components']
        n_epochs = umap_params_dict['n_epochs'] # Refined epochs
        
        config_name = f"refined_metric-{metric}_k-{n_neighbors}_md-{min_dist}_nc-{n_components}_ep-{n_epochs}"
        umap_source_params_str = f"metric:{metric},k:{n_neighbors},md:{min_dist},nc:{n_components},ep:{n_epochs}"

        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
                metric=metric, n_epochs=n_epochs, random_state=UMAP_RANDOM_STATE, verbose=False
            )
            t0 = time.time()
            refined_embedding = reducer.fit_transform(embeddings_100k) # Using subsample for now
            ref_time = time.time() - t0
            
            stage1b_refined_embeddings[config_name] = {
                'embedding': refined_embedding, 
                'params_str': umap_source_params_str,
                'n_components': n_components # Store for reference
            }

            plot_path = plots_dir_fold / "Stage1b_UMAP_Refined" / f"{config_name}_gt.png"
            plot_umap_embedding(refined_embedding, gt_labels_100k, f"{config_name} (GT Labels)", plot_path, is_gt_labels=True, gt_colors_map_dict=gt_colors_map)
            
            # Log this refinement step
            log_entry_ref = pd.DataFrame([{
                'fold_id': CURRENT_FOLD_ID, 'stage': 'Stage1b_UMAP_Refined',
                'umap_metric': metric, 'umap_n_neighbors': n_neighbors, 'umap_min_dist': min_dist,
                'umap_n_components': n_components, 'umap_n_epochs': n_epochs,
                'umap_time': ref_time, 'plot_gt_path': str(plot_path), 'error': None
            }])
            all_results_df = pd.concat([all_results_df, log_entry_ref], ignore_index=True)

        except Exception as e:
            print(f"ERROR during UMAP refinement for {config_name}: {e}")
            log_entry_ref_err = pd.DataFrame([{
                'fold_id': CURRENT_FOLD_ID, 'stage': 'Stage1b_UMAP_Refined_Error',
                'umap_metric': metric, 'umap_n_neighbors': n_neighbors, 'umap_min_dist': min_dist,
                'umap_n_components': n_components, 'umap_n_epochs': n_epochs, 'error': str(e)
            }])
            all_results_df = pd.concat([all_results_df, log_entry_ref_err], ignore_index=True)

    all_results_df.to_csv(master_log_file, sep='\t', index=False)
    print(f"Stage 1b UMAP refinement complete. {len(stage1b_refined_embeddings)} embeddings generated.")

    # Prepare other embedding sources for Stage 1c
    other_embeddings_for_hdbscan = {}
    # PCA
    try:
        pca = PCA(n_components=PCA_N_COMPONENTS, whiten=True, random_state=UMAP_RANDOM_STATE)
        pca_embeddings = pca.fit_transform(embeddings_100k)
        other_embeddings_for_hdbscan['PCA20'] = {'embedding': pca_embeddings, 'params_str': f"n_components:{PCA_N_COMPONENTS},whiten:True"}
        plot_pca_path = plots_dir_fold / "Stage1b_UMAP_Refined" / "PCA20_gt.png" # Plot PCA in 2D for viz
        if PCA_N_COMPONENTS > 2: # Need to reduce PCA further for 2D plot
             pca_viz = umap.UMAP(n_components=2, random_state=UMAP_RANDOM_STATE).fit_transform(pca_embeddings)
        else:
             pca_viz = pca_embeddings
        plot_umap_embedding(pca_viz, gt_labels_100k, "PCA20 (GT Labels - UMAP viz)", plot_pca_path, is_gt_labels=True, gt_colors_map_dict=gt_colors_map)

    except Exception as e:
        print(f"ERROR during PCA: {e}")
        # Log PCA error if needed

    # Original Embeddings (ID)
    # For HDBSCAN, if original dim is too high, might be slow or not effective.
    # Here we assume embeddings_100k is the "original" high-dim feature space for clustering
    other_embeddings_for_hdbscan['ID'] = {'embedding': embeddings_100k, 'params_str': f"original_dim:{embeddings_100k.shape[1]}"}
    # Plotting original high-dim data usually requires UMAP reduction for visualization
    try:
        id_viz = umap.UMAP(n_components=2, random_state=UMAP_RANDOM_STATE).fit_transform(embeddings_100k)
        plot_id_path = plots_dir_fold / "Stage1b_UMAP_Refined" / "ID_gt.png"
        plot_umap_embedding(id_viz, gt_labels_100k, "Original Embeddings (ID - UMAP viz for GT)", plot_id_path, is_gt_labels=True, gt_colors_map_dict=gt_colors_map)
    except Exception as e:
         print(f"Error plotting UMAP of original embeddings: {e}")


    # === Stage 1c: Full HDBSCAN Sweep ===
    print("\n--- Stage 1c: Full HDBSCAN Sweep ---")
    stage1c_results_list = []
    
    # Combine refined UMAPs with PCA and ID
    all_embeddings_for_hdbscan = {**stage1b_refined_embeddings, **other_embeddings_for_hdbscan}

    # Generate HDBSCAN parameter combinations
    hdb_min_samples_options = list(HDBSCAN_MIN_SAMPLES_ABS)
    # if HDBSCAN_MIN_SAMPLES_REL_FACTOR:
    #    for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
    #        for factor in HDBSCAN_MIN_SAMPLES_REL_FACTOR:
    #            hdb_min_samples_options.append(max(1, int(mcs * factor))) # Ensure at least 1
    # hdb_min_samples_options = sorted(list(set(hdb_min_samples_options))) # Deduplicate

    hdbscan_configs = list(itertools.product(
        HDBSCAN_MIN_CLUSTER_SIZES,
        hdb_min_samples_options, # Use the combined list
        HDBSCAN_CLUSTER_SELECTION_METHODS,
        [HDBSCAN_ALLOW_SINGLE_CLUSTER],
        HDBSCAN_ALPHAS
    ))
    print(f"Total HDBSCAN configurations per embedding source: {len(hdbscan_configs)}")

    hdbscan_tasks = []
    for emb_name, emb_data_dict in all_embeddings_for_hdbscan.items():
        emb_array = emb_data_dict['embedding']
        emb_params_str = emb_data_dict['params_str']
        for hdb_config_tuple in hdbscan_configs:
            hdbscan_tasks.append(
                delayed(run_single_hdbscan_config)(
                    hdb_config_tuple, emb_array, emb_name, emb_params_str,
                    gt_labels_100k, gt_colors_map, CURRENT_FOLD_ID,
                    str(plots_dir_fold), "Stage1c_HDBSCAN_Sweep"
                )
            )
    
    print(f"Total HDBSCAN runs for Stage 1c: {len(hdbscan_tasks)}")
    stage1c_results_list = Parallel(n_jobs=N_JOBS)(
        task for task in tqdm(hdbscan_tasks, desc="Stage 1c HDBSCANs")
    )

    stage1c_df = pd.DataFrame([res for res in stage1c_results_list if res is not None])
    all_results_df = pd.concat([all_results_df, stage1c_df], ignore_index=True)
    all_results_df.to_csv(master_log_file, sep='\t', index=False)
    print(f"Stage 1c results logged. Found {len(stage1c_df)} valid results.")


    # === Stage 5: Final Selection for the Fold ===
    print("\n--- Stage 5: Final Selection for Fold ---")
    # Filter for Stage1c results, valid (no error), and apply early rejection for HDBSCAN noise
    final_selection_candidates = all_results_df[
        (all_results_df['stage'] == 'Stage1c_HDBSCAN_Sweep') &
        (all_results_df['error'].isna()) &
        (all_results_df['noise_pct'] <= EARLY_REJECT_HDBSCAN_NOISE_THRESHOLD)
    ].copy()

    if not final_selection_candidates.empty:
        best_config_for_fold = final_selection_candidates.loc[final_selection_candidates['macro_fer'].idxmin()]
        
        print("\nBest Configuration for Fold:")
        print(best_config_for_fold)
        
        # Here you would typically take `best_config_for_fold` and run it on `embeddings_full`
        # and `gt_labels_full` if they are different from the subsampled versions.
        # The 'hdbscan_labels_ref' from the best config (which was on subsampled data)
        # would be the proxy labels if you stop here.
        # If running on full data:
        # 1. Get best UMAP params (from best_config_for_fold['umap_source_params'])
        # 2. Run UMAP on embeddings_full
        # 3. Get best HDBSCAN params (from best_config_for_fold['hdbscan_params'])
        # 4. Run HDBSCAN on the full_data_embedding
        # 5. These HDBSCAN labels are your proxy_labels_for_stage2

        print("\n(Placeholder for running best config on full data and saving proxy labels)")
        # Example: proxy_labels = best_config_for_fold['hdbscan_labels_ref'] (if using subsample labels)
        # np.save(output_dir_fold / "proxy_labels_stage1.npy", proxy_labels)

    else:
        print("No suitable configuration found after Stage 1c for this fold.")

    print(f"\nUmbra Sweep for Fold {CURRENT_FOLD_ID} complete.")
    print(f"All results logged to: {master_log_file}")
    print(f"Plots saved in subdirectories of: {plots_dir_fold}")

