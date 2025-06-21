import time
import numpy as np
import cupy as cp
import cuml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import v_measure_score
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter, defaultdict
import os
import pathlib
import itertools
import pandas as pd
import gc # For garbage collection

# =============================================================================
# START OF CONFIGURATION PARAMETERS
# =============================================================================

base_path = "/media/george-vengrovski/Desk SSD/TweetyBERT_Zenedo/LLB_Fold_Data"

ALL_AVAILABLE_FOLD_PATHS_STR = [
    f"{base_path}/llb3_fold1.npz",
    f"{base_path}/llb3_fold2.npz",
    f"{base_path}/llb3_fold3.npz",
    f"{base_path}/llb3_fold4.npz",
    f"{base_path}/llb11_fold1.npz",
    f"{base_path}/llb11_fold2.npz",
    f"{base_path}/llb11_fold3.npz",
    f"{base_path}/llb16_fold1.npz",
    f"{base_path}/llb16_fold2.npz",
]

# --- Test Set Definition ---
# IMPORTANT: Specify one fold from each bird type to be EXCLUDED from HPO
TEST_SET_FILES_STR_MAP = {
    'llb3': "files/llb3_fold5.npz",
    'llb11': "files/llb11_fold4.npz",
    'llb16': "files/llb16_fold3.npz",
}

N_DATA_POINTS = 1_000_000
SILENCE_LABEL_VALUE = 0

OUTPUT_BASE_DIR = pathlib.Path("./full_search_V1_1M")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- Parameter Grids (Core config for pruning = UMAP + HDBSCAN + Smoothing) ---
UMAP_PARAM_GRID_CONFIG = {
    'n_components': [2, 8, 32],      # As discussed
    'n_neighbors': [15, 50, 150],    # As discussed
    'min_dist': [0.1, 0.25],
    'metric': ["cosine"],  # euclidean on raw data, cosine via unit-norm + euclidean
    'random_state': [42],  # Added to ensure brute_force_knn algorithm is used consistently
}
HDBSCAN_PARAM_GRID_CONFIG = {
    'min_cluster_size': [500, 2500, 5000],
    'min_samples': [5, 50]
}
SMOOTHING_PARAM_GRID_CONFIG = {
    'smoothing_window': [0]
}

# --- Search Strategy Parameters ---
OPTIMIZATION_METRIC = 'total_fer' 
OPTIMIZATION_DIRECTION = 'minimize' 

# --- Timing Estimation ---
DEFAULT_AVG_T_UMAP = 180  # Estimated seconds for 1M points, will be updated
DEFAULT_AVG_T_HDBSCAN = 60 # Estimated seconds, will be updated
DEFAULT_AVG_T_SMOOTHING_BLOCK = 30 # Estimated seconds for all smoothing windows, will be updated

# --- UMAP build_kwds for brute_force_knn ---
UMAP_BUILD_KWDS = {
    "nnd_graph_degree": 256,
    "nnd_intermediate_graph_degree": 256,
    "build_algo": "brute_force_knn"
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions from Script 1 (for pre-processing)
# ─────────────────────────────────────────────────────────────────────────────

def fill_noise_with_nearest_label(labels):
    """
    For each noise point (labeled -1), find the nearest non-noise
    label to the left or right and assign it to this point.
    """
    labels = labels.copy() # Work on a copy
    noise_indices = np.where(labels == -1)[0]
    non_noise_indices = np.where(labels != -1)[0]

    if len(non_noise_indices) == 0:
        return labels # All noise, nothing to do

    for idx in noise_indices:
        # Find distances to all non-noise points
        distances = np.abs(non_noise_indices - idx)
        nearest_non_noise_idx = non_noise_indices[np.argmin(distances)]
        labels[idx] = labels[nearest_non_noise_idx]
    return labels

def syllable_to_phrase_labels(arr, silence=0):
    """
    Convert a sequence of syllable labels into a sequence of phrase labels,
    merging silence bins with their nearest adjacent syllables.
    """
    new_arr = np.array(arr, dtype=int)
    length = len(new_arr)
    if length == 0:
        return new_arr

    def find_silence_runs(labels):
        runs = []
        in_silence = False
        start = None
        for i, val in enumerate(labels):
            if val == silence and not in_silence:
                in_silence = True
                start = i
            elif val != silence and in_silence:
                runs.append((start, i - 1))
                in_silence = False
        if in_silence:
            runs.append((start, length - 1))
        return runs

    silence_runs = find_silence_runs(new_arr)

    for start_idx, end_idx in silence_runs:
        left_label = new_arr[start_idx - 1] if start_idx > 0 else None
        right_label = new_arr[end_idx + 1] if end_idx < length - 1 else None

        if left_label is None and right_label is None:
            continue
        elif left_label is None:
            new_arr[start_idx:end_idx+1] = right_label
        elif right_label is None:
            new_arr[start_idx:end_idx+1] = left_label
        elif left_label == right_label:
            new_arr[start_idx:end_idx+1] = left_label
        else:
            for i in range(start_idx, end_idx + 1):
                dist_left = i - (start_idx - 1)
                dist_right = (end_idx + 1) - i
                if dist_left <= dist_right: # Tie goes left
                    new_arr[i] = left_label
                else:
                    new_arr[i] = right_label
    return new_arr

def majority_vote(data, window_size=1):
    """
    Return an array of the same length as 'data',
    where each index i is replaced by the majority over
    a window around i. No padding is added.
    """
    data = np.asarray(data)
    n = len(data)
    if window_size <= 1 or n == 0:
        return data.copy()
    
    half_w = window_size // 2
    output = np.zeros_like(data)
    
    for i in range(n):
        start = max(0, i - half_w)
        end   = min(n, i + half_w + 1)
        window = data[start:end]
        c = Counter(window)
        major_label = max(c, key=c.get)
        output[i] = major_label
    return output

# =============================================================================
# END OF CONFIGURATION PARAMETERS
# =============================================================================

class Script2Evaluator:
    """
    Evaluates clustering using the exact algorithm from Script 2.
    - Ensures -1 is in confusion matrix.
    - Maps unmapped GT labels to -1.
    - Re-labels unmapped predicted clusters to -1 before final FER calculation.
    """
    def __init__(self, gt: np.ndarray, pred: np.ndarray):
        # Truncate arrays to same length, same as Script 1's __init__
        if gt.shape != pred.shape:
            min_len = min(len(gt), len(pred))
            if abs(len(gt) - len(pred)) > 100:
                raise ValueError(f"gt (shape {gt.shape}) and pred (shape {pred.shape}) arrays must have identical shape.")
            print(f"Warning: GT and Pred shapes differ ({gt.shape} vs {pred.shape}). Truncating to {min_len}")
            gt = gt[:min_len]
            pred = pred[:min_len]

        self.gt_raw = gt.astype(int)
        self.pred_raw = pred.astype(int)

        # --- Core evaluation logic from Script 2 ---
        M_norm, unique_gt, unique_pred = self._create_shared_area_matrix(self.gt_raw, self.pred_raw)
        
        mapping, leftover_pred = self._create_diagonal_label_mapping(M_norm, unique_gt, unique_pred)

        # Create the "cleaned" prediction array where leftover predictions are set to -1
        self.pred_cleaned = self.pred_raw.copy()
        for lv in leftover_pred:
            self.pred_cleaned[self.pred_cleaned == lv] = -1
        
        # Create the mapped_gt array where unmapped GTs are -1
        self.mapped_gt = np.array([mapping[g] for g in self.gt_raw])

    def _create_shared_area_matrix(self, ground_truth: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds and column-normalizes the confusion matrix, ensuring -1 is present."""
        # Ensure -1 is in both sets for the matrix construction
        unique_gt = np.unique(np.concatenate([ground_truth, [-1]]))
        unique_pred = np.unique(np.concatenate([predicted, [-1]]))
        
        gt_map = {label: i for i, label in enumerate(unique_gt)}
        pred_map = {label: i for i, label in enumerate(unique_pred)}

        M = np.zeros((len(unique_gt), len(unique_pred)), dtype=int)
        for g_val, p_val in zip(ground_truth, predicted):
            M[gt_map[g_val], pred_map[p_val]] += 1
            
        col_sums = M.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            M_norm = np.divide(M, col_sums, where=col_sums != 0, out=np.zeros_like(M, dtype=float))
        
        return M_norm, unique_gt, unique_pred

    def _create_diagonal_label_mapping(self, normalized_matrix: np.ndarray, unique_gt_labels: np.ndarray, unique_pred_labels: np.ndarray) -> Tuple[Dict[int, int], List[int]]:
        """Uses Hungarian algorithm and maps unmapped GT to -1."""
        cost_matrix = -normalized_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = {}
        matched_gt = set()
        matched_pred = set()
        
        for r, c in zip(row_ind, col_ind):
            gt_lbl = int(unique_gt_labels[r])
            pd_lbl = int(unique_pred_labels[c])
            # Don't create a mapping for the explicit -1 row/col
            if gt_lbl != -1 and pd_lbl != -1:
                mapping[gt_lbl] = pd_lbl
                matched_gt.add(gt_lbl)
                matched_pred.add(pd_lbl)

        # Any GT label not matched (including -1) gets mapped to -1
        for g in unique_gt_labels:
            g = int(g)
            if g not in matched_gt:
                mapping[g] = -1
        
        leftover_pred = sorted(set(int(x) for x in unique_pred_labels) - matched_pred)
        
        return mapping, leftover_pred

    def total_fer(self) -> float:
        """FER with -1 err. Equivalent to `overall_fer_any`."""
        if self.mapped_gt.size == 0:
            return {}
        mismatch_any = np.sum(self.mapped_gt != self.pred_cleaned)
        return 100.0 * (mismatch_any / self.mapped_gt.size)

    def v_measure(self) -> float:
        """V-measure score, using the cleaned prediction array."""
        if self.gt_raw.size == 0 or self.pred_cleaned.size == 0:
            return 0.0
        try:
            # Score the original GT against the prediction array that has had junk clusters removed
            return v_measure_score(self.gt_raw, self.pred_cleaned)
        except ValueError:
            return 0.0

    def matched_fer(self) -> float:
        """Mapped-Only FER. Equivalent to `overall_fer_mapped`."""
        if self.mapped_gt.size == 0:
            return 0.0
        
        mapped_mask = (self.mapped_gt != -1)
        frames_mapped = np.sum(mapped_mask)
        if frames_mapped == 0:
            return 0.0
        
        mismatch_mapped = np.sum(self.mapped_gt[mapped_mask] != self.pred_cleaned[mapped_mask])
        return 100.0 * (mismatch_mapped / frames_mapped)

    def macro_fer(self)->float:
        """Calculates macro-averaged frame error rate."""
        unique_gt_types = np.unique(self.gt_raw)
        per_type_fer = []

        for gt_label_type in unique_gt_types:
            if gt_label_type == -1: # Typically -1 GT is not evaluated in macro average
                continue
            
            type_mask = (self.gt_raw == gt_label_type)
            total_for_type = np.sum(type_mask)
            
            if total_for_type == 0:
                continue
            
            # Use self.mapped_gt and self.pred_cleaned which have the full logic applied
            errors_for_type = np.sum(self.mapped_gt[type_mask] != self.pred_cleaned[type_mask])
            per_type_fer.append(errors_for_type / total_for_type)
        
        return 100.0 * np.mean(per_type_fer) if per_type_fer else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Helper function to generate a hashable key from a config dictionary
# ─────────────────────────────────────────────────────────────────────────────
def config_to_key(config_dict: Dict[str, Any], param_names: List[str]) -> Tuple:
    # Ensure all expected param_names are present in config_dict before creating key
    return tuple(sorted((k, config_dict.get(k)) for k in param_names))


# ─────────────────────────────────────────────────────────────────────────────
# Function to save intermediate results
# ─────────────────────────────────────────────────────────────────────────────
def save_results_to_csv(results_list: List[Dict], file_path: pathlib.Path, write_header: bool):
    if not results_list: return
    
    # Determine full set of keys for header from the first comprehensive record if possible
    # This is to ensure consistent CSV structure even if some error records have fewer keys.
    all_possible_keys = set()
    for item in results_list:
        all_possible_keys.update(item.keys())
    
    # Define a preferred order, adding any other keys at the end
    preferred_header_order = ['fold_path_str', 'config_str_readable'] + \
                             list(UMAP_PARAM_GRID_CONFIG.keys()) + \
                             list(HDBSCAN_PARAM_GRID_CONFIG.keys()) + \
                             ['smoothing_window'] + \
                             [OPTIMIZATION_METRIC, 'v_measure', 'total_fer', 
                              'matched_fer', 'macro_fer', 'n_gt_types', 'n_pred_types', 
                              'pct_types_mapped', 'pct_frames_mapped', 'time_umap', 
                              'time_hdbscan', 'time_eval_block_all_smoothing', 
                              'oom_flag_umap', 'oom_flag_hdbscan', 'error_message']
    
    final_header = [k for k in preferred_header_order if k in all_possible_keys]
    for k in sorted(list(all_possible_keys)): # Add any remaining keys
        if k not in final_header:
            final_header.append(k)

    df_to_save = pd.DataFrame(results_list)[final_header] # Ensure column order

    try:
        df_to_save.to_csv(file_path, index=False, header=write_header, mode='a' if not write_header else 'w')
        # print(f"    Appended/Wrote {len(results_list)} results to {file_path}")
    except Exception as e_csv:
        print(f"    ERROR saving results to CSV {file_path}: {e_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Full Grid Search Script Logic
# ─────────────────────────────────────────────────────────────────────────────

# --- Timing Accumulators ---
cumulative_umap_time = 0.0; num_umap_runs_timed = 0
cumulative_hdbscan_time = 0.0; num_hdbscan_runs_timed = 0
cumulative_eval_block_time = 0.0; num_eval_blocks_timed = 0

# Convert string paths to Path objects and check existence
all_available_path_objects = [pathlib.Path(f) for f in ALL_AVAILABLE_FOLD_PATHS_STR]
existing_available_files = [f for f in all_available_path_objects if f.exists()] 

if not existing_available_files:
    print("CRITICAL ERROR: No files from ALL_AVAILABLE_FOLD_PATHS_STR exist. Please check paths.")
    exit()

# --- Test Set Definition ---
test_set_path_objects = {pathlib.Path(f) for f in TEST_SET_FILES_STR_MAP.values()}
test_set_paths = {f for f in test_set_path_objects if f.exists() and f in existing_available_files}

# --- Prepare Fold Lists ---
hpo_folds_all = [f for f in existing_available_files if f not in test_set_paths]

if not hpo_folds_all:
    print("CRITICAL ERROR: No HPO folds available after excluding test set. Exiting.")
    exit()

# Organize HPO folds by bird type for balanced selection
hpo_folds_by_bird = defaultdict(list)
for f_path in hpo_folds_all:
    if 'llb3' in f_path.name: hpo_folds_by_bird['llb3'].append(f_path)
    elif 'llb16' in f_path.name: hpo_folds_by_bird['llb16'].append(f_path)
    elif 'llb11' in f_path.name: hpo_folds_by_bird['llb11'].append(f_path)
    # Add more bird types if necessary

BIRD_TYPES_FOR_HPO = sorted(list(hpo_folds_by_bird.keys()))
if not BIRD_TYPES_FOR_HPO:
    print("CRITICAL ERROR: No bird types found in HPO folds. Exiting.")
    exit()

print(f"HPO folds by bird type: {{ {', '.join([f'{bt}: {len(folds)}' for bt, folds in hpo_folds_by_bird.items()])} }}")
print(f"Total HPO folds: {len(hpo_folds_all)}")
print(f"Test set files (excluded from HPO): {[f.name for f in test_set_paths]}")

# 1. Generate CORE parameter configurations (UMAP + HDBSCAN only, smoothing applied separately)
umap_param_names = list(UMAP_PARAM_GRID_CONFIG.keys())
umap_param_value_lists = list(UMAP_PARAM_GRID_CONFIG.values())
hdbscan_param_names = list(HDBSCAN_PARAM_GRID_CONFIG.keys())
hdbscan_param_value_lists = list(HDBSCAN_PARAM_GRID_CONFIG.values())
smoothing_param_names = list(SMOOTHING_PARAM_GRID_CONFIG.keys())
smoothing_param_value_lists = list(SMOOTHING_PARAM_GRID_CONFIG.values())

# Generate UMAP+HDBSCAN combinations (without smoothing)
initial_core_configs_dicts = []
for umap_vals in itertools.product(*umap_param_value_lists):
    umap_dict = {k:v for k,v in zip(umap_param_names, umap_vals)}
    for hdbscan_vals in itertools.product(*hdbscan_param_value_lists):
        hdbscan_dict = {k:v for k,v in zip(hdbscan_param_names, hdbscan_vals)}
        initial_core_configs_dicts.append({**umap_dict, **hdbscan_dict})

# Generate all smoothing window values separately
smoothing_windows = [dict(zip(smoothing_param_names, vals)) for vals in itertools.product(*smoothing_param_value_lists)]

CORE_CONFIG_PARAM_NAMES = umap_param_names + hdbscan_param_names  # Only UMAP+HDBSCAN for core configs
initial_num_core_configs = len(initial_core_configs_dicts)
total_results_per_fold = initial_num_core_configs * len(smoothing_windows) # Total rows per fold
print(f"Generated {initial_num_core_configs} CORE (UMAP+HDBSCAN) configurations.")
print(f"Will evaluate {len(smoothing_windows)} smoothing windows for each successful clustering: {[sw['smoothing_window'] for sw in smoothing_windows]}")
print(f"Total results per fold: {initial_num_core_configs} configs × {len(smoothing_windows)} smoothing = {total_results_per_fold} results")

# --- CSV Checkpointing Setup ---
checkpoint_csv_path = OUTPUT_BASE_DIR / f"full_grid_search_ALL_RESULTS_{N_DATA_POINTS//1000}k.csv"
has_csv_header_been_written = False

# --- Resume Logic: Read existing CSV to know what's already been evaluated ---
already_completed_keys = set()
if checkpoint_csv_path.exists():
    print(f"Found existing results CSV: {checkpoint_csv_path}. Attempting to resume from previous run...")
    try:
        if checkpoint_csv_path.stat().st_size > 0:
            has_csv_header_been_written = True
            existing_df = pd.read_csv(checkpoint_csv_path)
            print(f"  Loaded {len(existing_df)} existing results from CSV.")
            # Rebuild the set of completed (fold, config) keys
            full_config_param_names = CORE_CONFIG_PARAM_NAMES + smoothing_param_names
            for _, row in existing_df.iterrows():
                try:
                    fold_path_str = str(row['fold_path_str'])
                    config_dict = {k: row[k] for k in full_config_param_names if k in row and pd.notna(row[k])}
                    if len(config_dict) == len(full_config_param_names): # Is it a full config record?
                        full_config_key = config_to_key(config_dict, full_config_param_names)
                        already_completed_keys.add((fold_path_str, full_config_key))
                except Exception as e_resume:
                    continue
            print(f"  Identified {len(already_completed_keys)} completed (fold, config) combinations to skip.")
    except Exception as e_csv_read:
        print(f"  Warning: Could not read existing CSV for resume ({e_csv_read}). Starting fresh.")
        has_csv_header_been_written = False
else:
    print(f"No existing results CSV found. Starting fresh run.")

# --- Main Loop: Full Grid Search over all HPO folds ---
for fold_idx, data_fpath in enumerate(hpo_folds_all):
    print(f"\n===== Processing HPO Fold {fold_idx + 1}/{len(hpo_folds_all)}: {data_fpath.name} =====")
    X_gpu_file_for_current_fold = None; current_fold_results_buffer = []
    fold_results_saved_count = 0  # Track results saved for this fold
    try:
        data=np.load(data_fpath); X_full=data["predictions"]; gt_full=data["ground_truth_labels"]
        n_pts=min(N_DATA_POINTS,X_full.shape[0]); X_hd=X_full[:n_pts].astype(np.float32,copy=False); gt_eval=gt_full[:n_pts]
        
        # Pre-process GT labels according to Script 1 logic
        gt_processed = syllable_to_phrase_labels(gt_eval, silence=SILENCE_LABEL_VALUE)

        X_gpu_file_for_current_fold=cp.asarray(X_hd)

        # Group configs by UMAP parameters
        umap_groups = defaultdict(list)
        for core_config_dict in initial_core_configs_dicts:
            umap_params = {k: v for k, v in core_config_dict.items() if k in UMAP_PARAM_GRID_CONFIG}
            umap_key = config_to_key(umap_params, list(UMAP_PARAM_GRID_CONFIG.keys()))
            umap_groups[umap_key].append(core_config_dict)

        num_total_configs_for_fold = initial_num_core_configs
        num_unique_umap_configs = len(umap_groups)
        print(f"        Processing {num_total_configs_for_fold} configs with {num_unique_umap_configs} unique UMAP configurations")

        config_idx = 0  # Track overall progress

        # Process each unique UMAP configuration
        for umap_idx, (umap_key, core_configs_for_this_umap) in enumerate(umap_groups.items()):
            umap_params_dict = dict(umap_key)
            print(f"\n      UMAP Config {umap_idx+1}/{num_unique_umap_configs}: {umap_params_dict}")

            # Prepare data based on metric: normalize for cosine, use raw for euclidean
            metric = umap_params_dict['metric']
            if metric == 'cosine':
                # Use unit-normalized data with euclidean metric (equivalent to cosine)
                norms = cp.linalg.norm(X_gpu_file_for_current_fold, axis=1, keepdims=True)
                norms = cp.where(norms == 0, 1, norms)  # Avoid division by zero
                X_for_umap = X_gpu_file_for_current_fold / norms
                umap_params_adjusted = umap_params_dict.copy()
                umap_params_adjusted['metric'] = 'euclidean'  # Use euclidean on normalized data
            else:
                # Use raw data with euclidean metric
                X_for_umap = X_gpu_file_for_current_fold
                umap_params_adjusted = umap_params_dict.copy()

            # Run UMAP once for this parameter set
            t_umap = float('nan')
            emb_gpu = None
            oom_umap_flag = False
            umap_error_msg = None

            try:
                n_neighbors_val = umap_params_adjusted['n_neighbors']
                print(f"        Running UMAP with n_neighbors={n_neighbors_val}")

                umap_model = cuml.UMAP(**umap_params_adjusted, init="spectral", n_epochs=200, build_kwds=UMAP_BUILD_KWDS)
                t_s=time.time(); emb_gpu=umap_model.fit_transform(X_for_umap); t_umap=time.time()-t_s
                cumulative_umap_time += t_umap; num_umap_runs_timed += 1
                del umap_model; gc.collect()
                print(f"        UMAP completed in {t_umap:.1f}s")
            except Exception as e_umap:
                error_str=str(e_umap); oom_umap_flag="out_of_memory" in error_str.lower() or "bad_alloc" in error_str.lower()
                umap_error_msg = f"UMAP Error{' (OOM)' if oom_umap_flag else ''}: {error_str}"
                print(f"        {umap_error_msg}")
                if 'umap_model' in locals(): del umap_model; gc.collect()
                if emb_gpu is not None: del emb_gpu; cp.get_default_memory_pool().free_all_blocks()
                emb_gpu = None

            # Process all HDBSCAN configs that use this UMAP embedding
            for core_config_dict in core_configs_for_this_umap:
                config_idx += 1
                current_hdbscan_params = {k:v for k,v in core_config_dict.items() if k in HDBSCAN_PARAM_GRID_CONFIG}

                print(f"\n        Processing Core Config {config_idx}/{num_total_configs_for_fold}")
                print(f"          HDBSCAN: {current_hdbscan_params}")

                t_hdb, eval_block_duration_all_smooth = float('nan'), float('nan')
                hdb_labels_np = None
                oom_hdbscan_flag = False
                current_config_error_msg = umap_error_msg  # Inherit UMAP error if any

                if current_config_error_msg is None and emb_gpu is not None:
                    try: # HDBSCAN
                        hdb_model=cuml.HDBSCAN(**current_hdbscan_params, metric='euclidean',prediction_data=False)
                        t_s=time.time(); hdb_labels_gpu=hdb_model.fit_predict(emb_gpu); hdb_labels_np=cp.asnumpy(hdb_labels_gpu); t_hdb=time.time()-t_s
                        del hdb_labels_gpu
                        cumulative_hdbscan_time += t_hdb; num_hdbscan_runs_timed += 1
                        del hdb_model; gc.collect()
                    except Exception as e_hdb:
                        error_str=str(e_hdb); oom_hdbscan_flag="out_of_memory" in error_str.lower() or "bad_alloc" in error_str.lower()
                        current_config_error_msg = f"HDBSCAN Error{' (OOM)' if oom_hdbscan_flag else ''}: {error_str}"
                        print(f"          {current_config_error_msg}")
                        if 'hdb_model' in locals(): del hdb_model; gc.collect()

                # --- Evaluate ALL smoothing windows for this UMAP+HDBSCAN combination ---
                if current_config_error_msg is None and hdb_labels_np is not None:
                    eval_block_start_time = time.time()
                    for smoothing_config in smoothing_windows:
                        sm_win_val = smoothing_config['smoothing_window']

                        # RESUME LOGIC: Check if this exact combination has been completed
                        full_config_dict = {**core_config_dict, **smoothing_config}
                        full_config_param_names = list(full_config_dict.keys())
                        full_config_key = config_to_key(full_config_dict, full_config_param_names)
                        if (str(data_fpath), full_config_key) in already_completed_keys:
                            print(f"          - Smoothing {sm_win_val}: SKIPPING (already completed in previous run)")
                            fold_results_saved_count +=1 # Count as "saved" for progress tracking
                            continue
                        
                        # --- New Pre-processing Pipeline ---
                        # 1. Fill noise (-1) from HDBSCAN output
                        pred_filled = fill_noise_with_nearest_label(hdb_labels_np)
                        # 2. Apply smoothing
                        smoothed_preds = majority_vote(pred_filled, sm_win_val) if sm_win_val > 0 else pred_filled.copy()
                        
                        # 3. Use the new Script2Evaluator class
                        cm = Script2Evaluator(gt=gt_processed, pred=smoothed_preds)
                        
                        opt_metric_val = getattr(cm, OPTIMIZATION_METRIC.lower().replace(" ", "_"))()

                        # Create result record
                        result_record = {
                            "fold_path_str": str(data_fpath), **full_config_dict,
                            OPTIMIZATION_METRIC: opt_metric_val,
                            "v_measure": cm.v_measure(), "total_fer": cm.total_fer(), "matched_fer": cm.matched_fer(), "macro_fer": cm.macro_fer(),
                            # Note: Script2Evaluator does not have a .stats() method, so these are removed.
                            # "n_gt_types":stats['n_gt_types'], "n_pred_clusters":stats['n_pred_types'], "pct_types_mapped":stats['pct_types_mapped'], "pct_frames_mapped":stats['pct_frames_mapped'],
                            "time_umap":t_umap, "time_hdbscan":t_hdb, "time_eval_block_all_smoothing":float('nan'),
                            "oom_flag_umap":oom_umap_flag, "oom_flag_hdbscan":oom_hdbscan_flag, "error_message": None }

                        print(f"          ✓ Smoothing {sm_win_val}: {OPTIMIZATION_METRIC}={opt_metric_val:.4f}, v_measure={cm.v_measure():.4f}")
                        current_fold_results_buffer.append(result_record)

                    eval_block_duration_all_smooth = time.time() - eval_block_start_time
                    for i in range(len(smoothing_windows)):
                        current_fold_results_buffer[-(i+1)]['time_eval_block_all_smoothing'] = eval_block_duration_all_smooth

                    # Save all smoothing results for this config immediately
                    save_results_to_csv(current_fold_results_buffer[-len(smoothing_windows):], checkpoint_csv_path, not has_csv_header_been_written)
                    has_csv_header_been_written = True
                    fold_results_saved_count += len(smoothing_windows)
                    cumulative_eval_block_time += eval_block_duration_all_smooth; num_eval_blocks_timed += 1

                else: # UMAP or HDBSCAN error, or HDBSCAN produced no labels
                    if hdb_labels_np is None and current_config_error_msg is None : current_config_error_msg = "HDBSCAN produced no labels"
                    error_results = []
                    for smoothing_config in smoothing_windows:
                        # RESUME LOGIC for errors
                        full_config_dict = {**core_config_dict, **smoothing_config}
                        full_config_param_names = list(full_config_dict.keys())
                        full_config_key = config_to_key(full_config_dict, full_config_param_names)
                        if (str(data_fpath), full_config_key) in already_completed_keys:
                            print(f"          - Smoothing {smoothing_config['smoothing_window']}: SKIPPING ERROR (already logged)")
                            continue

                        error_record = {
                            "fold_path_str": str(data_fpath), **full_config_dict,
                            OPTIMIZATION_METRIC: float('inf') if OPTIMIZATION_DIRECTION == 'minimize' else float('-inf'),
                            "v_measure": float('nan'), "total_fer": float('nan'), "matched_fer": float('nan'), "macro_fer": float('nan'),
                            "n_gt_types":float('nan'), "n_pred_clusters":float('nan'), "pct_types_mapped":float('nan'), "pct_frames_mapped":float('nan'),
                            "time_umap":t_umap, "time_hdbscan":t_hdb, "time_eval_block_all_smoothing":float('nan'),
                            "oom_flag_umap":oom_umap_flag, "oom_flag_hdbscan":oom_hdbscan_flag, "error_message": current_config_error_msg }

                        print(f"          ✗ Smoothing {smoothing_config['smoothing_window']}: ERROR - {current_config_error_msg}")
                        error_results.append(error_record)
                        current_fold_results_buffer.append(error_record)

                    if error_results:
                        save_results_to_csv(error_results, checkpoint_csv_path, not has_csv_header_been_written)
                        has_csv_header_been_written = True
                        fold_results_saved_count += len(error_results)

            # Clean up UMAP embedding after processing all HDBSCAN configs for this UMAP
            if emb_gpu is not None: 
                del emb_gpu; cp.get_default_memory_pool().free_all_blocks()
                emb_gpu = None

            configs_in_this_group = len(core_configs_for_this_umap)
            results_in_this_group = configs_in_this_group * len(smoothing_windows)
            print(f"        ✓ UMAP group completed: {configs_in_this_group} configs × {len(smoothing_windows)} smoothing = {results_in_this_group} results logged")
            print(f"        Progress for {data_fpath.name}: {fold_results_saved_count}/{total_results_per_fold} results logged")

            if fold_results_saved_count > 0 and fold_results_saved_count < total_results_per_fold:
                avg_t_u_current = (cumulative_umap_time / num_umap_runs_timed) if num_umap_runs_timed > 0 else DEFAULT_AVG_T_UMAP
                avg_t_h_current = (cumulative_hdbscan_time / num_hdbscan_runs_timed) if num_hdbscan_runs_timed > 0 else DEFAULT_AVG_T_HDBSCAN
                avg_t_s_current = (cumulative_eval_block_time / num_eval_blocks_timed) if num_eval_blocks_timed > 0 else DEFAULT_AVG_T_SMOOTHING_BLOCK

                remaining_umap_groups = len(umap_groups) - (umap_idx + 1)
                remaining_configs_this_fold = num_total_configs_for_fold - config_idx

                est_time_remaining_this_fold_sec = remaining_umap_groups * avg_t_u_current + remaining_configs_this_fold * (avg_t_h_current + avg_t_s_current)
                print(f"        Est. time remaining for this fold: {est_time_remaining_this_fold_sec / 60:.1f} mins")

    finally:
        if X_gpu_file_for_current_fold is not None: del X_gpu_file_for_current_fold; X_gpu_file_for_current_fold = None
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        current_fold_results_buffer = []
        print(f"    ✓ Fold {data_fpath.name} completed: {fold_results_saved_count} total results saved to CSV")

print(f"\n===== Full grid search process complete. All collected results saved to: {checkpoint_csv_path} =====")

# --- Final Ranking Display (load from the comprehensive CSV) ---
if checkpoint_csv_path.exists() and checkpoint_csv_path.stat().st_size > 0:
    print("\n--- Ranking Configurations Based on Mean Performance Across All HPO Folds ---")
    final_df = pd.read_csv(checkpoint_csv_path)
    
    # Filter for valid, numeric results
    final_df[OPTIMIZATION_METRIC] = pd.to_numeric(final_df[OPTIMIZATION_METRIC], errors='coerce')
    valid_results_df = final_df.dropna(subset=[OPTIMIZATION_METRIC]).copy()
    valid_results_df = valid_results_df[~valid_results_df[OPTIMIZATION_METRIC].isin([np.inf, -np.inf])]

    if not valid_results_df.empty:
        config_cols = CORE_CONFIG_PARAM_NAMES + smoothing_param_names
        # Group by all parameter columns to get a unique key for each configuration
        ranking = valid_results_df.groupby(config_cols)[OPTIMIZATION_METRIC].agg(['mean', 'std', 'count']).reset_index()
        ranking = ranking.sort_values(by='mean', ascending=(OPTIMIZATION_DIRECTION == 'minimize'))

        print(f"Top 10 configurations ranked by mean '{OPTIMIZATION_METRIC}':")
        print(ranking.head(10).to_string())

        # Save the final ranking to its own CSV
        ranking_csv_path = OUTPUT_BASE_DIR / f"full_grid_search_RANKING_{N_DATA_POINTS//1000}k.csv"
        ranking.to_csv(ranking_csv_path, index=False)
        print(f"\nFull ranking saved to: {ranking_csv_path}")
    else:
        print("No valid results found in the CSV to generate a final ranking.")
else:
    print("Results CSV not found or is empty. Cannot generate final ranking.")