import time
import numpy as np
import cupy as cp
import cuml
# import matplotlib.pyplot as plt # Only needed if ClusteringMetrics.plot is called
# from matplotlib.gridspec import GridSpec # Only needed if ClusteringMetrics.plot is called
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

# --- File Processing ---
ALL_AVAILABLE_FOLD_PATHS_STR = [
    # IMPORTANT: Populate this with your *actual* 18 file paths
    "files/llb3_fold2.npz",
    "files/llb3_fold4.npz",
    "files/llb3_fold1.npz",
    "files/llb3_fold3.npz",
    "files/llb3_fold5.npz",
    "files/llb16_fold2.npz",
    "files/llb16_fold4.npz",
    "files/llb16_fold1.npz",
    "files/llb16_fold3.npz",
    "files/llb16_fold5.npz",
    "files/llb11_fold2.npz",
    "files/llb11_fold4.npz",
    "files/llb11_fold1.npz",
    "files/llb11_fold3.npz",
    "files/llb11_fold5.npz",
]

# --- Test Set Definition ---
# IMPORTANT: Specify one fold from each bird type to be EXCLUDED from HPO
TEST_SET_FILES_STR_MAP = {
    'llb3': "files/llb3_fold5.npz",   # Example, choose your actual test fold
    'llb16': "files/llb16_fold5.npz", # Example
    'llb11': "files/llb11_fold5.npz"  # Example
}

# --- Specific Burn-in Folds ---
# IMPORTANT: Specify 1 distinct HPO fold from each bird type for burn-in.
# These MUST NOT be in TEST_SET_FILES_STR_MAP.
SPECIFIC_BURN_IN_FILES_STR = [
    "files/llb3_fold1.npz",
    "files/llb16_fold1.npz",
    "files/llb11_fold1.npz"
]

N_DATA_POINTS = 1_000_000
SILENCE_LABEL_VALUE = 0

OUTPUT_BASE_DIR = pathlib.Path("./adaptive_search_V4_1M")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- Parameter Grids (Core config for pruning = UMAP + HDBSCAN + Smoothing) ---
UMAP_PARAM_GRID_CONFIG = {
    'n_components': [2, 8, 32],
    'n_neighbors': [15, 50, 100], 
    'min_dist': [0.1, 0.25],
    'metric': ["euclidean", "cosine"],  # euclidean on raw data, cosine via unit-norm + euclidean
    'random_state': [42],  # Added to ensure brute_force_knn algorithm is used consistently
}
HDBSCAN_PARAM_GRID_CONFIG = {
    'min_cluster_size': [500, 2500, 5000],
    'min_samples': [5, 50]
}
# Smoothing is now part of the core config, not applied to results afterwards
SMOOTHING_PARAM_GRID_CONFIG = {
    'smoothing_window': [0, 100, 200]
}

# --- Adaptive Search Strategy Parameters ---
OPTIMIZATION_METRIC = 'total_fer' 
OPTIMIZATION_DIRECTION = 'minimize' 
# No longer need SMOOTHING_WINDOW_FOR_RANKING since smoothing is part of core config
# SMOOTHING_WINDOW_FOR_RANKING = 100 

PRUNE_KEEP_FRACTION = 0.33 # Keep top 33% at each pruning stage

# Define checkpoints based on number of "multi-bird evaluation units" (batches of up to 3 folds)
# A multi-bird eval unit attempts to run one new fold from each bird type.
# Max possible multi-bird eval units is determined by bird with fewest HPO folds.
# Pruning happens after these many *multi-bird eval units* are completed.
# Burn-in always uses the first multi-bird eval unit (N_INITIAL_BURN_IN_UNITS = 1).
N_INITIAL_BURN_IN_UNITS = 1 
CHECKPOINT_AFTER_N_UNITS = [1, 2, 4, 6] # Example: Prune after 1st unit (burn-in), then after 2nd, 4th, 6th unit. Adjust based on total folds.


# --- Timing Estimation ---
DEFAULT_AVG_T_UMAP = 180  # Estimated seconds for 1M points, will be updated
DEFAULT_AVG_T_HDBSCAN = 60 # Estimated seconds, will be updated
DEFAULT_AVG_T_SMOOTHING_BLOCK = 30 # Estimated seconds for all smoothing windows, will be updated

# =============================================================================
# END OF CONFIGURATION PARAMETERS
# =============================================================================

# ... [ClusteringMetrics Class and Smoothing Helper Functions - OMITTED FOR BREVITY, ASSUME THEY ARE HERE] ...
# ─────────────────────────────────────────────────────────────────────────────
# ClusteringMetrics Class (calculates metrics and generates dashboard plot)
# ─────────────────────────────────────────────────────────────────────────────
class ClusteringMetrics:
    """Evaluate clustering vs. ground‑truth phrase labels."""
    def __init__(self, gt: np.ndarray, pred: np.ndarray, silence: int = 0):
        gt = np.asarray(gt).ravel(); pred = np.asarray(pred).ravel()
        if gt.shape != pred.shape:
            min_len = min(len(gt), len(pred))
            if abs(len(gt) - len(pred)) > 100: print(f"Warning: GT/Pred shapes differ ({gt.shape} vs {pred.shape}). Truncating to {min_len}")
            gt = gt[:min_len]; pred = pred[:min_len]
        self.gt_raw = gt.astype(int); self.pred = pred.astype(int)
        self.gt = self._merge_silence(self.gt_raw, silence_label=silence) 
        self.gt_types = np.unique(self.gt); self.pred_types = np.unique(self.pred)
        self._build_confusion(); self.mapping = self._hungarian()

    @staticmethod
    def _merge_silence(arr: np.ndarray, silence_label: int) -> np.ndarray:
        if arr.size == 0: return arr
        out = arr.copy(); i = 0
        while i < len(out):
            if out[i] != silence_label: i += 1; continue
            j = i
            while j < len(out) and out[j] == silence_label: j += 1
            left_val = out[i-1] if i > 0 else None; right_val = out[j] if j < len(out) else None
            fill_value = silence_label 
            if left_val is not None and left_val != silence_label: fill_value = left_val
            elif right_val is not None and right_val != silence_label: fill_value = right_val
            elif (left_val is None or left_val == silence_label) and (right_val is None or right_val == silence_label): pass
            out[i:j] = fill_value; i = j
        return out

    def _build_confusion(self) -> None:
        if not self.gt_types.size or not self.pred_types.size: self.C=np.array([],dtype=int).reshape(0,0); self.C_norm=np.array([],dtype=float).reshape(0,0); return
        gt_idx={l:i for i,l in enumerate(self.gt_types)}; pr_idx={l:i for i,l in enumerate(self.pred_types)}
        self.C=np.zeros((len(self.gt_types),len(self.pred_types)),dtype=int)
        for g,p in zip(self.gt,self.pred):
            if g in gt_idx and p in pr_idx: np.add.at(self.C,(gt_idx[g],pr_idx[p]),1)
        cs=self.C.sum(axis=0,keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'): self.C_norm=np.divide(self.C,cs,where=cs!=0,out=np.zeros_like(self.C,dtype=float))

    def _hungarian(self) -> Dict[int,int]:
        if not hasattr(self,'C_norm') or self.C_norm.size==0: return {}
        cost_matrix=-self.C_norm; 
        try: r_ind,c_ind=linear_sum_assignment(cost_matrix)
        except ValueError: return {}
        mp={}
        for r,c in zip(r_ind,c_ind):
            if r<len(self.gt_types) and c<len(self.pred_types): mp[self.gt_types[r]]=self.pred_types[c]
        return mp

    def v_measure(self) -> float:
        if self.gt.size==0 or self.pred.size==0 or len(np.unique(self.gt))==0 or len(np.unique(self.pred))==0: return 0.0
        try:
            mgt=np.min(self.gt); mpr=np.min(self.pred)
            gt_a=self.gt-mgt if mgt<0 else self.gt; pr_a=self.pred-mpr if mpr<0 else self.pred
            return v_measure_score(gt_a,pr_a)
        except ValueError: return 1.0 if len(self.gt_types)<=1 and len(self.pred_types)<=1 and (len(self.gt_types)==0 or len(self.pred_types)==0 or self.gt_types[0]==self.pred_types[0]) else 0.0

    def _fer_generic(self,use_mask:Optional[np.ndarray]=None)->float:
        if self.gt.size==0: return 100.0
        eff_gt,eff_pred=(self.gt[use_mask],self.pred[use_mask]) if use_mask is not None else (self.gt,self.pred)
        if eff_gt.size==0: return 0.0
        corr=sum(1 for g,p in zip(eff_gt,eff_pred) if g in self.mapping and self.mapping[g]==p)
        return 100.0*(1.0-corr/eff_gt.size) if eff_gt.size > 0 else 0.0
        
    def total_fer(self)->float: return self._fer_generic()
    def matched_fer(self)->float: return self._fer_generic(np.isin(self.gt,list(self.mapping.keys()))) if self.mapping else 0.0
    frame_error_rate=total_fer

    def macro_fer(self)->float:
        if not self.gt_types.size: return 100.0
        p_fer=[]
        for gt_t in self.gt_types:
            msk=(self.gt==gt_t)
            if not np.any(msk): continue
            if gt_t not in self.mapping: p_fer.append(1.0); continue
            errs=np.sum(self.pred[msk]!=self.mapping[gt_t]); tot=np.sum(msk)
            p_fer.append(errs/tot if tot>0 else 0.0)
        return 100.0*np.mean(p_fer) if p_fer else 100.0

    def stats(self)->Dict[str,Any]:
        if self.gt.size==0: return {"pct_types_mapped":0,"pct_frames_mapped":0,"mapped_counts":{},"unmapped_counts":{},"n_gt_types":0,"n_pred_types":0}
        cnts=Counter(self.gt); m_gt_t=set(self.mapping.keys())
        m_frms=sum(cnts[gtl] for gtl in m_gt_t if gtl in cnts)
        n_gt,n_pr=len(self.gt_types),len(self.pred_types)
        return {"pct_types_mapped":100*len(m_gt_t)/n_gt if n_gt else 0,
                "pct_frames_mapped":100*m_frms/self.gt.size if self.gt.size else 0,
                "mapped_counts":{k:v for k,v in cnts.items() if k in m_gt_t},
                "unmapped_counts":{k:v for k,v in cnts.items() if k not in m_gt_t},
                "n_gt_types":n_gt,"n_pred_types":n_pr}

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing Helper Functions (Copied from your provided script)
# ─────────────────────────────────────────────────────────────────────────────
def basic_majority_vote(labels:np.ndarray,window_size:int) -> np.ndarray:
    if window_size<=1 or len(labels)==0: return labels.copy()
    n=len(labels); smoothed=np.copy(labels); h_win=window_size//2
    for i in range(n):
        s,e=max(0,i-h_win),min(n,i+h_win+1); win=labels[s:e]
        if len(win)>0:
            cnts=Counter(win); top2=cnts.most_common(2)
            if len(top2)==1 or top2[0][1]>top2[1][1]: smoothed[i]=top2[0][0]
            else:
                tied=[item[0] for item in top2 if item[1]==top2[0][1]]
                smoothed[i]=labels[i] if labels[i] in tied else sorted(tied)[0]
    return smoothed

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
# Main Adaptive Search Script Logic
# ─────────────────────────────────────────────────────────────────────────────

# --- Timing Accumulators ---
cumulative_umap_time = 0.0; num_umap_runs_timed = 0
cumulative_hdbscan_time = 0.0; num_hdbscan_runs_timed = 0
cumulative_eval_block_time = 0.0; num_eval_blocks_timed = 0

# Convert string paths to Path objects and check existence
all_available_path_objects = [pathlib.Path(f) for f in ALL_AVAILABLE_FOLD_PATHS_STR]
existing_available_files = [f for f in all_available_path_objects if f.exists()] 

specific_burn_in_path_objects = [pathlib.Path(f) for f in SPECIFIC_BURN_IN_FILES_STR]
# Ensure existing_specific_burn_in_files are derived from files that actually exist
existing_specific_burn_in_files = [f for f in specific_burn_in_path_objects if f.exists() and f in existing_available_files]


ACTUAL_BURN_IN_FILES = []
REMAINING_FILES_FOR_RUNGS = []

# This logic now correctly uses existing_available_files
if len(existing_specific_burn_in_files) == len(SPECIFIC_BURN_IN_FILES_STR) and len(existing_specific_burn_in_files) > 0:
    ACTUAL_BURN_IN_FILES = existing_specific_burn_in_files
    print(f"Using specified burn-in files: {[f.name for f in ACTUAL_BURN_IN_FILES]}")
    # Ensure remaining files don't include burn-in files if they were in the main list
    # And that remaining_files are also existing files
    remaining_set = set(existing_available_files) - set(ACTUAL_BURN_IN_FILES)
    REMAINING_FILES_FOR_RUNGS = [f for f in existing_available_files if f in remaining_set] # Preserve order
else:
    print(f"Warning: Not all specific burn-in files found, or they are not in the available list, or no specific burn-in files were requested.")
    if len(existing_available_files) >= 3: # Check against all existing files
        ACTUAL_BURN_IN_FILES = existing_available_files[:len(SPECIFIC_BURN_IN_FILES_STR)] # Take up to 3 for burn-in if available
        REMAINING_FILES_FOR_RUNGS = existing_available_files[len(SPECIFIC_BURN_IN_FILES_STR):]
        print(f"Using first {len(ACTUAL_BURN_IN_FILES)} available files for burn-in: {[f.name for f in ACTUAL_BURN_IN_FILES]}")
    elif existing_available_files: # Fewer than 3 files available in total
        ACTUAL_BURN_IN_FILES = existing_available_files # Use all available
        REMAINING_FILES_FOR_RUNGS = []
        print(f"Warning: Fewer than {len(SPECIFIC_BURN_IN_FILES_STR)} files available. Using all {len(ACTUAL_BURN_IN_FILES)} available for burn-in: {[f.name for f in ACTUAL_BURN_IN_FILES]}")
    # else: # no existing_available_files, handled by the CRITICAL ERROR below

if not ACTUAL_BURN_IN_FILES and not REMAINING_FILES_FOR_RUNGS and existing_available_files:
    # This case implies existing_available_files is not empty, but somehow ACTUAL_BURN_IN_FILES ended up empty.
    # Default to using all existing if SPECIFIC_BURN_IN_FILES was empty and logic above didn't catch it.
    if len(SPECIFIC_BURN_IN_FILES_STR) == 0 and existing_available_files:
        print("No specific burn-in files defined, and default selection logic might need review. Using all available for HPO, first for burn-in if possible.")
        if len(existing_available_files) >=1 :
             ACTUAL_BURN_IN_FILES = existing_available_files[:1] # Default to 1 burn-in fold
             REMAINING_FILES_FOR_RUNGS = existing_available_files[1:]
        else: # Should be caught by the critical error below
            pass


if not existing_available_files: # Moved this check here
    print("CRITICAL ERROR: No files from ALL_AVAILABLE_FOLD_PATHS_STR exist. Please check paths.")
    exit()
elif not ACTUAL_BURN_IN_FILES and not REMAINING_FILES_FOR_RUNGS and not (len(SPECIFIC_BURN_IN_FILES_STR) == 0 and len(existing_available_files) > 0) :
    # This condition means that after attempting to assign burn-in and remaining files, both are empty,
    # unless it was intentional because SPECIFIC_BURN_IN_FILES was empty (which is now handled to default to at least one burn-in if files exist).
    print("CRITICAL ERROR: No files allocated for burn-in or subsequent rungs, but files exist. Check burn-in/test set logic. Exiting.")
    exit()

# --- Test Set Definition --- (This section should come AFTER defining existing_available_files)
test_set_path_objects = {pathlib.Path(f) for f in TEST_SET_FILES_STR_MAP.values()}
test_set_paths = {f for f in test_set_path_objects if f.exists() and f in existing_available_files} #Ensure test files exist and are in available list

# --- Prepare Fold Lists --- (This line caused the original error)
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

# Organize burn-in files by bird type for the first rung
burn_in_folds_by_bird = defaultdict(list)
for f_path in ACTUAL_BURN_IN_FILES:
    if 'llb3' in f_path.name: burn_in_folds_by_bird['llb3'].append(f_path)
    elif 'llb16' in f_path.name: burn_in_folds_by_bird['llb16'].append(f_path)
    elif 'llb11' in f_path.name: burn_in_folds_by_bird['llb11'].append(f_path)
    # Add more bird types if necessary

BIRD_TYPES_FOR_HPO = sorted(list(hpo_folds_by_bird.keys()))
if not BIRD_TYPES_FOR_HPO:
    print("CRITICAL ERROR: No bird types found in HPO folds. Exiting.")
    exit()

print(f"HPO folds by bird type: {{ {', '.join([f'{bt}: {len(folds)}' for bt, folds in hpo_folds_by_bird.items()])} }}")
print(f"Burn-in folds by bird type: {{ {', '.join([f'{bt}: {len(folds)}' for bt, folds in burn_in_folds_by_bird.items()])} }}")
print(f"Test set files (excluded from HPO): {[f.name for f in test_set_paths]}")


# --- Initialization for Adaptive Search ---
results_buffer_for_csv = [] 
core_config_performance_tracker = defaultdict(lambda: {'scores_for_ranking': [], 'folds_evaluated_count': 0, 'folds_evaluated': set(), 'last_fold_idx_processed': -1})

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
smoothing_windows = list(itertools.product(*smoothing_param_value_lists))
smoothing_windows = [dict(zip(smoothing_param_names, vals)) for vals in smoothing_windows]

CORE_CONFIG_PARAM_NAMES = umap_param_names + hdbscan_param_names  # Only UMAP+HDBSCAN for core configs
active_core_configs_keys = {config_to_key(cfg, CORE_CONFIG_PARAM_NAMES) for cfg in initial_core_configs_dicts}
initial_num_core_configs = len(active_core_configs_keys)
total_results_per_fold = initial_num_core_configs * len(smoothing_windows)
print(f"Generated {initial_num_core_configs} CORE (UMAP+HDBSCAN) configurations.")
print(f"Will evaluate {len(smoothing_windows)} smoothing windows for each successful clustering: {[sw['smoothing_window'] for sw in smoothing_windows]}")
print(f"Total results per fold: {initial_num_core_configs} configs × {len(smoothing_windows)} smoothing = {total_results_per_fold} results")

# --- CSV Checkpointing Setup ---
checkpoint_csv_path = OUTPUT_BASE_DIR / f"adaptive_search_ALL_RESULTS_{N_DATA_POINTS//1000}k.csv"
has_csv_header_been_written = False

# --- Resume Logic: Read existing CSV and rebuild performance tracker ---
if checkpoint_csv_path.exists():
    print(f"Found existing results CSV: {checkpoint_csv_path}. Attempting to resume from previous run...")
    try:
        if checkpoint_csv_path.stat().st_size > 0:
            has_csv_header_been_written = True
            
            # Read existing results to rebuild performance tracker
            existing_df = pd.read_csv(checkpoint_csv_path)
            print(f"  Loaded {len(existing_df)} existing results from CSV.")
            
            # Rebuild core_config_performance_tracker from existing data
            configs_resumed = 0
            for _, row in existing_df.iterrows():
                try:
                    # Reconstruct config key from CSV row
                    config_dict_from_row = {}
                    for param_name in CORE_CONFIG_PARAM_NAMES:
                        if param_name in row and pd.notna(row[param_name]):
                            config_dict_from_row[param_name] = row[param_name]
                    
                    if len(config_dict_from_row) == len(CORE_CONFIG_PARAM_NAMES):  # Complete config
                        config_key = config_to_key(config_dict_from_row, CORE_CONFIG_PARAM_NAMES)
                        fold_path_str = str(row['fold_path_str'])
                        
                        # Add to performance tracker
                        core_config_performance_tracker[config_key]['folds_evaluated'].add(fold_path_str)
                        
                        # Add score for ranking if it's valid
                        opt_metric_value = row.get(OPTIMIZATION_METRIC)
                        if pd.notna(opt_metric_value) and opt_metric_value not in [float('inf'), float('-inf')]:
                            core_config_performance_tracker[config_key]['scores_for_ranking'].append(float(opt_metric_value))
                        else:
                            # Add bad score for failed evaluations
                            core_config_performance_tracker[config_key]['scores_for_ranking'].append(
                                float('inf') if OPTIMIZATION_DIRECTION == 'minimize' else float('-inf')
                            )
                        
                        core_config_performance_tracker[config_key]['folds_evaluated_count'] += 1
                        configs_resumed += 1
                
                except Exception as e_resume:
                    # Skip malformed rows
                    continue
            
            print(f"  Successfully resumed {configs_resumed} config-fold combinations.")
            print(f"  Performance tracker now contains {len(core_config_performance_tracker)} configurations.")
            
    except Exception as e_csv_read:
        print(f"  Warning: Could not read existing CSV for resume ({e_csv_read}). Starting fresh.")
        has_csv_header_been_written = False
else:
    print(f"No existing results CSV found. Starting fresh run.")


# --- Main Loop through Rungs (defined by number of multi-bird evaluation units) ---
next_fold_idx_per_bird = {bird_type: 0 for bird_type in BIRD_TYPES_FOR_HPO}
burn_in_fold_idx_per_bird = {bird_type: 0 for bird_type in BIRD_TYPES_FOR_HPO}
rung_units_processed_cumulative = 0
max_possible_rung_units = max(len(hpo_folds_by_bird[bt]) for bt in BIRD_TYPES_FOR_HPO) if BIRD_TYPES_FOR_HPO else 0
# Adjust checkpoints to be in terms of rung_units
CHECKPOINT_AFTER_N_UNITS_ADJ = sorted(list(set(c for c in CHECKPOINT_AFTER_N_UNITS if c > 0 and c <= max_possible_rung_units)))
if not CHECKPOINT_AFTER_N_UNITS_ADJ or CHECKPOINT_AFTER_N_UNITS_ADJ[-1] < max_possible_rung_units:
    if max_possible_rung_units not in CHECKPOINT_AFTER_N_UNITS_ADJ and max_possible_rung_units > 0:
        CHECKPOINT_AFTER_N_UNITS_ADJ.append(max_possible_rung_units)
        CHECKPOINT_AFTER_N_UNITS_ADJ = sorted(list(set(CHECKPOINT_AFTER_N_UNITS_ADJ)))

print(f"Adaptive search checkpoints (after # multi-bird eval units): {CHECKPOINT_AFTER_N_UNITS_ADJ}")

for rung_idx, checkpoint_after_this_many_units in enumerate(CHECKPOINT_AFTER_N_UNITS_ADJ):
    print(f"\n===== RUNG {rung_idx + 1}: Target to complete {checkpoint_after_this_many_units} multi-bird evaluation units =====")
    print(f"  Number of active CORE configurations entering this rung: {len(active_core_configs_keys)}")

    if not active_core_configs_keys: print("  No active configurations. Stopping."); break

    # Determine how many new multi-bird evaluation units to process in this rung
    units_to_process_this_rung = checkpoint_after_this_many_units - rung_units_processed_cumulative
    if units_to_process_this_rung <= 0:
        print(f"  No new evaluation units for this rung checkpoint. Proceeding to prune based on data up to unit {rung_units_processed_cumulative}.")
    else:
        print(f"  Will process {units_to_process_this_rung} new multi-bird evaluation unit(s).")

    # --- Time Estimation for this Rung's new evaluations ---
    if units_to_process_this_rung > 0:
        avg_t_u = (cumulative_umap_time / num_umap_runs_timed) if num_umap_runs_timed > 0 else DEFAULT_AVG_T_UMAP
        avg_t_h = (cumulative_hdbscan_time / num_hdbscan_runs_timed) if num_hdbscan_runs_timed > 0 else DEFAULT_AVG_T_HDBSCAN
        avg_t_s_block = (cumulative_eval_block_time / num_eval_blocks_timed) if num_eval_blocks_timed > 0 else DEFAULT_AVG_T_SMOOTHING_BLOCK
        
        # Estimate number of actual UMAP and HDBSCAN runs for the new units
        # For each unit, up to len(BIRD_TYPES_FOR_HPO) folds are processed.
        # UMAP runs: unique UMAP configs * num_folds_in_unit
        # HDBSCAN runs: active_core_configs * num_folds_in_unit
        # This estimation is for ONE unit, then multiply by units_to_process_this_rung
        
        # Estimate for one multi-bird eval unit:
        # For burn-in (first N_INITIAL_BURN_IN_UNITS), use burn-in folds
        # For subsequent rungs, use regular HPO folds
        if rung_units_processed_cumulative < N_INITIAL_BURN_IN_UNITS:
            folds_in_one_unit_est = sum(1 for bt in BIRD_TYPES_FOR_HPO if burn_in_fold_idx_per_bird[bt] < len(burn_in_folds_by_bird[bt]))
        else:
            folds_in_one_unit_est = sum(1 for bt in BIRD_TYPES_FOR_HPO if next_fold_idx_per_bird[bt] < len(hpo_folds_by_bird[bt]))
        
        unique_umap_param_tuples_active = set()
        temp_umap_names = list(UMAP_PARAM_GRID_CONFIG.keys())
        for core_key in active_core_configs_keys:
            core_dict = dict(core_key)
            umap_part_dict = {k: core_dict[k] for k in temp_umap_names}
            unique_umap_param_tuples_active.add(config_to_key(umap_part_dict, temp_umap_names))
        
        num_unique_active_umap = len(unique_umap_param_tuples_active)
        num_active_core = len(active_core_configs_keys)

        est_time_one_unit = (num_unique_active_umap * avg_t_u) + \
                            (num_active_core * avg_t_h) + \
                            (num_active_core * avg_t_s_block)
        
        est_time_this_rung_new_evals_sec = units_to_process_this_rung * est_time_one_unit * folds_in_one_unit_est # Approximate, as folds_in_one_unit_est can decrease
        if folds_in_one_unit_est > 0: # Only print if there are folds to process
            print(f"  Estimated time for new evaluations in this rung: {est_time_this_rung_new_evals_sec / 3600:.2f} hours ({est_time_this_rung_new_evals_sec / 60:.1f} minutes)")


    # Loop for the number of multi-bird evaluation units for this rung
    for unit_num_in_rung in range(units_to_process_this_rung):
        current_multi_bird_eval_unit_idx = rung_units_processed_cumulative + unit_num_in_rung + 1
        print(f"\n    -- Multi-bird Evaluation Unit {current_multi_bird_eval_unit_idx} --")
        
        current_batch_of_folds = []
        
        # Use burn-in files for the first N_INITIAL_BURN_IN_UNITS units
        if current_multi_bird_eval_unit_idx <= N_INITIAL_BURN_IN_UNITS:
            print(f"      Using BURN-IN folds for unit {current_multi_bird_eval_unit_idx}")
            for bird_type in BIRD_TYPES_FOR_HPO:
                if burn_in_fold_idx_per_bird[bird_type] < len(burn_in_folds_by_bird[bird_type]):
                    current_batch_of_folds.append(burn_in_folds_by_bird[bird_type][burn_in_fold_idx_per_bird[bird_type]])
        else:
            print(f"      Using regular HPO folds for unit {current_multi_bird_eval_unit_idx}")
            for bird_type in BIRD_TYPES_FOR_HPO:
                if next_fold_idx_per_bird[bird_type] < len(hpo_folds_by_bird[bird_type]):
                    current_batch_of_folds.append(hpo_folds_by_bird[bird_type][next_fold_idx_per_bird[bird_type]])
        
        if not current_batch_of_folds:
            print("      No more folds available from any bird type for this unit. Ending unit processing.")
            units_to_process_this_rung = unit_num_in_rung # Adjust units actually processed
            break 

        print(f"      Processing folds for this unit: {[f.name for f in current_batch_of_folds]}")

        # Process this batch of (up to 3) folds
        for data_fpath in current_batch_of_folds:
            print(f"\n      --- Processing Fold: {data_fpath.name} (Unit {current_multi_bird_eval_unit_idx}) ---")
            
            X_gpu_file_for_current_fold = None; current_fold_results_buffer = []
            fold_results_saved_count = 0  # Track results saved for this fold
            try:
                data=np.load(data_fpath); X_full=data["predictions"]; gt_full=data["ground_truth_labels"]
                n_pts=min(N_DATA_POINTS,X_full.shape[0]); X_hd=X_full[:n_pts].astype(np.float32,copy=False); gt_eval=gt_full[:n_pts]
                ds_indices_smooth=np.zeros(len(gt_eval),dtype=int)
                X_gpu_file_for_current_fold=cp.asarray(X_hd)
                
                # Add this right after line 556 to debug:
                print(f"        DEBUG: Current fold path: '{str(data_fpath)}'")
                print(f"        DEBUG: Sample tracked paths: {list(list(core_config_performance_tracker.values())[0]['folds_evaluated'])[:3] if core_config_performance_tracker else 'None'}")

                # Group active configs by unique UMAP parameters to avoid redundant UMAP computations
                active_configs_for_this_fold = []
                for key in active_core_configs_keys:
                    fold_path_variations = [
                        str(data_fpath),
                        data_fpath.name,  # Just filename
                        str(data_fpath.relative_to(pathlib.Path.cwd())) if data_fpath.is_absolute() else str(data_fpath)
                    ]
                    # Check if this config has already been evaluated on this fold
                    # Look in performance tracker (which includes completed configs from CSV)
                    if key in core_config_performance_tracker:
                        already_evaluated = any(path_var in core_config_performance_tracker[key]['folds_evaluated'] for path_var in fold_path_variations)
                        if not already_evaluated:
                            active_configs_for_this_fold.append(key)
                    else:
                        # Config not in tracker yet, so it needs to be evaluated
                        active_configs_for_this_fold.append(key)
                
                # Group configs by UMAP parameters
                umap_groups = defaultdict(list)
                for core_config_key in active_configs_for_this_fold:
                    core_config_dict = dict(core_config_key)
                    umap_params = {k:v for k,v in core_config_dict.items() if k in UMAP_PARAM_GRID_CONFIG}
                    umap_key = config_to_key(umap_params, list(UMAP_PARAM_GRID_CONFIG.keys()))
                    umap_groups[umap_key].append(core_config_key)
                
                num_total_configs_for_fold = len(active_configs_for_this_fold)
                num_unique_umap_configs = len(umap_groups)
                print(f"        Processing {num_total_configs_for_fold} configs with {num_unique_umap_configs} unique UMAP configurations")
                
                config_idx = 0  # Track overall progress
                
                # Process each unique UMAP configuration
                for umap_idx, (umap_key, core_configs_for_this_umap) in enumerate(umap_groups.items()):
                    umap_params_dict = dict(umap_key)
                    print(f"\n        UMAP Config {umap_idx+1}/{num_unique_umap_configs}: {umap_params_dict}")
                    
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
                        print(f"          Running UMAP with n_neighbors={n_neighbors_val}")
                        
                        umap_model = cuml.UMAP(**umap_params_adjusted, init="spectral", n_epochs=200)
                        t_s=time.time(); emb_gpu=umap_model.fit_transform(X_for_umap); t_umap=time.time()-t_s
                        cumulative_umap_time += t_umap; num_umap_runs_timed += 1
                        del umap_model; gc.collect()
                        print(f"          UMAP completed in {t_umap:.1f}s")
                    except Exception as e_umap:
                        error_str=str(e_umap); oom_umap_flag="out_of_memory" in error_str.lower() or "bad_alloc" in error_str.lower()
                        umap_error_msg = f"UMAP Error{' (OOM)' if oom_umap_flag else ''}: {error_str}"
                        print(f"          {umap_error_msg}")
                        if 'umap_model' in locals(): del umap_model; gc.collect()
                        if emb_gpu is not None: del emb_gpu; cp.get_default_memory_pool().free_all_blocks()
                        emb_gpu = None
                    
                    # Process all HDBSCAN configs that use this UMAP embedding
                    for core_config_key_to_eval in core_configs_for_this_umap:
                        config_idx += 1
                        core_config_dict = dict(core_config_key_to_eval)
                        current_hdbscan_params = {k:v for k,v in core_config_dict.items() if k in HDBSCAN_PARAM_GRID_CONFIG}
                        
                        print(f"\n          Processing Core Config {config_idx}/{num_total_configs_for_fold}")
                        print(f"            HDBSCAN: {current_hdbscan_params}")

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
                                print(f"            {current_config_error_msg}")
                                if 'hdb_model' in locals(): del hdb_model; gc.collect()

                        # --- Evaluate ALL smoothing windows for this UMAP+HDBSCAN combination ---
                        if current_config_error_msg is None and hdb_labels_np is not None:
                            eval_block_start_time = time.time()
                            
                            # Try all smoothing windows on the same clustering results
                            for smoothing_config in smoothing_windows:
                                sm_win_val = smoothing_config['smoothing_window']
                                smoothed_preds = basic_majority_vote(hdb_labels_np, sm_win_val) if sm_win_val > 0 else hdb_labels_np.copy()
                                cm = ClusteringMetrics(gt=gt_eval, pred=smoothed_preds, silence=SILENCE_LABEL_VALUE)
                                stats = cm.stats()
                                opt_metric_val = getattr(cm, OPTIMIZATION_METRIC.lower().replace(" ", "_"))()
                                
                                # Create full config dict including smoothing window
                                full_config_dict = {**core_config_dict, **smoothing_config}
                                
                                # Create result record
                                result_record = {
                                    "fold_path_str": str(data_fpath), **full_config_dict,
                                    OPTIMIZATION_METRIC: opt_metric_val,
                                    "v_measure": cm.v_measure(), "total_fer": cm.total_fer(), "matched_fer": cm.matched_fer(), "macro_fer": cm.macro_fer(),
                                    "n_gt_types":stats['n_gt_types'], "n_pred_clusters":stats['n_pred_types'], "pct_types_mapped":stats['pct_types_mapped'], "pct_frames_mapped":stats['pct_frames_mapped'],
                                    "time_umap":t_umap, "time_hdbscan":t_hdb, "time_eval_block_all_smoothing":float('nan'), # Will be filled after all smoothing
                                    "oom_flag_umap":oom_umap_flag, "oom_flag_hdbscan":oom_hdbscan_flag, "error_message": None }
                                
                                # Print result to terminal
                                print(f"            ✓ Smoothing {sm_win_val}: {OPTIMIZATION_METRIC}={opt_metric_val:.4f}, v_measure={cm.v_measure():.4f}, clusters={stats['n_pred_types']}")
                                
                                # Add to buffer for timing update and ranking
                                current_fold_results_buffer.append(result_record)
                            
                            eval_block_duration_all_smooth = time.time() - eval_block_start_time
                            # Update timing for all smoothing window results from this config
                            for i in range(len(smoothing_windows)):
                                current_fold_results_buffer[-(i+1)]['time_eval_block_all_smoothing'] = eval_block_duration_all_smooth
                            
                            # Save all smoothing results for this config immediately
                            save_results_to_csv(current_fold_results_buffer[-len(smoothing_windows):], checkpoint_csv_path, not has_csv_header_been_written)
                            has_csv_header_been_written = True
                            fold_results_saved_count += len(smoothing_windows)
                            
                            cumulative_eval_block_time += eval_block_duration_all_smooth; num_eval_blocks_timed += 1
                            
                            # For ranking, use the best smoothing window result for this core config
                            best_metric_for_ranking = min([current_fold_results_buffer[-(i+1)][OPTIMIZATION_METRIC] for i in range(len(smoothing_windows))]) if OPTIMIZATION_DIRECTION == 'minimize' else max([current_fold_results_buffer[-(i+1)][OPTIMIZATION_METRIC] for i in range(len(smoothing_windows))])
                            core_config_performance_tracker[core_config_key_to_eval]['scores_for_ranking'].append(best_metric_for_ranking)

                        else: # UMAP or HDBSCAN error, or HDBSCAN produced no labels
                            if hdb_labels_np is None and current_config_error_msg is None : current_config_error_msg = "HDBSCAN produced no labels"
                            
                            # Log error for all smoothing windows
                            error_results = []
                            for smoothing_config in smoothing_windows:
                                full_config_dict = {**core_config_dict, **smoothing_config}
                                error_record = {
                                    "fold_path_str": str(data_fpath), **full_config_dict,
                                    OPTIMIZATION_METRIC: float('inf') if OPTIMIZATION_DIRECTION == 'minimize' else float('-inf'),
                                    "v_measure": float('nan'), "total_fer": float('nan'), "matched_fer": float('nan'), "macro_fer": float('nan'),
                                    "n_gt_types":float('nan'), "n_pred_clusters":float('nan'), "pct_types_mapped":float('nan'), "pct_frames_mapped":float('nan'),
                                    "time_umap":t_umap, "time_hdbscan":t_hdb, "time_eval_block_all_smoothing":float('nan'),
                                    "oom_flag_umap":oom_umap_flag, "oom_flag_hdbscan":oom_hdbscan_flag, "error_message": current_config_error_msg }
                                
                                # Print error to terminal
                                print(f"            ✗ Smoothing {smoothing_config['smoothing_window']}: ERROR - {current_config_error_msg}")
                                
                                error_results.append(error_record)
                                current_fold_results_buffer.append(error_record)
                            
                            # Save error results immediately
                            save_results_to_csv(error_results, checkpoint_csv_path, not has_csv_header_been_written)
                            has_csv_header_been_written = True
                            fold_results_saved_count += len(error_results)
                            
                            core_config_performance_tracker[core_config_key_to_eval]['scores_for_ranking'].append(float('inf') if OPTIMIZATION_DIRECTION == 'minimize' else float('-inf'))
                        
                        core_config_performance_tracker[core_config_key_to_eval]['folds_evaluated_count'] += 1
                        core_config_performance_tracker[core_config_key_to_eval]['folds_evaluated'].add(str(data_fpath))
                    
                    # Clean up UMAP embedding after processing all HDBSCAN configs for this UMAP
                    if emb_gpu is not None: 
                        del emb_gpu; cp.get_default_memory_pool().free_all_blocks()
                        emb_gpu = None
                    
                    # Print summary for this UMAP group
                    configs_in_this_group = len(core_configs_for_this_umap)
                    results_in_this_group = configs_in_this_group * len(smoothing_windows)
                    print(f"          ✓ UMAP group completed: {configs_in_this_group} configs × {len(smoothing_windows)} smoothing = {results_in_this_group} results saved")
                    print(f"          Progress: {fold_results_saved_count}/{num_total_configs_for_fold * len(smoothing_windows)} results saved for {data_fpath.name}")
                    
                    # --- Per-UMAP-group Time Estimation Update & Print ---
                    if num_eval_blocks_timed > 0:
                        avg_t_u_current = (cumulative_umap_time / num_umap_runs_timed) if num_umap_runs_timed > 0 else DEFAULT_AVG_T_UMAP
                        avg_t_h_current = (cumulative_hdbscan_time / num_hdbscan_runs_timed) if num_hdbscan_runs_timed > 0 else DEFAULT_AVG_T_HDBSCAN
                        avg_t_s_current = (cumulative_eval_block_time / num_eval_blocks_timed) if num_eval_blocks_timed > 0 else DEFAULT_AVG_T_SMOOTHING_BLOCK
                        
                        remaining_umap_groups = len(umap_groups) - (umap_idx + 1)
                        remaining_configs_this_fold = num_total_configs_for_fold - config_idx
                        
                        # Estimate remaining time for this fold
                        est_time_remaining_this_fold_sec = remaining_umap_groups * avg_t_u_current + remaining_configs_this_fold * (avg_t_h_current + avg_t_s_current)
                        
                        # Estimate for remaining folds in this multi-bird unit
                        remaining_folds_in_unit = len(current_batch_of_folds) - (current_batch_of_folds.index(data_fpath) + 1)
                        est_time_remaining_unit_sec = remaining_folds_in_unit * (num_unique_umap_configs * avg_t_u_current + num_total_configs_for_fold * (avg_t_h_current + avg_t_s_current))
                        
                        total_rem_sec = est_time_remaining_this_fold_sec + est_time_remaining_unit_sec
                        print(f"          Est. time remaining for current batch: {total_rem_sec / 60:.1f} mins")

            finally: # For the fold processing try-block
                if X_gpu_file_for_current_fold is not None: del X_gpu_file_for_current_fold; X_gpu_file_for_current_fold = None
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect() # More aggressive garbage collection
                # Results are now saved immediately after each config, no need for fold-level batching
                current_fold_results_buffer = [] # Clear buffer
                print(f"      ✓ Fold {data_fpath.name} completed: {fold_results_saved_count} total results saved to CSV")

        # Update fold indices for the folds processed in this unit
        if current_multi_bird_eval_unit_idx <= N_INITIAL_BURN_IN_UNITS:
            # Update burn-in fold indices
            for bird_type_processed in BIRD_TYPES_FOR_HPO:
                if burn_in_fold_idx_per_bird[bird_type_processed] < len(burn_in_folds_by_bird[bird_type_processed]):
                     # Check if a burn-in fold from this bird type was actually part of current_batch_of_folds
                     was_processed_this_unit = any(bf == burn_in_folds_by_bird[bird_type_processed][burn_in_fold_idx_per_bird[bird_type_processed]] for bf in current_batch_of_folds)
                     if was_processed_this_unit:
                        burn_in_fold_idx_per_bird[bird_type_processed] += 1
        else:
            # Update regular HPO fold indices
            for bird_type_processed in BIRD_TYPES_FOR_HPO:
                if next_fold_idx_per_bird[bird_type_processed] < len(hpo_folds_by_bird[bird_type_processed]):
                     # Check if a fold from this bird type was actually part of current_batch_of_folds
                     was_processed_this_unit = any(bf == hpo_folds_by_bird[bird_type_processed][next_fold_idx_per_bird[bird_type_processed]] for bf in current_batch_of_folds)
                     if was_processed_this_unit:
                        next_fold_idx_per_bird[bird_type_processed] += 1
        
        rung_units_processed_cumulative +=1 # Increment after processing a multi-bird unit

        # --- PRUNING STEP at the end of the RUNG (after processing all units for this rung checkpoint) ---
        if rung_units_processed_cumulative >= checkpoint_after_this_many_units or unit_num_in_rung == units_to_process_this_rung -1: # Prune at checkpoint or end of planned units for rung
            print(f"\n  --- Pruning after {rung_units_processed_cumulative} multi-bird units (Rung {rung_idx + 1} completed) ---")
            
            configs_to_rank_for_pruning = []
            for cfg_key in list(active_core_configs_keys): 
                # Config has been evaluated on 'folds_evaluated_count' distinct folds for the ranking metric
                num_folds_evaluated_for_cfg = core_config_performance_tracker[cfg_key]['folds_evaluated_count']
                # For pruning, we need it to have been evaluated on all folds *targeted by this checkpoint*
                # which is rung_units_processed_cumulative * (avg folds per unit, roughly).
                # A simpler way: use all scores accumulated so far for ranking.
                
                if core_config_performance_tracker[cfg_key]['scores_for_ranking']: # If it has any scores
                    avg_score = np.nanmean(core_config_performance_tracker[cfg_key]['scores_for_ranking'])
                    configs_to_rank_for_pruning.append({'config_key': cfg_key, 'avg_score': avg_score, 'num_scores': len(core_config_performance_tracker[cfg_key]['scores_for_ranking'])})
                else: # No valid scores yet (e.g. all errors for ranking smoothing window)
                    configs_to_rank_for_pruning.append({'config_key': cfg_key, 'avg_score': float('inf') if OPTIMIZATION_DIRECTION == 'minimize' else float('-inf'), 'num_scores': 0})


            if not configs_to_rank_for_pruning:
                print("    No configurations with scores to perform pruning. Keeping all current active configs.")
            else:
                ranking_df = pd.DataFrame(configs_to_rank_for_pruning)
                num_configs_before_prune = len(ranking_df)

                # Only rank configs that have seen at least N_INITIAL_BURN_IN_UNITS * (min_folds_per_unit, e.g. 1)
                # For simplicity, we rank all that have scores. More robust: rank only those seen on all "rung_units_processed_cumulative" folds.
                # The current tracker's 'scores_for_ranking' will average over however many folds that config was successfully run on (for the ranking smoothing window).

                num_to_keep = int(np.ceil(num_configs_before_prune * PRUNE_KEEP_FRACTION))
                ascending_order = True if OPTIMIZATION_DIRECTION == 'minimize' else False
                survivors_df = ranking_df.sort_values(by='avg_score', ascending=ascending_order).head(num_to_keep)
                
                new_active_configs_keys = set(survivors_df['config_key'].unique())
                
                print(f"    Pruning: Ranked {num_configs_before_prune} configs. Keeping top {num_to_keep} ({len(new_active_configs_keys)} unique).")
                print(f"    Discarded {len(active_core_configs_keys) - len(new_active_configs_keys)} configurations.")
                active_core_configs_keys = new_active_configs_keys
            
            if not active_core_configs_keys: print("  All configurations pruned. Stopping search."); break # Break Rung Loop

    # --- Final Save of any remaining results in buffer ---
    if results_buffer_for_csv: 
        save_results_to_csv(results_buffer_for_csv, checkpoint_csv_path, not has_csv_header_been_written)
        results_buffer_for_csv = []

    print(f"\n===== Adaptive search process complete. All collected results saved to: {checkpoint_csv_path} =====")
    
    # --- Final Ranking Display (load from the comprehensive CSV) ---
    if checkpoint_csv_path.exists():
        final_summary_df_loaded = pd.read_csv(checkpoint_csv_path)
        if not final_summary_df_loaded.empty and active_core_configs_keys:
            print("\n--- Top configurations that survived all prunings ---")
            
            temp_core_config_names = CORE_CONFIG_PARAM_NAMES # UMAP+HDBSCAN keys
            
            # Create config_key from the columns to match our active_core_configs_keys
            # This needs to handle potential missing columns in error rows if they are not filtered out first
            def get_config_key_from_row(row, param_names_list):
                cfg_dict = {}
                all_keys_present = True
                for p_name in param_names_list:
                    if p_name not in row or pd.isna(row[p_name]):
                        # This row doesn't represent a full core config (e.g. it's an error row from before config was fully defined)
                        # A more robust way is to ensure config_key is saved with each row if possible
                        # For now, we will try to make a key, it might be partial for error rows
                        # but filtering below should handle it.
                        cfg_dict[p_name] = row.get(p_name) # Will be None if not there
                    else:
                        cfg_dict[p_name] = row[p_name]
                return config_to_key(cfg_dict, param_names_list)

            final_summary_df_loaded['core_config_key_temp'] = final_summary_df_loaded.apply(
                lambda row: get_config_key_from_row(row, temp_core_config_names), axis=1
            )
            
            final_survivors_data = final_summary_df_loaded[
                final_summary_df_loaded['core_config_key_temp'].isin(active_core_configs_keys) &
                final_summary_df_loaded[OPTIMIZATION_METRIC].notna() &
                pd.to_numeric(final_summary_df_loaded[OPTIMIZATION_METRIC], errors='coerce').notna() & # ensure it's numeric after coerce
                (final_summary_df_loaded[OPTIMIZATION_METRIC] != float('inf')) & 
                (final_summary_df_loaded[OPTIMIZATION_METRIC] != float('-inf'))
            ].copy()

            if not final_survivors_data.empty:
                # Average the OPTIMIZATION_METRIC for each surviving core_config_key across all folds it was evaluated on
                avg_perf_final_survivors = final_survivors_data.groupby('core_config_key_temp')[OPTIMIZATION_METRIC].mean().reset_index()
                final_ranked_survivors_df = avg_perf_final_survivors.sort_values(by=OPTIMIZATION_METRIC, ascending=(OPTIMIZATION_DIRECTION == 'minimize'))

                print(f"Top surviving configurations (max 10 shown) ranked by mean '{OPTIMIZATION_METRIC}':")
                for rank, (_, row) in enumerate(final_ranked_survivors_df.head(10).iterrows()):
                    config_dict_readable = dict(row['core_config_key_temp'])
                    print(f"  Rank {rank+1}: Avg {OPTIMIZATION_METRIC}={row[OPTIMIZATION_METRIC]:.4f}, Config={config_dict_readable}")
            else: print("  No valid data to rank final survivors.")
        elif not active_core_configs_keys: print("  No configurations survived all pruning stages.")
        else: print("  Final results CSV is empty or no survivors to rank.")
    else: print(f"\n===== Adaptive search complete. Results CSV not found at {checkpoint_csv_path}. =====")