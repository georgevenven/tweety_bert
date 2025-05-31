import time
import numpy as np
import cupy as cp
import cuml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import v_measure_score
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import os
import pathlib
import itertools # For generating parameter combinations

# ─────────────────────────────────────────────────────────────────────────────
# ClusteringMetrics Class (calculates metrics and generates dashboard plot)
# ─────────────────────────────────────────────────────────────────────────────
class ClusteringMetrics:
    """Evaluate clustering vs. ground‑truth phrase labels."""

    def __init__(self, gt: np.ndarray, pred: np.ndarray, silence: int = 0):
        # Ensure gt and pred are 1D arrays
        gt = np.asarray(gt).ravel()
        pred = np.asarray(pred).ravel()

        if gt.shape != pred.shape:
            min_len = min(len(gt), len(pred))
            # Only print warning if difference is substantial
            if abs(len(gt) - len(pred)) > 100: # Arbitrary threshold
                 print(f"Warning: GT and Pred shapes differ significantly ({gt.shape} vs {pred.shape}). Truncating to shortest length: {min_len}")
            gt = gt[:min_len]
            pred = pred[:min_len]

        self.gt_raw = gt.astype(int)
        self.pred = pred.astype(int)
        self.gt = self._merge_silence(self.gt_raw, silence_label=silence) 

        self.gt_types = np.unique(self.gt)
        self.pred_types = np.unique(self.pred)

        self._build_confusion()
        self.mapping = self._hungarian()

    @staticmethod
    def _merge_silence(arr: np.ndarray, silence_label: int) -> np.ndarray:
        """Fill contiguous *silence_label* runs with the nearest neighbour label."""
        if arr.size == 0:
            return arr
        out = arr.copy()
        i = 0
        while i < len(out):
            if out[i] != silence_label:
                i += 1
                continue
            
            j = i
            while j < len(out) and out[j] == silence_label:
                j += 1
            
            left_val = out[i-1] if i > 0 else None
            right_val = out[j] if j < len(out) else None
            fill_value = silence_label 

            if left_val is not None and left_val != silence_label:
                fill_value = left_val
            elif right_val is not None and right_val != silence_label:
                fill_value = right_val
            else: 
                non_silence_values = arr[arr != silence_label]
                if non_silence_values.size > 0:
                    if left_val is None and right_val is None : 
                        pass 
                    elif left_val is None : 
                         if right_val is not None and right_val != silence_label: fill_value = right_val
                    elif right_val is None : 
                         if left_val is not None and left_val != silence_label: fill_value = left_val
            out[i:j] = fill_value
            i = j
        return out

    def _build_confusion(self) -> None:
        if not self.gt_types.size or not self.pred_types.size:
            self.C = np.array([], dtype=int).reshape(0,0) 
            self.C_norm = np.array([], dtype=float).reshape(0,0)
            return

        gt_idx = {l: i for i, l in enumerate(self.gt_types)}
        pr_idx = {l: i for i, l in enumerate(self.pred_types)}
        self.C = np.zeros((len(self.gt_types), len(self.pred_types)), dtype=int)
        
        for g_val, p_val in zip(self.gt, self.pred):
            if g_val in gt_idx and p_val in pr_idx:
                 np.add.at(self.C, (gt_idx[g_val], pr_idx[p_val]), 1)
        
        col_sum = self.C.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'): 
            self.C_norm = np.divide(self.C, col_sum, where=col_sum != 0, out=np.zeros_like(self.C, dtype=float))

    def _hungarian(self) -> Dict[int, int]:
        if not hasattr(self, 'C_norm') or self.C_norm.size == 0:
            return {}
        cost_matrix = -self.C_norm 
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ValueError: 
            return {}
            
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            if r < len(self.gt_types) and c < len(self.pred_types):
                mapping[self.gt_types[r]] = self.pred_types[c]
        return mapping

    def v_measure(self) -> float:
        if self.gt.size == 0 or self.pred.size == 0 or len(np.unique(self.gt))==0 or len(np.unique(self.pred))==0 :
            return 0.0
        try:
            min_gt_label = np.min(self.gt)
            min_pred_label = np.min(self.pred)
            gt_adjusted = self.gt - min_gt_label if min_gt_label < 0 else self.gt
            pred_adjusted = self.pred - min_pred_label if min_pred_label < 0 else self.pred
            return v_measure_score(gt_adjusted, pred_adjusted)
        except ValueError:
            if len(self.gt_types) <= 1 and len(self.pred_types) <= 1:
                if len(self.gt_types) == 0 or len(self.pred_types) == 0: return 0.0
                return 1.0 if self.gt_types[0] == self.pred_types[0] else 0.0
            return 0.0

    def _fer_generic(self, use_mask: Optional[np.ndarray] = None) -> float:
        if self.gt.size == 0:
            return 100.0 
        effective_gt = self.gt
        effective_pred = self.pred
        if use_mask is not None:
            effective_gt = self.gt[use_mask]
            effective_pred = self.pred[use_mask]
        if effective_gt.size == 0:
            return 0.0 
        correct = 0
        for g, p in zip(effective_gt, effective_pred):
            if g in self.mapping and self.mapping[g] == p:
                correct += 1
        
        if effective_gt.size == 0:
            return 0.0 
        return 100.0 * (1.0 - correct / effective_gt.size)


    def total_fer(self) -> float:
        return self._fer_generic()

    def matched_fer(self) -> float:
        if not self.mapping: 
            return 0.0 
        mapped_mask = np.isin(self.gt, list(self.mapping.keys()))
        return self._fer_generic(use_mask=mapped_mask)

    frame_error_rate = total_fer 

    def macro_fer(self) -> float:
        if not self.gt_types.size:
            return 100.0
        per_type_fer = []
        for gt_label_type in self.gt_types:
            type_mask = (self.gt == gt_label_type)
            if not np.any(type_mask):
                continue 
            if gt_label_type not in self.mapping:
                per_type_fer.append(1.0) 
                continue
            mapped_pred_label = self.mapping[gt_label_type]
            errors_for_type = np.sum(self.pred[type_mask] != mapped_pred_label)
            total_for_type = np.sum(type_mask)
            per_type_fer.append(errors_for_type / total_for_type if total_for_type > 0 else 0.0)
        return 100.0 * np.mean(per_type_fer) if per_type_fer else 100.0

    def stats(self) -> Dict[str, Any]:
        if self.gt.size == 0: 
            return dict(
                pct_types_mapped=0, pct_frames_mapped=0,
                mapped_counts={}, unmapped_counts={},
                n_gt_types=0, n_pred_types=0
            )
        counts = Counter(self.gt) 
        mapped_gt_types = set(self.mapping.keys())
        mapped_frames = sum(counts[gt_label] for gt_label in mapped_gt_types if gt_label in counts)
        total_frames = self.gt.size
        n_gt_types = len(self.gt_types)
        n_pred_types = len(self.pred_types)
        return dict(
            pct_types_mapped=100 * len(mapped_gt_types) / n_gt_types if n_gt_types else 0,
            pct_frames_mapped=100 * mapped_frames / total_frames if total_frames else 0,
            mapped_counts={k: v for k, v in counts.items() if k in mapped_gt_types},
            unmapped_counts={k: v for k, v in counts.items() if k not in mapped_gt_types},
            n_gt_types=n_gt_types,
            n_pred_types=n_pred_types,
        )

    def plot(self, title: str = "Clustering Evaluation", figsize=(18, 10)) -> plt.Figure:
        st = self.stats()
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2])

        def _annot_bar(ax, data_dict, color, bar_title):
            if not data_dict:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(bar_title, fontsize=10)
                return
            sorted_items = sorted(data_dict.items(), key=lambda item: item[0])
            labels = [str(item[0]) for item in sorted_items]
            vals = np.array([item[1] for item in sorted_items])
            total_vals_sum = np.sum(vals)
            perc = 100 * vals / total_vals_sum if total_vals_sum > 0 else np.zeros_like(vals, dtype=float)
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, perc, color=color)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            for bar_obj, p_val_single in zip(bars, perc):
                if p_val_single > 0.5:
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, p_val_single, 
                            f"{p_val_single:.1f}%", ha="center", va="bottom", fontsize=7)
            ax.set_title(bar_title, fontsize=10)
            ax.set_ylabel("% frames in category", fontsize=9)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim(0, max(10, np.max(perc) * 1.1 if perc.size > 0 and np.max(perc) > 0 else 10))

        ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
        txt = (
            f"Total FER: {self.total_fer():.1f}%\n"
            f"Matched FER: {self.matched_fer():.1f}%\n"
            f"Macro FER: {self.macro_fer():.1f}%\n"
            f"V‑measure: {self.v_measure():.3f}\n\n"
            f"GT types: {st['n_gt_types']}\n"
            f"Pred types: {st['n_pred_types']}\n"
            f"Mapped GT types: {st['pct_types_mapped']:.1f}% ({len(st['mapped_counts'])}/{st['n_gt_types']})"
        )
        ax0.text(0.01, 0.99, txt, va="top", ha="left", fontsize=9, 
                 bbox=dict(fc="whitesmoke", alpha=.8, boxstyle="round,pad=0.5"))

        ax1 = fig.add_subplot(gs[0, 1])
        if st['n_gt_types'] > 0 :
             ax1.pie([st['pct_types_mapped'], 100 - st['pct_types_mapped']], 
                     labels=["Mapped", "Unmapped"], autopct="%.1f%%", 
                     colors=["#8fd175", "#f28e8e"], startangle=90, textprops={'fontsize': 8})
        else:
            ax1.text(0.5, 0.5, "No GT types", ha="center", va="center", fontsize=9)
        ax1.set_title("GT Label Types Mapped", fontsize=10)

        ax2 = fig.add_subplot(gs[0, 2])
        if self.gt.size > 0: 
            ax2.pie([st['pct_frames_mapped'], 100 - st['pct_frames_mapped']], 
                    labels=["Mapped", "Unmapped"], autopct="%.1f%%", 
                    colors=["#71b3ff", "#ffb471"], startangle=90, textprops={'fontsize': 8})
        else:
            ax2.text(0.5, 0.5, "No GT frames", ha="center", va="center", fontsize=9)
        ax2.set_title("GT Frames Mapped", fontsize=10)
        
        ax3 = fig.add_subplot(gs[1, 0:2]) 
        _annot_bar(ax3, st['mapped_counts'], "#4daf4a", "Mapped GT Labels (% of Mapped Frames)")
        
        ax4 = fig.add_subplot(gs[1, 2]) 
        _annot_bar(ax4, st['unmapped_counts'], "#d73027", "Unmapped GT Labels (% of Unmapped Frames)")

        fig.suptitle(title, fontsize=14, y=0.99)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97]) 
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def basic_majority_vote(labels: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or len(labels) == 0:
        return labels.copy() 
    n = len(labels)
    smoothed_labels = np.copy(labels) 
    half_window = window_size // 2
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = labels[start:end]
        if len(window) > 0:
            counts = Counter(window)
            top_two = counts.most_common(2)
            if len(top_two) == 1: 
                 smoothed_labels[i] = top_two[0][0]
            elif top_two[0][1] > top_two[1][1]: 
                 smoothed_labels[i] = top_two[0][0]
            else: 
                tied_labels = [item[0] for item in top_two if item[1] == top_two[0][1]]
                if labels[i] in tied_labels:
                    smoothed_labels[i] = labels[i]
                else:
                    smoothed_labels[i] = sorted(tied_labels)[0] 
    return smoothed_labels

def smooth_labels_per_sequence(
    raw_labels: np.ndarray, 
    dataset_indices: np.ndarray, 
    window_size: int
) -> np.ndarray:
    if window_size <= 1 or raw_labels.size == 0:
        return raw_labels.copy()
    if dataset_indices.size == 0 or len(dataset_indices) != len(raw_labels):
        print("Warning: dataset_indices issue. Applying smoothing globally.")
        return basic_majority_vote(raw_labels, window_size)
    smoothed_labels = np.zeros_like(raw_labels)
    unique_indices = np.unique(dataset_indices)
    for seq_idx in unique_indices:
        mask = (dataset_indices == seq_idx)
        sequence_labels = raw_labels[mask]
        if len(sequence_labels) > 0:
            smoothed_labels[mask] = basic_majority_vote(sequence_labels, window_size)
    return smoothed_labels

# ─────────────────────────────────────────────────────────────────────────────
# Main Grid Search Script
# ─────────────────────────────────────────────────────────────────────────────

# ---------- Configuration ----------
FILES_TO_PROCESS = [
    # "files/llb3_fold2.npz",
    # "files/llb16_fold2.npz",
    "files/llb11_fold2.npz",
    # Add more files here if needed for a fuller run later
]
RANDOM_STATE = 42
N_DATA_POINTS = 1_000_000  # Reduced number of data points
SILENCE_LABEL_VALUE = 0 

OUTPUT_BASE_DIR = pathlib.Path("./grid_search_reports_reuse_umap_250k") # New output dir
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Define UMAP and HDBSCAN parameter grids separately
UMAP_PARAM_GRID_CONFIG = {
    'umap_n_components': [32],
    'umap_n_neighbors': [250], 
    'umap_min_dist': [0.1], 
    'umap_metric': ["euclidean"],
}
HDBSCAN_PARAM_GRID_CONFIG = {
    'hdbscan_min_cluster_size': [2500], 
    'hdbscan_min_samples': [200]
}
SMOOTHING_WINDOWS = [0, 50, 100, 200] 

all_run_metrics = [] 

valid_files_to_process = []
for f_path_str in FILES_TO_PROCESS:
    f_path = pathlib.Path(f_path_str)
    if f_path.exists():
        valid_files_to_process.append(f_path)
    else:
        print(f"Warning: File not found {f_path_str}, skipping.")

if not valid_files_to_process:
    print("No valid files found to process. Please check FILES_TO_PROCESS paths.")
else:
    print(f"Processing {len(valid_files_to_process)} files with N_DATA_POINTS={N_DATA_POINTS}.")
    num_umap_combos = np.prod([len(v) for v in UMAP_PARAM_GRID_CONFIG.values()])
    num_hdbscan_combos = np.prod([len(v) for v in HDBSCAN_PARAM_GRID_CONFIG.values()])
    total_pipeline_runs_per_file = num_umap_combos * num_hdbscan_combos
    print(f"UMAP parameter combinations per file: {num_umap_combos}")
    print(f"HDBSCAN parameter combinations per UMAP embedding: {num_hdbscan_combos}")
    print(f"Total UMAP runs to perform: {len(valid_files_to_process) * num_umap_combos}")
    estimated_total_eval_sets = len(valid_files_to_process) * total_pipeline_runs_per_file * len(SMOOTHING_WINDOWS)
    print(f"Total evaluation sets (including smoothing): {estimated_total_eval_sets}")


for file_idx, data_file_path in enumerate(valid_files_to_process):
    print(f"\n===== Processing File {file_idx+1}/{len(valid_files_to_process)}: {data_file_path.name} =====")
    try:
        data = np.load(data_file_path)
        X_high_dim_full = data["predictions"]
        gt_labels_full = data["ground_truth_labels"]
        
        current_n_points = min(N_DATA_POINTS, X_high_dim_full.shape[0])
        X_high_dim = X_high_dim_full[:current_n_points].astype(np.float32, copy=False)
        gt_labels_for_eval = gt_labels_full[:current_n_points]
        
        dataset_indices_for_smoothing = np.zeros(len(gt_labels_for_eval), dtype=int) 
        print(f"  Loaded data: X_high_dim shape {X_high_dim.shape}, gt_labels shape {gt_labels_for_eval.shape}")
        X_gpu_full_file = cp.asarray(X_high_dim) # Load data to GPU once per file

    except Exception as e:
        print(f"  Error loading data from {data_file_path.name}: {e}. Skipping file.")
        # Log error for this file
        all_run_metrics.append({
            "file": data_file_path.name, "error_message": f"Data loading error: {str(e)}",
            **{k: "N/A" for k in list(UMAP_PARAM_GRID_CONFIG.keys()) + list(HDBSCAN_PARAM_GRID_CONFIG.keys())}, 
            "smoothing_window": "N/A",
            "v_measure": "N/A", "total_fer": "N/A", "matched_fer": "N/A", "macro_fer": "N/A",
            "time_umap": "N/A", "time_hdbscan": "N/A"
        })
        continue

    # Create iterators for UMAP and HDBSCAN parameters
    umap_param_names = list(UMAP_PARAM_GRID_CONFIG.keys())
    umap_param_value_lists = list(UMAP_PARAM_GRID_CONFIG.values())
    
    hdbscan_param_names = list(HDBSCAN_PARAM_GRID_CONFIG.keys())
    hdbscan_param_value_lists = list(HDBSCAN_PARAM_GRID_CONFIG.values())

    umap_run_counter = 0
    total_umap_runs_for_file = np.prod([len(v) for v in umap_param_value_lists])

    for umap_combo_values in itertools.product(*umap_param_value_lists):
        umap_run_counter += 1
        current_umap_params = dict(zip(umap_param_names, umap_combo_values))
        
        print(f"\n  --- UMAP Run {umap_run_counter}/{total_umap_runs_for_file} for {data_file_path.name} ---")
        print(f"    UMAP Parameters: {current_umap_params}")
        
        time_umap = "N/A"
        embedding_gpu = None # Initialize to ensure it's defined

        try:
            umap_embedder = cuml.UMAP(
                n_components=current_umap_params['umap_n_components'],
                n_neighbors=current_umap_params['umap_n_neighbors'],
                min_dist=current_umap_params['umap_min_dist'],
                metric=current_umap_params['umap_metric'],
                init="spectral", 
                random_state=RANDOM_STATE,
                n_epochs=200, 
            )
            t0_umap = time.time()
            embedding_gpu = umap_embedder.fit_transform(X_gpu_full_file) # Use the full file's GPU data
            time_umap = time.time() - t0_umap
            print(f"      UMAP ({current_umap_params['umap_n_components']}D) completed in {time_umap:.2f}s. Embedding shape: {embedding_gpu.shape}")

        except Exception as e_umap:
            print(f"      Error during UMAP for {current_umap_params}: {e_umap}")
            # Log error for all HDBSCAN and smoothing combos for this failed UMAP
            for hdbscan_combo_values_error in itertools.product(*hdbscan_param_value_lists):
                current_hdbscan_params_error = dict(zip(hdbscan_param_names, hdbscan_combo_values_error))
                for sm_window_error in SMOOTHING_WINDOWS:
                    all_run_metrics.append({
                        "file": data_file_path.name, **current_umap_params, **current_hdbscan_params_error,
                        "smoothing_window": sm_window_error,
                        "v_measure": "ERROR_UMAP", "total_fer": "ERROR_UMAP", 
                        "matched_fer": "ERROR_UMAP", "macro_fer": "ERROR_UMAP",
                        "time_umap": time_umap if isinstance(time_umap, (int,float)) else "ERROR", 
                        "time_hdbscan": "N/A",
                        "error_message": f"UMAP Error: {str(e_umap)}"
                    })
            if embedding_gpu is not None: del embedding_gpu
            cp.get_default_memory_pool().free_all_blocks()
            continue # Skip to next UMAP parameter combination
        
        # --- Inner loop for HDBSCAN parameters, reusing the UMAP embedding ---
        hdbscan_run_counter = 0
        total_hdbscan_runs = np.prod([len(v) for v in hdbscan_param_value_lists])

        for hdbscan_combo_values in itertools.product(*hdbscan_param_value_lists):
            hdbscan_run_counter += 1
            current_hdbscan_params = dict(zip(hdbscan_param_names, hdbscan_combo_values))
            
            print(f"    --- HDBSCAN Run {hdbscan_run_counter}/{total_hdbscan_runs} (UMAP Run {umap_run_counter}) ---")
            print(f"      HDBSCAN Parameters: {current_hdbscan_params}")

            # Create combined params for directory and logging
            combined_current_params = {**current_umap_params, **current_hdbscan_params}
            param_str_for_dir = "-".join(f"{k.split('_')[-1][0]}{v}" for k,v in combined_current_params.items())
            report_path_level1 = OUTPUT_BASE_DIR / f"File_{data_file_path.stem}"
            report_path_level2 = report_path_level1 / param_str_for_dir
            report_path_level2.mkdir(parents=True, exist_ok=True)

            time_hdbscan = "N/A"
            hdbscan_labels_raw_np = None

            try:
                hdbscan_clusterer = cuml.HDBSCAN(
                    min_cluster_size=current_hdbscan_params['hdbscan_min_cluster_size'],
                    min_samples = current_hdbscan_params['hdbscan_min_samples'],
                    metric='euclidean', 
                    prediction_data=False 
                )
                t0_hdbscan = time.time()
                # Important: HDBSCAN runs on the embedding_gpu from the outer loop
                hdbscan_labels_raw_gpu = hdbscan_clusterer.fit_predict(embedding_gpu) 
                hdbscan_labels_raw_np = cp.asnumpy(hdbscan_labels_raw_gpu)
                time_hdbscan = time.time() - t0_hdbscan
                print(f"        HDBSCAN completed in {time_hdbscan:.2f}s. Unique labels: {np.unique(hdbscan_labels_raw_np).size}")
                
                del hdbscan_labels_raw_gpu # Free GPU memory for labels
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e_hdbscan:
                print(f"        Error during HDBSCAN for {current_hdbscan_params}: {e_hdbscan}")
                for sm_window_error in SMOOTHING_WINDOWS:
                    all_run_metrics.append({
                        "file": data_file_path.name, **combined_current_params,
                        "smoothing_window": sm_window_error,
                        "v_measure": "ERROR_HDBSCAN", "total_fer": "ERROR_HDBSCAN", 
                        "matched_fer": "ERROR_HDBSCAN", "macro_fer": "ERROR_HDBSCAN",
                        "time_umap": time_umap, 
                        "time_hdbscan": time_hdbscan if isinstance(time_hdbscan, (int,float)) else "ERROR",
                        "error_message": f"HDBSCAN Error: {str(e_hdbscan)}"
                    })
                cp.get_default_memory_pool().free_all_blocks() # Ensure cleanup even on hdbscan error
                continue # Skip to next HDBSCAN parameter combination
            
            # --- Innermost loop for smoothing and evaluation ---
            for smoothing_window in SMOOTHING_WINDOWS:
                if hdbscan_labels_raw_np is None: # Should not happen if HDBSCAN succeeded
                    print("        Skipping smoothing due to no HDBSCAN labels.")
                    continue

                if smoothing_window == 0:
                    smoothed_predictions = hdbscan_labels_raw_np.copy()
                else:
                    smoothed_predictions = basic_majority_vote(hdbscan_labels_raw_np, smoothing_window)
                
                cm = ClusteringMetrics(gt=gt_labels_for_eval, pred=smoothed_predictions, silence=SILENCE_LABEL_VALUE) 
                
                current_metrics_summary = {
                    "file": data_file_path.name, **combined_current_params, 
                    "smoothing_window": smoothing_window,
                    "v_measure": cm.v_measure(), "total_fer": cm.total_fer(),
                    "matched_fer": cm.matched_fer(), "macro_fer": cm.macro_fer(),
                    "n_gt_types": cm.stats()['n_gt_types'], "n_pred_clusters": cm.stats()['n_pred_types'], 
                    "pct_gt_types_mapped": cm.stats()['pct_types_mapped'],
                    "time_umap": time_umap, "time_hdbscan": time_hdbscan,
                    "error_message": None
                }
                all_run_metrics.append(current_metrics_summary)
                
                summary_text_path = report_path_level2 / f"summary_smooth_{smoothing_window}.txt"
                stats_data = cm.stats()
                with open(summary_text_path, "w") as f:
                    f.write(f"File: {data_file_path.name}\n")
                    f.write(f"Parameters: {combined_current_params}\n") # Log combined params
                    f.write(f"Smoothing Window: {smoothing_window}\n")
                    f.write("-------------------------------------------------\n")
                    f.write(f"V-measure Score          : {cm.v_measure():.4f}\n")
                    f.write(f"Total FER               : {cm.total_fer():.2f}%\n")
                    f.write(f"Matched-only FER        : {cm.matched_fer():.2f}%\n")
                    f.write(f"Macro Frame Error Rate   : {cm.macro_fer():.2f}%\n")
                    f.write(f"UMAP time (s)            : {time_umap:.2f}\n") 
                    f.write(f"HDBSCAN time (s)         : {time_hdbscan:.2f}\n")
                    f.write("-------------------------------------------------\n")
                    f.write(f"GT Label Types           : {stats_data['n_gt_types']}\n")
                    f.write(f"Predicted Clusters       : {stats_data['n_pred_types']}\n")
                    f.write(f"% GT Types Mapped        : {stats_data['pct_types_mapped']:.2f}%\n")
                    f.write(f"% GT Frames Mapped       : {stats_data['pct_frames_mapped']:.2f}%\n")
                
                print(f"        Metrics (smooth {smoothing_window}): V-m={cm.v_measure():.3f}, FER={cm.total_fer():.1f}%")
        
        # Clean up UMAP embedding from GPU after all its HDBSCAN variants are done
        if embedding_gpu is not None:
            del embedding_gpu
            cp.get_default_memory_pool().free_all_blocks()

    # Clean up the full file data from GPU after processing all UMAP/HDBSCAN for this file
    if 'X_gpu_full_file' in locals() and X_gpu_full_file is not None:
        del X_gpu_full_file
        cp.get_default_memory_pool().free_all_blocks()


if all_run_metrics:
    summary_file_path = OUTPUT_BASE_DIR / "grid_search_summary_reuse_umap_250k.csv"
    if all_run_metrics: 
        header = list(all_run_metrics[0].keys())
        with open(summary_file_path, "w") as f:
            f.write(",".join(header) + "\n")
            for metrics_dict in all_run_metrics:
                row_values = []
                for key in header: 
                    val = metrics_dict.get(key) 
                    if isinstance(val, float):
                        row_values.append(f"{val:.4f}")
                    elif val is None:
                        row_values.append("") 
                    else:
                        row_values.append(str(val).replace(",",";")) 
                f.write(",".join(row_values) + "\n")
        print(f"\n===== Grid search complete. Summary saved to: {summary_file_path} =====")
    else:
        print("\n===== Grid search complete. No metrics successfully collected. =====")
else:
    print("\n===== Grid search complete. No files processed or no metrics collected. =====") 