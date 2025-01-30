import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import os
import re
from sklearn.metrics import v_measure_score

# If needed: from src.analysis import ComputerClusterPerformance
from src.analysis import ComputerClusterPerformance

##############################################################################
#                          HELPER FUNCTIONS & CLASSES
##############################################################################

def get_bird_id_and_fold(filename: str) -> Tuple[str, str]:
    """
    Parse a file name like 'llb3_fold1.npz' to get:
      bird_id='llb3'
      fold_id='fold1'
    Adjust if your naming is different.
    """
    match = re.search(r'(llb\d+)_(fold\d+)\.npz', filename)
    if match:
        bird_id = match.group(1)
        fold_id = match.group(2)
    else:
        bird_id = "unknown_bird"
        fold_id = "unknown_fold"
    return bird_id, fold_id


class SequenceAnalyzer:
    """
    Utility to create a 'shared area' matrix:
      M[i,j] = number_of_frames where ground_truth == uniqueGT[i]
                                     AND predicted == uniquePred[j].
    Then we column‐normalize M so each column sums to 1 (where possible).
    
    Also ensures -1 is included in GT/pred sets (so unmatched frames appear).
    """
    @staticmethod
    def create_shared_area_matrix(
        ground_truth: np.ndarray,
        predicted:   np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          M (raw_counts),
          M_norm (column‐normalized),
          unique_gt,
          unique_pred,
          col_sums
        """
        # 1) Ensure -1 is in both sets
        if -1 not in ground_truth:
            ground_truth = np.concatenate([ground_truth, [-1]])
        if -1 not in predicted:
            predicted = np.concatenate([predicted, [-1]])
        
        unique_gt   = np.unique(ground_truth)
        unique_pred = np.unique(predicted)
        
        # 2) Build raw count matrix
        M = np.zeros((len(unique_gt), len(unique_pred)), dtype=int)
        for i, g in enumerate(unique_gt):
            for j, p in enumerate(unique_pred):
                M[i, j] = np.sum((ground_truth == g) & (predicted == p))
        
        # 3) Column‐normalize => each column sums to 1
        col_sums = M.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            M_norm = np.divide(M, col_sums, where=col_sums!=0)
        
        return M, M_norm, unique_gt, unique_pred, col_sums


def create_diagonal_label_mapping(
    normalized_matrix: np.ndarray,
    unique_gt_labels:  np.ndarray,
    unique_pred_labels: np.ndarray
) -> Tuple[Dict[int,int], List[int], List[int], List[int]]:
    """
    Use Hungarian to find assignment that maximizes sum of M_norm.
    Any GT label not matched => mapped to -1.
    Also returns row_ind, col_ind for logging, and leftover predicted labels.
    """
    cost_matrix = -normalized_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {}
    matched_gt = set()
    matched_pred = set()  # Track matched predicted labels
    
    for i in range(len(row_ind)):
        gt_idx = row_ind[i]
        pred_idx = col_ind[i]
        gt_lbl = int(unique_gt_labels[gt_idx])
        pd_lbl = int(unique_pred_labels[pred_idx])
        mapping[gt_lbl] = pd_lbl
        matched_gt.add(gt_lbl)
        matched_pred.add(pd_lbl)
    
    # Unmatched GT => -1
    for g in unique_gt_labels:
        g = int(g)
        if g not in matched_gt:
            mapping[g] = -1
    
    # Get leftover predicted labels
    leftover_pred = sorted(set(int(x) for x in unique_pred_labels) - matched_pred)
    
    return mapping, row_ind.tolist(), col_ind.tolist(), leftover_pred


def calculate_average_phrase_entropy(labels, target_label, dataset_indices):
    """
    For all contiguous runs of `target_label`,
    see what label we transition to next (only within the same song).
    Compute Shannon entropy of that distribution.
    """
    transition_counts = Counter()
    total_transitions = 0
    in_run = False
    
    for i in range(len(labels) - 1):
        if dataset_indices[i] != dataset_indices[i+1]:
            # new song => close run if in one
            if in_run:
                in_run = False
            continue
        
        curr = labels[i]
        nxt  = labels[i+1]
        if curr == target_label:
            in_run = True
        else:
            if in_run:
                transition_counts[nxt] += 1
                total_transitions += 1
                in_run = False
    
    if total_transitions == 0:
        return 0.0
    
    # Shannon entropy
    entropy = 0.0
    for count in transition_counts.values():
        p = count / total_transitions
        entropy -= p * np.log(p)
    return entropy


def calculate_average_phrase_length(labels, target_label, dataset_indices):
    """
    Average contiguous-run length for `target_label`, resetting if we cross a new song.
    """
    lengths = []
    run_len = 0
    prev_song_id = None
    
    for i, val in enumerate(labels):
        curr_song_id = dataset_indices[i]
        
        if prev_song_id is not None and curr_song_id != prev_song_id:
            # new song => close out any run
            if run_len > 0:
                lengths.append(run_len)
                run_len = 0
        
        if val == target_label:
            run_len += 1
        else:
            if run_len > 0:
                lengths.append(run_len)
                run_len = 0
        
        prev_song_id = curr_song_id
    
    # End final run
    if run_len > 0:
        lengths.append(run_len)
    
    return np.mean(lengths) if lengths else 0.0


def calculate_weighted_pearson(y_true, y_pred, weights):
    """
    Weighted Pearson correlation, weighting each sample by `weights`.
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    weights = np.array(weights)
    if len(y_true) == 0 or np.all(weights == 0):
        return 0.0
    
    mean_x = np.average(y_true, weights=weights)
    mean_y = np.average(y_pred, weights=weights)
    cov = np.average((y_true - mean_x)*(y_pred - mean_y), weights=weights)
    var_x = np.average((y_true - mean_x)**2, weights=weights)
    var_y = np.average((y_pred - mean_y)**2, weights=weights)
    denom = np.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov/denom


def find_files_in_folder(folder_path: str) -> List[str]:
    """
    Return sorted list of .npz files in folder_path.
    """
    all_paths = []
    for fn in os.listdir(folder_path):
        if fn.endswith('.npz'):
            all_paths.append(os.path.join(folder_path, fn))
    return sorted(all_paths)


##############################################################################
#                          PLOTTING UTILITIES
##############################################################################

def plot_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: str,
    x_labels=None,
    y_labels=None,
    cmap='viridis'
):
    """
    Basic matrix plot with no reordering.
    """
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    
    cax = ax.imshow(matrix, cmap=cmap, aspect='equal', origin='upper')
    fig.colorbar(cax, ax=ax, label='Value')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted Label Index")
    ax.set_ylabel("Ground Truth Label Index")
    
    if x_labels is not None and len(x_labels) == matrix.shape[1]:
        # Show every 2nd x-axis label
        x_ticks = np.arange(0, len(x_labels), 2)
        x_labels_spaced = [x_labels[i] if i < len(x_labels) else '' for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels_spaced, rotation=90)
    
    if y_labels is not None and len(y_labels) == matrix.shape[0]:
        # Show every 2nd y-axis label
        y_ticks = np.arange(0, len(y_labels), 2)
        y_labels_spaced = [y_labels[i] if i < len(y_labels) else '' for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels_spaced)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def diagonalize_and_plot_matrix(
    matrix: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    title: str,
    output_path: str
):
    """
    Reorder the matrix using Hungarian LSA on (-matrix.T) => maximizing sum of matrix,
    so matched row/col appear along the diagonal. We keep -1 in the row/col if present.
    """
    gt_labels   = np.array(gt_labels, dtype=int)
    pred_labels = np.array(pred_labels, dtype=int)
    
    cost_mat = -matrix.T
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    diag_mat = matrix[np.ix_(col_ind, row_ind)]
    
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    cax = ax.imshow(diag_mat, cmap='viridis', aspect='equal')
    fig.colorbar(cax, ax=ax, label='Value')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Reordered Predicted Label (Assigned to row's GT label)")
    ax.set_ylabel("Reordered Ground Truth Label")
    
    # Show every 2nd x-axis label
    x_ticks = np.arange(0, len(pred_labels), 2)
    x_labels = [pred_labels[row_ind[i]] if i < len(row_ind) else '' for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90)
    
    # Show every 2nd y-axis label
    y_ticks = np.arange(0, len(gt_labels), 2)
    y_labels = [gt_labels[col_ind[i]] if i < len(col_ind) else '' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def diagonalize_and_plot_matrix_full(
    matrix: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    title: str,
    output_path: str
):
    """Full reorder: matched pairs first, unmatched last."""
    cost_mat = -matrix.T
    row_idx, col_idx = linear_sum_assignment(cost_mat)
    
    # Get matched and unmatched indices
    matched_rows = list(col_idx)
    matched_cols = list(row_idx)
    all_rows = set(range(matrix.shape[0]))
    all_cols = set(range(matrix.shape[1]))
    leftover_rows = sorted(all_rows - set(matched_rows))
    leftover_cols = sorted(all_cols - set(matched_cols))
    
    # Reorder with unmatched at end
    new_rows = matched_rows + leftover_rows
    new_cols = matched_cols + leftover_cols
    M_full = matrix[np.ix_(new_rows, new_cols)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    cax = ax.imshow(M_full, cmap='viridis', aspect='equal')
    fig.colorbar(cax, ax=ax, label='Value')
    ax.set_title(title + " (Full Reorder)", fontsize=14)
    ax.set_xlabel("Pred (matched→unmatched)")
    ax.set_ylabel("GT (matched→unmatched)")
    
    # Show every 2nd x-axis label
    x_ticks = np.arange(0, len(pred_labels), 2)
    x_labels = [pred_labels[new_cols[i]] if i < len(new_cols) else '' for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90)
    
    # Show every 2nd y-axis label
    y_ticks = np.arange(0, len(gt_labels), 2)
    y_labels = [gt_labels[new_rows[i]] if i < len(new_rows) else '' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


##############################################################################
#                          MAIN ANALYSIS FLOW
##############################################################################

def process_all_folds(files: List[str],
                      smoothing_window: int,
                      labels_path: str,
                      output_dir: str = None,
                      do_step_plots: bool = False
                     ) -> Tuple[
                         List[float], List[float],  # phrase-level entropies
                         List[float], List[float],  # phrase-level lengths
                         List[str],   List[str],    # phrase-level bird/fold IDs
                         List[int],                 # phrase-level counts
                         float, float,              # overall FER (mapped-only), overall FER (-1=error)
                         Dict[Tuple[str,int], float]# (bird, gt_label): FER (mapped-only)
                     ]:
    """
    For each fold:
      1) Convert GT labels -> phrase
      2) Fill + smooth predicted
      3) Build matrix, diagonalize => mapping
      4) Compute mismatch in two ways
      5) Compute phrase-level stats (entropy, length)
      6) Possibly produce step-by-step outputs (plots + text logs).

    Returns:
      - phrase-level stats for correlation
      - two overall FER
      - a dict of per-bird/per-label mismatch% (mapped-only).
    """
    ground_truth_entropies = []
    hdbscan_entropies      = []
    ground_truth_lengths   = []
    hdbscan_lengths        = []
    bird_ids               = []
    fold_ids               = []
    phrase_counts          = []

    # Summation for mismatch
    total_frames_global_mapped   = 0
    total_mismatch_global_mapped = 0
    total_frames_global_any      = 0
    total_mismatch_global_any    = 0

    # Mismatch tracking: (bird, label)
    per_bird_label_counts    = defaultdict(int)
    per_bird_label_mismatches= defaultdict(int)
    
    analyzer = SequenceAnalyzer()

    def smooth_per_song(labels: np.ndarray,
                        dataset_indices: np.ndarray,
                        window_size: int,
                        computer_cluster: ComputerClusterPerformance):
        """
        For each 'song_id', do majority_vote smoothing over that subset.
        """
        if window_size <= 1:
            return labels.copy()
        smoothed = np.zeros_like(labels)
        unique_song_ids = np.unique(dataset_indices)
        for song_id in unique_song_ids:
            mask = (dataset_indices == song_id)
            sub_arr = labels[mask]
            try:
                sub_smoothed = computer_cluster.majority_vote(sub_arr, window_size)
                smoothed[mask] = sub_smoothed
            except ValueError as e:
                print(f"Warning: majority_vote failed for {song_id}: {str(e)}")
                smoothed[mask] = sub_arr
        return smoothed
    
    for fpath in tqdm(files, desc=f"Processing folds (window={smoothing_window})"):
        fname = os.path.basename(fpath)
        bird_id, fold_id = get_bird_id_and_fold(fname)
        
        data = np.load(fpath)
        ground_truth_labels = data['ground_truth_labels']
        hdbscan_labels      = data['hdbscan_labels']
        dataset_indices     = data['dataset_indices']
        
        computer_cluster = ComputerClusterPerformance(labels_path=labels_path)
        # Convert GT => phrase
        gt_phrase = computer_cluster.syllable_to_phrase_labels(ground_truth_labels, silence=0)
        
        # Fill noise => majority vote smoothing
        pred_filled = computer_cluster.fill_noise_with_nearest_label(hdbscan_labels)
        pred_smooth = smooth_per_song(pred_filled, dataset_indices, smoothing_window, computer_cluster)
        
        if len(pred_smooth)==0 or np.all(pred_smooth==-1):
            continue
        
        # Print unique label counts (optional)
        unique_gt_labels = np.unique(gt_phrase)
        unique_pred_labels = np.unique(pred_smooth)
        print(f"\nFile: {fname}")
        print(f"Unique GT labels ({len(unique_gt_labels)}): {sorted(unique_gt_labels)}")
        print(f"Unique Predicted labels ({len(unique_pred_labels)}): {sorted(unique_pred_labels)}")
        
        # 1) Build matrix
        M_raw, M_norm, uniq_gt, uniq_pred, col_sums = analyzer.create_shared_area_matrix(gt_phrase, pred_smooth)
        if len(uniq_gt)==0 or len(uniq_pred)==0:
            continue
        
        # 2) Hungarian => mapping
        mapping, row_ind, col_ind, leftover_pred = create_diagonal_label_mapping(
            M_norm, uniq_gt, uniq_pred)
        
        # Force any frames using leftover predicted labels to -1
        for lv in leftover_pred:
            pred_smooth[pred_smooth == lv] = -1
        
        mapped_gt = np.array([mapping[g] for g in gt_phrase])
        total_frames = len(mapped_gt)
        if total_frames == 0:
            continue
        
        # 3) mismatch
        mapped_mask   = (mapped_gt != -1)
        mismatch_mapped = np.sum((mapped_mask) & (mapped_gt != pred_smooth))
        frames_mapped   = np.sum(mapped_mask)
        
        mismatch_any = np.sum(mapped_gt != pred_smooth)
        
        total_frames_global_mapped   += frames_mapped
        total_mismatch_global_mapped += mismatch_mapped
        total_frames_global_any      += total_frames
        total_mismatch_global_any    += mismatch_any
        
        # per-bird mismatch
        for i in range(total_frames):
            gt_lbl  = gt_phrase[i]
            pred_lbl= pred_smooth[i]
            if gt_lbl == -1:
                continue
            if mapping[gt_lbl] == -1:
                continue
            per_bird_label_counts[(bird_id, gt_lbl)] += 1
            if mapped_gt[i] != pred_lbl:
                per_bird_label_mismatches[(bird_id, gt_lbl)] += 1
        
        # 4) phrase-level entropies & lengths
        uniq_mapped = np.unique(mapped_gt)
        for label in uniq_mapped:
            if label == -1:
                continue
            gt_entropy = calculate_average_phrase_entropy(mapped_gt, label, dataset_indices)
            hd_entropy = calculate_average_phrase_entropy(pred_smooth, label, dataset_indices)
            gt_length  = calculate_average_phrase_length(mapped_gt, label, dataset_indices)
            hd_length  = calculate_average_phrase_length(pred_smooth, label, dataset_indices)
            pcount = np.sum(mapped_gt == label)
            
            ground_truth_entropies.append(gt_entropy)
            hdbscan_entropies.append(hd_entropy)
            ground_truth_lengths.append(gt_length)
            hdbscan_lengths.append(hd_length)
            phrase_counts.append(pcount)
            
            bird_ids.append(bird_id)
            fold_ids.append(fold_id)
        
        # 5) step‐by‐step logs
        if output_dir and do_step_plots:
            fold_subdir = os.path.join(output_dir, f"{bird_id}_{fold_id}")
            os.makedirs(fold_subdir, exist_ok=True)
            
            plot_matrix(M_raw,
                f"[Raw Co-Occurrences] {bird_id}_{fold_id} (win={smoothing_window})",
                os.path.join(fold_subdir, "01_M_raw.png"),
                x_labels=uniq_pred, y_labels=uniq_gt
            )
            plot_matrix(M_norm,
                f"[Normalized] {bird_id}_{fold_id} (win={smoothing_window})",
                os.path.join(fold_subdir, "02_M_norm.png"),
                x_labels=uniq_pred, y_labels=uniq_gt
            )
            
            diagonalize_and_plot_matrix(
                M_raw, uniq_gt, uniq_pred,
                f"[Raw Counts Diagonalized] {bird_id}_{fold_id} (win={smoothing_window})",
                os.path.join(fold_subdir, "03_M_raw_diag.png")
            )
            diagonalize_and_plot_matrix(
                M_norm, uniq_gt, uniq_pred,
                f"[Norm Diagonalized] {bird_id}_{fold_id} (win={smoothing_window})",
                os.path.join(fold_subdir, "04_M_norm_diag.png")
            )
            
            # Add full reorder plots
            diagonalize_and_plot_matrix_full(
                M_raw, uniq_gt, uniq_pred,
                f"[Raw Full Reorder] {bird_id}_{fold_id} (win={smoothing_window})",
                os.path.join(fold_subdir, "05_M_raw_fullreorder.png")
            )
            diagonalize_and_plot_matrix_full(
                M_norm, uniq_gt, uniq_pred,
                f"[Norm Full Reorder] {bird_id}_{fold_id} (win={smoothing_window})",
                os.path.join(fold_subdir, "06_M_norm_fullreorder.png")
            )
            
            # write text logs
            mapping_log_path = os.path.join(fold_subdir, f"mapping_steps_{bird_id}_{fold_id}.txt")
            with open(mapping_log_path, 'w') as txtf:
                txtf.write(f"==== STEPS for {bird_id}_{fold_id} (win={smoothing_window}) ====\n\n")
                txtf.write("1) M_raw:\n"+ str(M_raw) +"\n\n")
                txtf.write("2) M_norm:\n"+ str(M_norm) +"\n\n")
                cost_mat_str = str(-M_norm)
                txtf.write("3) Cost Matrix = -M_norm:\n"+ cost_mat_str +"\n\n")
                txtf.write("3) row_ind -> col_ind from Hungarian:\n")
                txtf.write(f"   row_ind={row_ind}\n   col_ind={col_ind}\n\n")
                txtf.write("   => row_ind[i] matched with col_ind[i]\n\n")
                
                txtf.write("4) Final Mapping (GT_label -> Pred_label):\n")
                for g_lbl in sorted(mapping.keys()):
                    txtf.write(f"   {g_lbl} -> {mapping[g_lbl]}\n")
                
                txtf.write("\nLeftover predicted labels (unused):\n")
                for p_lbl in leftover_pred:
                    txtf.write(f"   {p_lbl}\n")
    
    # 6) overall mismatch
    if total_frames_global_mapped>0:
        overall_fer_mapped = 100.0 * (total_mismatch_global_mapped / total_frames_global_mapped)
    else:
        overall_fer_mapped = np.nan
    
    if total_frames_global_any>0:
        overall_fer_any = 100.0 * (total_mismatch_global_any / total_frames_global_any)
    else:
        overall_fer_any = np.nan
    
    # 7) per-bird phrase FER
    per_bird_phrase_fer = {}
    for (b,lbl), tcount in per_bird_label_counts.items():
        mism = per_bird_label_mismatches[(b,lbl)]
        fer_ = 100.0*(mism/tcount) if tcount>0 else np.nan
        per_bird_phrase_fer[(b,lbl)] = fer_
    
    # 8) summary
    if output_dir is not None:
        summ_path = os.path.join(output_dir, f'summary_window{str(smoothing_window)}.txt')
        with open(summ_path, 'w') as f:
            f.write(f"=== Summary for Smoothing Window = {smoothing_window} ===\n\n")
            f.write(" -- Overall Frame Error Rates --\n")
            f.write(f"    1) Mapped-Only FER  : {overall_fer_mapped:.2f}%\n")
            f.write(f"    2) Include -1 in FER: {overall_fer_any:.2f}%\n\n")
            
            f.write(" -- Per-Bird Phrase FER (mapped only) --\n")
            birds_in_dict = sorted({k[0] for k in per_bird_phrase_fer.keys()})
            for b in birds_in_dict:
                f.write(f"    Bird '{b}':\n")
                labels_for_b = sorted(lbl for (bb,lbl) in per_bird_phrase_fer.keys() if bb==b)
                for lbl in labels_for_b:
                    ferval = per_bird_phrase_fer[(b,lbl)]
                    f.write(f"       Label {lbl:3d}: {ferval:5.2f}% mismatch\n")
                f.write("\n")
    
    return (ground_truth_entropies, hdbscan_entropies,
            ground_truth_lengths, hdbscan_lengths,
            bird_ids, fold_ids, phrase_counts,
            overall_fer_mapped, overall_fer_any, per_bird_phrase_fer)


def calculate_max_values_for_folder(files: List[str], labels_path: str) -> Tuple[float,float]:
    """
    Finds max possible phrase entropy and phrase length in GT across all folds.
    """
    max_entropy = 0.0
    max_length  = 0.0
    computer_cluster = ComputerClusterPerformance(labels_path=labels_path)
    
    for fpath in tqdm(files, desc="Calculating max values"):
        data = np.load(fpath)
        gt_labels = data['ground_truth_labels']
        gt_phrase = computer_cluster.syllable_to_phrase_labels(gt_labels, silence=0)
        
        # This function also needs dataset_indices to handle boundaries
        # If you want to be consistent with your approach, do the same:
        dataset_indices = data['dataset_indices']
        
        for lbl in np.unique(gt_phrase):
            if lbl == 0:  # skip silence
                continue
            e = calculate_average_phrase_entropy(gt_phrase, lbl, dataset_indices)
            l = calculate_average_phrase_length(gt_phrase, lbl, dataset_indices)
            if e > max_entropy:
                max_entropy = e
            if l > max_length:
                max_length = l
    
    return max_entropy, max_length


def plot_correlation_comparisons(
    ground_truth_entropies, hdbscan_entropies,
    ground_truth_lengths,   hdbscan_lengths,
    bird_ids, fold_ids,
    phrase_counts,
    per_bird_phrase_fer: Dict[Tuple[str,int], float],
    smoothing_window, max_entropy, max_length,
    output_dir=None
):
    """
    Scatter plots + per‐bird summary for phrase-level correlation (Entropy, Length).
    """

    plt.style.use('default')
    FONT_SIZE = 18
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.linewidth': 2,
        'axes.grid': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Convert to arrays
    gte = np.array(ground_truth_entropies)
    hte = np.array(hdbscan_entropies)
    gtl = np.array(ground_truth_lengths)
    htl = np.array(hdbscan_lengths)
    pcounts = np.array(phrase_counts)
    birds   = np.array(bird_ids)
    folds   = np.array(fold_ids)
    
    # Dot sizes
    min_size = 100
    max_size = 800
    if len(pcounts) > 0:
        rng = pcounts.max() - pcounts.min() + 1e-9
        norm_counts = (pcounts - pcounts.min()) / rng
    else:
        norm_counts = np.zeros_like(pcounts)
    sizes = min_size + norm_counts*(max_size - min_size)
    
    ############################################################################
    # ENTROPY CORRELATION
    ############################################################################
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    
    line_x = np.linspace(0, max_entropy, 200)
    ax.plot(line_x, line_x, 'k--', alpha=0.5, zorder=1)
    
    unique_birds = np.unique(birds)
    palette = sns.color_palette("husl", n_colors=len(unique_birds))
    bird_color_map = dict(zip(unique_birds, palette))
    
    for i in range(len(gte)):
        b = birds[i]
        clr = bird_color_map[b]
        ax.scatter(gte[i], hte[i], s=sizes[i],
                   color=clr, alpha=0.7,
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    r_pearson = calculate_weighted_pearson(gte, hte, pcounts)
    r_sq = r_pearson**2
    
    ax.set_xlabel("GT Phrase Entropy", fontsize=FONT_SIZE)
    ax.set_ylabel("HDBSCAN Phrase Entropy", fontsize=FONT_SIZE)
    ax.set_title(f"Phrase Entropy Comparison\n(Smoothing={smoothing_window}, r={r_pearson:.3f}, r²={r_sq:.3f})",
                 fontsize=FONT_SIZE)
    ax.set_xlim(0, max_entropy)
    ax.set_ylim(0, max_entropy)
    
    from matplotlib.lines import Line2D
    legend_elems = []
    for b in unique_birds:
        legend_elems.append(Line2D([0],[0],
                                   marker='o', color='black', label=b,
                                   markersize=8, markerfacecolor=bird_color_map[b]))
    ax.legend(handles=legend_elems, title="Bird ID", fontsize=FONT_SIZE-2, loc='lower right')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'phrase_entropy_correlation_{smoothing_window}.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'phrase_entropy_correlation_{smoothing_window}.svg'),
                    format='svg', bbox_inches='tight')
    else:
        plt.savefig(f'phrase_entropy_correlation_{smoothing_window}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    ############################################################################
    # LENGTH CORRELATION
    ############################################################################
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    
    line_x = np.linspace(0, max_length, 200)
    ax.plot(line_x, line_x, 'k--', alpha=0.5, zorder=1)
    
    for i in range(len(gtl)):
        b = birds[i]
        clr = bird_color_map[b]
        ax.scatter(gtl[i], htl[i], s=sizes[i],
                   color=clr, alpha=0.7,
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    r_pearson_len = calculate_weighted_pearson(gtl, htl, pcounts)
    r_sq_len = r_pearson_len**2
    
    ax.set_xlabel("GT Phrase Length", fontsize=FONT_SIZE)
    ax.set_ylabel("HDBSCAN Phrase Length", fontsize=FONT_SIZE)
    ax.set_title(f"Phrase Length Comparison\n(Smoothing={smoothing_window}, r={r_pearson_len:.3f}, r²={r_sq_len:.3f})",
                 fontsize=FONT_SIZE)
    ax.set_xlim(0, max_length)
    ax.set_ylim(0, max_length)
    
    ax.legend(handles=legend_elems, title="Bird ID", fontsize=FONT_SIZE-2, loc='lower right')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'phrase_length_correlation_{smoothing_window}.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'phrase_length_correlation_{smoothing_window}.svg'),
                    format='svg', bbox_inches='tight')
    else:
        plt.savefig(f'phrase_length_correlation_{smoothing_window}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    ############################################################################
    #   PER-BIRD SUMMARY
    ############################################################################
    print("\n================= PER-BIRD SUMMARY STATS =================\n")
    from collections import defaultdict
    by_bird = defaultdict(lambda: {
        "gt_entropy": [],
        "hd_entropy": [],
        "gt_length":  [],
        "hd_length":  [],
        "counts":     []
    })
    for i in range(len(birds)):
        b = birds[i]
        by_bird[b]["gt_entropy"].append(gte[i])
        by_bird[b]["hd_entropy"].append(hte[i])
        by_bird[b]["gt_length"].append(gtl[i])
        by_bird[b]["hd_length"].append(htl[i])
        by_bird[b]["counts"].append(pcounts[i])
    
    for b in sorted(by_bird.keys()):
        b_gt_e = np.array(by_bird[b]["gt_entropy"])
        b_hd_e = np.array(by_bird[b]["hd_entropy"])
        b_gt_l = np.array(by_bird[b]["gt_length"])
        b_hd_l = np.array(by_bird[b]["hd_length"])
        b_wts  = np.array(by_bird[b]["counts"])
        
        r_e  = calculate_weighted_pearson(b_gt_e, b_hd_e, b_wts)
        r2_e = r_e**2
        r_l  = calculate_weighted_pearson(b_gt_l, b_hd_l, b_wts)
        r2_l = r_l**2
        avg_r2= (r2_e + r2_l)/2.0
        
        print(f"  Bird: {b}")
        print(f"    Weighted r^2(Entropy) = {r2_e:.3f}")
        print(f"    Weighted r^2(Length)  = {r2_l:.3f}")
        print(f"    Average r^2           = {avg_r2:.3f}\n")


def plot_v_measure_by_window(window_results, entropy_results, length_results, output_dir=None):
    """
    Plot V-measure on right axis, Entropy and Length correlations on left axis.
    window_results = [(window_size, v_score), ...]
    entropy_results = [(window_size, r_entropy), ...]
    length_results = [(window_size, r_length), ...]
    """
    ws = [x[0] for x in window_results]
    vs = [x[1] for x in window_results]
    re = [x[1] for x in entropy_results]
    rl = [x[1] for x in length_results]
    
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    # Create second y-axis sharing same x-axis
    ax2 = ax1.twinx()
    
    # Plot correlations on left axis
    l1 = ax1.plot(ws, re, 's-', color='#2ecc71', linewidth=2, markersize=8, label='Entropy r')
    l2 = ax1.plot(ws, rl, '^-', color='#3498db', linewidth=2, markersize=8, label='Duration r')
    
    # Plot V-measure on right axis
    l3 = ax2.plot(ws, vs, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='V-measure')
    
    # Set labels
    ax1.set_xlabel("Smoothing Window Size")
    ax1.set_ylabel("Correlation (r)")
    ax2.set_ylabel("V-measure")
    
    plt.title("Evaluation Metrics vs Window Size")
    
    # Add grid but only for left axis
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    
    # Calculate means for text box
    mean_v = np.nanmean(vs) if len(vs)>0 else 0.0
    mean_re = np.nanmean(re) if len(re)>0 else 0.0
    mean_rl = np.nanmean(rl) if len(rl)>0 else 0.0
    
    text_str = (f'Mean V-score: {mean_v:.3f}\n'
                f'Mean Entropy r: {mean_re:.3f}\n'
                f'Mean Duration r: {mean_rl:.3f}')
    
    # Position text box
    ax1.text(0.02, 0.98, text_str,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'metrics_by_window.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'metrics_by_window.svg'), format='svg')
    else:
        plt.savefig('metrics_by_window.png', dpi=300)
        plt.savefig('metrics_by_window.svg', format='svg')
    plt.close()


##############################################################################
#                               MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    folder_path = "/media/george-vengrovski/66AA-9C0A/temp"
    output_dir  = "results/proxy_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different window sizes
    window_dirs = {}
    
    # Smoothing windows to test
    # (Example range: 0, 252, 504, etc.)
    window_sizes = list(range(0, 501, 50))
    
    for wsize in window_sizes:
        window_dirs[wsize] = os.path.join(output_dir, f'window_{wsize}')
        os.makedirs(window_dirs[wsize], exist_ok=True)
    
    # Create best_window directory
    best_window_dir = os.path.join(output_dir, 'best_window')
    os.makedirs(best_window_dir, exist_ok=True)
    
    # 1) Gather fold files
    all_npz_files = find_files_in_folder(folder_path)
    
    # 2) Compute global maxima for phrase entropy/length (to scale scatter plots)
    max_entropy, max_length = calculate_max_values_for_folder(all_npz_files, labels_path=folder_path)
    
    all_results = []
    v_measure_results = []
    entropy_results = []
    length_results = []
    
    best_wsize = None
    best_v_score = -999
    
    ###########################################################################
    # Run each window, store intermediate data
    ###########################################################################
    for wsize in tqdm(window_sizes, desc="Testing smoothing windows"):
        current_window_dir = window_dirs[wsize]
        
        # We'll do step_plots for wsize=0 (just as an example)
        do_plots = (wsize == 0)
        
        (gt_e, hd_e, gt_l, hd_l, 
         birds, folds, counts,
         overall_fer_mapped, overall_fer_any, bird_phrase_fer) = process_all_folds(
             all_npz_files, wsize, labels_path=folder_path,
             output_dir=current_window_dir,
             do_step_plots=do_plots
        )
        
        # Weighted correlations
        r_e = calculate_weighted_pearson(np.array(gt_e), np.array(hd_e), np.array(counts))
        r_l = calculate_weighted_pearson(np.array(gt_l), np.array(hd_l), np.array(counts))
        
        # For demonstration, let's do a v_measure with just the first file
        # If you want a global v_measure, you'd do a more careful approach
        if len(all_npz_files) > 0:
            data = np.load(all_npz_files[0])
            comp_cluster = ComputerClusterPerformance(labels_path=folder_path)
            gtp = comp_cluster.syllable_to_phrase_labels(data['ground_truth_labels'], silence=0)
            # fill & smooth
            pf   = comp_cluster.fill_noise_with_nearest_label(data['hdbscan_labels'])
            ps   = comp_cluster.majority_vote(pf, wsize)
            v_score = v_measure_score(gtp, ps)
        else:
            v_score = 0.0
        
        all_results.append((wsize, r_e, r_l, v_score, overall_fer_mapped, overall_fer_any))
        v_measure_results.append((wsize, v_score))
        entropy_results.append((wsize, r_e))
        length_results.append((wsize, r_l))
        
        if v_score > best_v_score:
            best_v_score = v_score
            best_wsize = wsize
    
    ###########################################################################
    # Re-run the best window w/ step_plots=True
    ###########################################################################
    best_window_corr_dir = window_dirs[best_wsize]
    (gt_e_best, hd_e_best, gt_l_best, hd_l_best,
     birds_best, folds_best, counts_best,
     overall_fer_mapped_best, overall_fer_any_best, bird_phrase_fer_best) = \
         process_all_folds(
             all_npz_files, best_wsize, labels_path=folder_path,
             output_dir=best_window_corr_dir,
             do_step_plots=True
         )
    
    ###########################################################################
    # Plot metrics across windows
    ###########################################################################
    plot_v_measure_by_window(v_measure_results, entropy_results, length_results, output_dir=output_dir)
    
    ###########################################################################
    # Build a final summary across all windows
    ###########################################################################
    summary_path = os.path.join(output_dir, "all_windows_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("============================================================\n")
        f.write("          ALL WINDOWS SUMMARY (V-Measure + Correlation)     \n")
        f.write("============================================================\n\n")
        
        header = (
            "Window |   r_e   |   r_l   | v_score |   FER(mapped)%  |   FER(-1)%  \n"
            "-------+---------+---------+---------+-----------------+------------\n"
        )
        f.write(header)
        
        for i, (wsize, re_, rl_, v_, fer_map, fer_any) in enumerate(all_results):
            f.write(
                f"{wsize:6d} | {re_:7.3f} | {rl_:7.3f} | {v_:7.3f} |"
                f"     {fer_map:7.2f}%      |   {fer_any:7.2f}%   \n"
            )
        
        f.write("\n\n--- BEST WINDOW SELECTION ---\n\n")
        f.write(f"  * The best window by V-measure is: {best_wsize}  (v_score={best_v_score:.3f})\n\n")
        
        f.write("=== FINAL RESULTS for BEST WINDOW ===\n")
        f.write(f"Window = {best_wsize}\n")
        f.write(f"Overall FER (mapped only) : {overall_fer_mapped_best:.2f}%\n")
        f.write(f"Overall FER (with -1 err) : {overall_fer_any_best:.2f}%\n\n")
        
        f.write("Per-Bird, Per-Label FER (mapped only):\n")
        f.write("--------------------------------------\n")
        birds_in_dict = sorted({k[0] for k in bird_phrase_fer_best.keys()})
        for b in birds_in_dict:
            f.write(f"  Bird '{b}':\n")
            labels_for_b = sorted(lbl for (bb,lbl) in bird_phrase_fer_best.keys() if bb==b)
            for lbl in labels_for_b:
                ferval = bird_phrase_fer_best[(b,lbl)]
                f.write(f"    Label {lbl:2d} => mismatch: {ferval:.2f}%\n")
            f.write("\n")
    
    ###########################################################################
    # Also produce correlation plots for the best window
    ###########################################################################
    plot_correlation_comparisons(
        gt_e_best, hd_e_best, gt_l_best, hd_l_best,
        birds_best, folds_best, counts_best,
        bird_phrase_fer_best,
        best_wsize, max_entropy, max_length,
        output_dir=best_window_corr_dir
    )
    
    print("\nALL DONE!")
    print(" * Detailed per-window summaries are in each window_{N} folder.")
    print(" * A consolidated 'all_windows_summary.txt' is in:", output_dir)
    print(" * The best window's step-by-step diagonalization logs, plots, etc. can be found in", best_window_corr_dir)
