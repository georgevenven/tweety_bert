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

# If you need your local code:
# from src.analysis import ComputerClusterPerformance
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
      M[i,j] = #frames where ground_truth == uniqueGT[i]
                                AND predicted == uniquePred[j].
    Then we column‐normalize M so each column sums to 1 (where possible).
    
    Also includes an extra row/column for '-1' if not already present,
    so we can visualize unmatched GT/pred in the final matrix plots.
    """
    @staticmethod
    def create_shared_area_matrix(
        ground_truth: np.ndarray,
        predicted:   np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # 1) Force inclusion of -1 if not present, so unmatched frames show up
        if -1 not in ground_truth:
            ground_truth = np.concatenate([ground_truth, [-1]])
        if -1 not in predicted:
            predicted = np.concatenate([predicted,    [-1]])
        
        unique_gt   = np.unique(ground_truth)
        unique_pred = np.unique(predicted)
        
        # 2) Build matrix
        M = np.zeros((len(unique_gt), len(unique_pred)), dtype=int)
        for i, g in enumerate(unique_gt):
            for j, p in enumerate(unique_pred):
                # Count frames matching (g, p)
                M[i, j] = np.sum((ground_truth == g) & (predicted == p))
        
        # 3) Column‐normalize so sum=1
        col_sums = M.sum(axis=0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            M_norm = np.divide(M, col_sums, where=col_sums!=0)
        
        return M_norm, unique_gt, unique_pred


def create_diagonal_label_mapping(
    normalized_matrix: np.ndarray,
    unique_gt_labels:  np.ndarray,
    unique_pred_labels: np.ndarray
) -> Dict[int, int]:
    """
    Use Hungarian algorithm to find assignment maximizing overlap.
    unmatched GT => mapped to -1
    unmatched Pred => also -1, though we typically only store GT->Pred.
    """
    cost_matrix = -normalized_matrix  # Minimizing negative => maximizing positive
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    mapping = {}
    matched_gt = set()
    matched_pred = set()
    
    for i in range(len(row_ind)):
        gt_idx  = row_ind[i]
        pred_idx= col_ind[i]
        gt_lbl  = unique_gt_labels[gt_idx]
        pd_lbl  = unique_pred_labels[pred_idx]
        mapping[gt_lbl] = pd_lbl
        matched_gt.add(gt_lbl)
        matched_pred.add(pd_lbl)
    
    # Unmatched GT => -1
    for g in unique_gt_labels:
        if g not in matched_gt:
            mapping[g] = -1
    
    # If you want to store unmatched predicted => -1, you could
    # but typically we only do GT->pred in this mapping
    return mapping


def calculate_average_phrase_entropy(labels, target_label):
    """
    For all contiguous runs of `target_label`,
    see what label we transition to next.
    Then compute Shannon entropy of that distribution.
    """
    transition_counts = Counter()
    total_transitions = 0
    in_run = False
    
    for i in range(len(labels) - 1):
        curr = labels[i]
        nxt  = labels[i+1]
        if curr == target_label:
            in_run = True
        else:
            if in_run:
                transition_counts[nxt] += 1
                total_transitions += 1
                in_run = False
    
    # If ended in target_label, no next_label => no new transition.
    if total_transitions == 0:
        return 0.0
    
    # Shannon entropy
    entropy = 0.0
    for count in transition_counts.values():
        p = count / total_transitions
        entropy -= p * np.log(p)
    return entropy


def calculate_average_phrase_length(labels, target_label):
    """
    Average contiguous-run length for `target_label`.
    """
    lengths = []
    run_len = 0
    for val in labels:
        if val == target_label:
            run_len += 1
        else:
            if run_len > 0:
                lengths.append(run_len)
                run_len = 0
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
    Gather all .npz files in that folder, sorted.
    """
    all_paths = []
    for fn in os.listdir(folder_path):
        if fn.endswith('.npz'):
            all_paths.append(os.path.join(folder_path, fn))
    return sorted(all_paths)


##############################################################################
#                          MAIN ANALYSIS FLOW
##############################################################################

def process_all_folds(files: List[str], 
                      smoothing_window: int, 
                      labels_path: str, 
                      output_dir: str = None
                     ) -> Tuple[
                         List[float], List[float],  # phrase-level entropies
                         List[float], List[float],  # phrase-level lengths
                         List[str],   List[str],    # phrase-level bird/fold IDs
                         List[int],                 # phrase-level counts
                         float,                     # overall FER
                         Dict[Tuple[str,int], float]# (bird, gt_label) -> FER
                     ]:
    """
    For each fold in `files`:
      1) Convert GT labels -> phrase (bird-specific)
      2) Fill + smooth predicted
      3) Build normalized matrix (with -1 row/col)
      4) Diagonalize => mapping
      5) Per-bird, per-GT-label mismatch => FER
      6) Phrase-level metrics for correlation (entropy, length)

    Returns:
      - phrase-level arrays (lists).
      - an overall FER across all folds/birds
      - a dict {(bird, gt_label): fer%}
    """
    # --- PHRASE-level arrays ---
    ground_truth_entropies = []
    hdbscan_entropies      = []
    ground_truth_lengths   = []
    hdbscan_lengths        = []
    bird_ids               = []
    fold_ids               = []
    phrase_counts          = []

    # Overall mismatch
    total_frames_global   = 0
    total_mismatch_global = 0

    # Per-bird, per-label mismatch
    # Example: per_bird_label_counts[(bird_id, gt_label)] = total frames
    per_bird_label_counts    = defaultdict(int)
    per_bird_label_mismatches= defaultdict(int)
    
    analyzer = SequenceAnalyzer()

    def smooth_per_song(labels: np.ndarray,
                        dataset_indices: np.ndarray,
                        window_size: int,
                        computer_cluster: ComputerClusterPerformance):
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
        
        # Create matrix & diagonalize
        M_norm, uniq_gt, uniq_pred = analyzer.create_shared_area_matrix(gt_phrase, pred_smooth)
        if len(uniq_gt)==0 or len(uniq_pred)==0:
            continue
        
        mapping = create_diagonal_label_mapping(M_norm, uniq_gt, uniq_pred)
        # mapped_gt => the predicted label that GT label maps to
        mapped_gt = np.array([mapping[g] for g in gt_phrase])

        total_frames = len(mapped_gt)
        if total_frames > 0:
            # Overall mismatch
            mismatch = np.sum(mapped_gt != pred_smooth)
            total_frames_global   += total_frames
            total_mismatch_global += mismatch

            # Per-bird label mismatch
            # If mapping says label => -1 => 100% mismatch for that label's frames
            for i in range(total_frames):
                gt_lbl  = gt_phrase[i]
                pred_lbl= pred_smooth[i]
                if gt_lbl == -1:
                    continue  # skip if you treat them as no-labeled frames
                
                per_bird_label_counts[(bird_id, gt_lbl)] += 1
                
                # If mapping[gt_lbl] == -1 => unmatched => all frames mismatch
                # or if pred_lbl != mapping[gt_lbl]
                # Actually simpler: mismatch if mapped_gt[i] != pred_lbl
                if mapped_gt[i] != pred_lbl:
                    per_bird_label_mismatches[(bird_id, gt_lbl)] += 1
        
        # Now phrase-level entropies/lengths for correlation
        uniq_mapped = np.unique(mapped_gt)
        for label in uniq_mapped:
            if label == -1:
                continue
            # measure entropies/length
            gt_entropy = calculate_average_phrase_entropy(mapped_gt, label)
            hd_entropy = calculate_average_phrase_entropy(pred_smooth, label)
            gt_length  = calculate_average_phrase_length(mapped_gt, label)
            hd_length  = calculate_average_phrase_length(pred_smooth, label)
            pcount = np.sum(mapped_gt == label)
            
            ground_truth_entropies.append(gt_entropy)
            hdbscan_entropies.append(hd_entropy)
            ground_truth_lengths.append(gt_length)
            hdbscan_lengths.append(hd_length)
            phrase_counts.append(pcount)
            
            bird_ids.append(bird_id)
            fold_ids.append(fold_id)
    
    # ---------------------------------------------
    #  Compute overall FER
    # ---------------------------------------------
    if total_frames_global>0:
        overall_fer = (total_mismatch_global / total_frames_global)*100.0
    else:
        overall_fer = np.nan
    
    # ---------------------------------------------
    #  Build final dict: (bird, label)-> mismatch%
    # ---------------------------------------------
    per_bird_phrase_fer = {}
    for (b, lbl), total_count in per_bird_label_counts.items():
        mismatches = per_bird_label_mismatches[(b,lbl)]
        fer_pct = (mismatches / total_count)*100.0 if total_count>0 else np.nan
        per_bird_phrase_fer[(b,lbl)] = fer_pct
    
    # Optionally write a text summary
    if output_dir is not None:
        summ_path = os.path.join(output_dir, f'summary_window{str(smoothing_window)}.txt')
        with open(summ_path, 'w') as f:
            f.write(f"Smoothing Window = {smoothing_window}\n")
            f.write(f"OVERALL FER = {overall_fer:.2f}%\n\n")
            f.write("Per-Bird Phrase FER:\n")
            f.write("-"*40 + "\n")
            # group by bird
            birds_in_dict = sorted(set([k[0] for k in per_bird_phrase_fer.keys()]))
            for b in birds_in_dict:
                f.write(f" Bird: {b}\n")
                # gather labels for this bird
                labels_for_b = sorted([lbl for (bb,lbl) in per_bird_phrase_fer.keys() if bb==b])
                for lbl in labels_for_b:
                    ferval = per_bird_phrase_fer[(b,lbl)]
                    f.write(f"   Label {lbl}: {ferval:.2f}%\n")
                f.write("\n")
    
    return (ground_truth_entropies, hdbscan_entropies,
            ground_truth_lengths, hdbscan_lengths,
            bird_ids, fold_ids, phrase_counts,
            overall_fer, per_bird_phrase_fer)


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
        
        for lbl in np.unique(gt_phrase):
            if lbl==0:  # skip silence
                continue
            e = calculate_average_phrase_entropy(gt_phrase, lbl)
            l = calculate_average_phrase_length(gt_phrase, lbl)
            max_entropy = max(max_entropy, e)
            max_length  = max(max_length,  l)
    
    return max_entropy, max_length


def plot_diagonalized_matrix(matrix: np.ndarray, frame_error_rate: float, output_dir: str):
    """
    Reorder the matrix with Hungarian LSA, show unmatched -1 row/column as well.
    Force a square data region using aspect='equal'.
    """
    cost_mat = -matrix.T
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    diag_mat = matrix[np.ix_(col_ind, row_ind)]
    
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    
    cax = ax.imshow(diag_mat, cmap='viridis', aspect='equal')
    fig.colorbar(cax, ax=ax, label='Normalized Shared Area')
    ax.set_title(f'Diagonalized Mapping Matrix\nFER: {frame_error_rate:.2f}%', fontsize=14)
    ax.set_xlabel('Reordered Predicted Labels')
    ax.set_ylabel('Reordered Ground Truth Labels')
    ax.set_xticks(range(diag_mat.shape[1]))
    ax.set_yticks(range(diag_mat.shape[0]))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagonalized_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'diagonalized_matrix.svg'), format='svg', bbox_inches='tight')
    plt.close()


def plot_correlation_by_window(window_results, output_dir=None):
    """
    Plot correlation vs. window size.
    window_results = [(w, r_e, r_l, r_avg), ...]
    """
    ws = [x[0] for x in window_results]
    re = [x[1] for x in window_results]
    rl = [x[2] for x in window_results]
    ra = [x[3] for x in window_results]
    
    fig, ax = plt.subplots(figsize=(9,9))
    ax.plot(ws, re, 'o-', label='Entropy', color='#3498db', linewidth=2)
    ax.plot(ws, rl, 'o-', label='Length', color='#f39c12', linewidth=2)
    ax.plot(ws, ra, 'o--', label='Avg', color='#2ecc71', linewidth=2)
    ax.set_xlabel("Smoothing Window Size")
    ax.set_ylabel("Pearson Correlation (weighted)")
    ax.set_title("Pearson Correlation vs Window Size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Remove aspect='equal' and adjust y-axis limits
    ymin = min(min(re), min(rl), min(ra)) - 0.1
    ymax = max(max(re), max(rl), max(ra)) + 0.1
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_by_window.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'correlation_by_window.svg'), format='svg')
    else:
        plt.savefig('correlation_by_window.png', dpi=300)
        plt.savefig('correlation_by_window.svg', format='svg')
    plt.close()


def plot_correlation_comparisons(
    ground_truth_entropies, hdbscan_entropies,
    ground_truth_lengths,   hdbscan_lengths,
    bird_ids, fold_ids,
    phrase_counts,
    per_bird_phrase_fer: Dict[Tuple[str,int], float],
    # The dictionary: (bird, gt_label) -> mismatch%
    smoothing_window, max_entropy, max_length,
    output_dir=None
):
    """
    Scatter plots + per‐bird summary. 
    We'll read the FER from per_bird_phrase_fer if we find (bird, label).
    But note that we have phrase-level data only for matched labels. 
    If a label was unmatched, it wouldn't appear in the final arrays anyway.
    
    We'll scale marker size bigger (min_size=15).
    We'll force aspect='equal' for the main data region.
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
    
    # Convert lists -> arrays
    gte = np.array(ground_truth_entropies)
    hte = np.array(hdbscan_entropies)
    gtl = np.array(ground_truth_lengths)
    htl = np.array(hdbscan_lengths)
    pcounts = np.array(phrase_counts)
    birds   = np.array(bird_ids)
    folds   = np.array(fold_ids)
    
    # Build a "phrase-level FER" array if possible
    # We do (bird, label) => fer, but the label is not stored in phrase arrays. 
    # We only have the final label used for correlation. 
    # We'll just store "NaN" in the placeholder, or keep it if you want 
    # to do a different approach. We'll do a simpler approach: skip storing phrase-level fer.
    # We'll do a "by_bird" summary that doesn't rely on that array.
    # For the main correlation scatter, we don't color by fer.
    
    # Increase min dot size ~ 2-3x
    min_size = 15
    max_size = 150
    # scale by pcounts
    rng = pcounts.max() - pcounts.min() + 1e-9
    norm_counts = (pcounts - pcounts.min()) / rng
    sizes = min_size + norm_counts*(max_size - min_size)
    
    ############################################################################
    #  ENTROPY CORRELATION
    ############################################################################
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect('equal', adjustable='box')
    
    # 1) Diagonal line
    line_x = np.linspace(0, max_entropy, 200)
    ax.plot(line_x, line_x, 'k--', alpha=0.5, zorder=1)
    
    # 2) Scatter
    # color by bird
    unique_birds = np.unique(birds)
    palette = sns.color_palette("husl", n_colors=len(unique_birds))
    bird_color_map = dict(zip(unique_birds, palette))
    
    for i in range(len(gte)):
        b = birds[i]
        clr = bird_color_map[b]
        ax.scatter(gte[i], hte[i], s=sizes[i],
                   color=clr, alpha=0.7,
                   edgecolor='black', linewidth=0.5, zorder=2)
    
    # 3) Weighted Pearson
    r_pearson = calculate_weighted_pearson(gte, hte, pcounts)
    r_sq = r_pearson**2
    ax.text(0.05,0.95,
            f'Weighted Pearson r={r_pearson:.3f}\nr^2={r_sq:.3f}',
            transform=ax.transAxes,
            fontsize=FONT_SIZE,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=4))
    
    ax.set_xlabel("GT Phrase Entropy", fontsize=FONT_SIZE)
    ax.set_ylabel("HDBSCAN Phrase Entropy", fontsize=FONT_SIZE)
    ax.set_title(f"Phrase Entropy Comparison\n(Smoothing={smoothing_window})", fontsize=FONT_SIZE)
    ax.set_xlim(0, max_entropy)
    ax.set_ylim(0, max_entropy)
    
    from matplotlib.lines import Line2D
    legend_elems = []
    for b in unique_birds:
        legend_elems.append(Line2D([0],[0], marker='o', color='black',
                                   label=b, markersize=8,
                                   markerfacecolor=bird_color_map[b]))
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
    #  LENGTH CORRELATION
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
    ax.text(0.05,0.95,
            f'Weighted Pearson r={r_pearson_len:.3f}\nr^2={r_sq_len:.3f}',
            transform=ax.transAxes,
            fontsize=FONT_SIZE,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=4))
    
    ax.set_xlabel("GT Phrase Length", fontsize=FONT_SIZE)
    ax.set_ylabel("HDBSCAN Phrase Length", fontsize=FONT_SIZE)
    ax.set_title(f"Phrase Length Comparison\n(Smoothing={smoothing_window})", fontsize=FONT_SIZE)
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
    #   PER-BIRD SUMMARY (only synergy with the *phrase-level* arrays)
    ############################################################################
    print("\n================= PER-BIRD SUMMARY STATS =================\n")
    # We'll group phrase-level data by bird and do correlation stats. 
    # We'll also attempt to retrieve mismatch from per_bird_phrase_fer if we have the label. 
    # But we didn't store the label in these arrays. 
    # So we can only do correlation summary here, or reconstruct if needed.
    
    # We'll skip the "FER(%)" in that summary unless we have a direct link from each phrase to (bird,label).
    # For now, we can just do the correlation summary. 
    # If you'd prefer to show the bird-level average FER from your dictionary, do that too.
    
    by_bird = defaultdict(lambda: {
        "gt_entropy": [],
        "hd_entropy": [],
        "gt_length":  [],
        "hd_length":  [],
        "counts":     []
        # We won't store fer since we don't have label mapping here
    })
    
    for i in range(len(birds)):
        b = birds[i]
        by_bird[b]["gt_entropy"].append(gte[i])
        by_bird[b]["hd_entropy"].append(hte[i])
        by_bird[b]["gt_length"].append(gtl[i])
        by_bird[b]["hd_length"].append(htl[i])
        by_bird[b]["counts"].append(pcounts[i])
    
    for b in sorted(by_bird.keys()):
        gte_arr = np.array(by_bird[b]["gt_entropy"])
        hte_arr = np.array(by_bird[b]["hd_entropy"])
        gtl_arr = np.array(by_bird[b]["gt_length"])
        htl_arr = np.array(by_bird[b]["hd_length"])
        c_arr   = np.array(by_bird[b]["counts"])
        
        r_e = calculate_weighted_pearson(gte_arr, hte_arr, c_arr)
        r2_e= r_e**2
        r_l = calculate_weighted_pearson(gtl_arr, htl_arr, c_arr)
        r2_l= r_l**2
        avg_r2= (r2_e + r2_l)/2.0
        
        print(f"Bird: {b}")
        print(f"  Weighted r^2(Entropy) = {r2_e:.3f}")
        print(f"  Weighted r^2(Length)  = {r2_l:.3f}")
        print(f"  Average r^2           = {avg_r2:.3f}\n")


def plot_v_measure_by_window(window_results, output_dir=None):
    """
    Plot V-measure (or some other measure) vs. window. 
    window_results = [(window_size, v_score), ...]
    """
    ws = [x[0] for x in window_results]
    vs = [x[1] for x in window_results]
    
    fig, ax = plt.subplots(figsize=(9,9))
    ax.plot(ws, vs, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax.set_xlabel("Smoothing Window Size")
    ax.set_ylabel("V-measure Score")
    ax.set_title("V-measure Score vs Window Size")
    ax.grid(True, alpha=0.3)
    
    # Remove aspect='equal' and adjust y-axis limits
    ymin = min(vs) - 0.1 if vs else 0
    ymax = max(vs) + 0.1 if vs else 1.0
    ax.set_ylim(ymin, ymax)
    
    mean_v = np.nanmean(vs) if len(vs)>0 else 0.0
    ax.text(0.05, 0.95,
            f'Mean V-score: {mean_v:.3f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'v_measure_by_window.png'), dpi=300)
        plt.savefig(os.path.join(output_dir, 'v_measure_by_window.svg'), format='svg')
    else:
        plt.savefig('v_measure_by_window.png', dpi=300)
        plt.savefig('v_measure_by_window.svg', format='svg')
    plt.close()


##############################################################################
#                               MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    folder_path = "/media/george-vengrovski/George-SSD/folds_for_paper_llb"
    output_dir  = "results/proxy_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different window sizes
    window_dirs = {}
    
    # Smoothing windows to test
    window_sizes = list(range(0, 501, 25))  # 0 to 500 step size 25
    
    for wsize in window_sizes:
        window_dirs[wsize] = os.path.join(output_dir, f'window_{wsize}')
        os.makedirs(window_dirs[wsize], exist_ok=True)
    
    # Create best_window directory upfront
    best_window_dir = os.path.join(output_dir, 'best_window')
    os.makedirs(best_window_dir, exist_ok=True)
    
    # 1) Gather fold files
    all_npz_files = find_files_in_folder(folder_path)
    
    # 2) Global maxima
    max_entropy, max_length = calculate_max_values_for_folder(all_npz_files, labels_path=folder_path)
    
    all_results = []
    v_measure_results = []
    
    for wsize in tqdm(window_sizes, desc="Testing smoothing windows"):
        current_window_dir = window_dirs[wsize]
        
        (gt_e, hd_e, gt_l, hd_l, 
         birds, folds, counts,
         overall_fer, bird_phrase_fer) = process_all_folds(
             all_npz_files, wsize, labels_path=folder_path,
             output_dir=current_window_dir  # Save window-specific results
        )
        
        r_e = calculate_weighted_pearson(np.array(gt_e), np.array(hd_e), np.array(counts))
        r_l = calculate_weighted_pearson(np.array(gt_l), np.array(hd_l), np.array(counts))
        r_avg = (r_e + r_l)/2.0
        
        v_score = 0.0
        all_results.append((wsize, r_e, r_l, r_avg))
        v_measure_results.append((wsize, v_score))
        
        # Example: show diagonalized matrix for first file
        if len(all_npz_files)>0:
            first_file = all_npz_files[0]
            data = np.load(first_file)
            comp_cluster = ComputerClusterPerformance(labels_path=folder_path)
            gtlab = data['ground_truth_labels']
            pdlab = data['hdbscan_labels']
            gtp = comp_cluster.syllable_to_phrase_labels(gtlab, silence=0)
            pdlab_filled = comp_cluster.fill_noise_with_nearest_label(pdlab)
            pdlab_smooth = comp_cluster.majority_vote(pdlab_filled, wsize)
            
            analyzer = SequenceAnalyzer()
            M_norm, u_gt, u_pr = analyzer.create_shared_area_matrix(gtp, pdlab_smooth)
            plot_diagonalized_matrix(M_norm, overall_fer, current_window_dir)
        
        # Save correlation plots for each window
        plot_correlation_comparisons(
            gt_e, hd_e, gt_l, hd_l,
            birds, folds, counts,
            bird_phrase_fer,
            wsize, max_entropy, max_length,
            output_dir=current_window_dir
        )
    
    # Save overall results
    plot_correlation_by_window(all_results, output_dir=output_dir)
    plot_v_measure_by_window(v_measure_results, output_dir=output_dir)
    
    # Save summary to file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("========== SMOOTHING WINDOW RESULTS ==========\n")
        for i, (ws, re_, rl_, ra_) in enumerate(all_results):
            vv = v_measure_results[i][1]
            f.write(f"Window={ws}:  r_e={re_:.3f}, r_l={rl_:.3f}, avg={ra_:.3f}, v_score={vv:.3f}\n")
        f.write("==============================================\n\n")
        
        best_wsize = sorted(all_results, key=lambda x: x[3])[-1][0]
        f.write(f"BEST WINDOW SIZE = {best_wsize}\n\n")
        
        # Print final results for best window
        (gt_e, hd_e, gt_l, hd_l,
         birds, folds, counts,
         overall_fer, bird_phrase_fer) = process_all_folds(
             all_npz_files, best_wsize, labels_path=folder_path,
             output_dir=best_window_dir  # Using the created directory
        )
        
        f.write(f"Final Results (window={best_wsize}):\n")
        f.write(f"Overall FER: {overall_fer:.2f}%\n")
        f.write("Per-Bird, Per-Label FER:\n")
        f.write("-"*50 + "\n")
        
        all_birds_in_dict = sorted({k[0] for k in bird_phrase_fer.keys()})
        for b in all_birds_in_dict:
            f.write(f" Bird {b}:\n")
            labels_for_b = sorted([lbl for (bb,lbl) in bird_phrase_fer.keys() if bb==b])
            for lbl in labels_for_b:
                ferval = bird_phrase_fer[(b,lbl)]
                f.write(f"   GT Label {lbl}: {ferval:.2f}% mismatch\n")
            f.write("\n")

        # Also create all plots for best window
        plot_correlation_comparisons(
            gt_e, hd_e, gt_l, hd_l,
            birds, folds, counts,
            bird_phrase_fer,
            best_wsize, max_entropy, max_length,
            output_dir=best_window_dir
        )

    print("DONE! All results saved to results/proxy_metrics/")
