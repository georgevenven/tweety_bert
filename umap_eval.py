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

# Make sure these exist in your local code:
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
      M[i,j] = number of frames in which ground_truth == uniqueGT[i]
                                    and predicted    == uniquePred[j].
    Then we column‐normalize M so each column sums to 1 (where possible).
    """
    @staticmethod
    def create_shared_area_matrix(ground_truth: np.ndarray,
                                  predicted:   np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_gt  = np.unique(ground_truth)
        unique_pred = np.unique(predicted)
        
        M = np.zeros((len(unique_gt), len(unique_pred)))
        for i, g in enumerate(unique_gt):
            for j, p in enumerate(unique_pred):
                M[i, j] = np.sum((ground_truth == g) & (predicted == p))
        
        # Normalize columns so sum=1
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
    Given the normalized matrix of shape (len(unique_gt), len(unique_pred)),
    use Hungarian algorithm to find a 'diagonal' assignment that maximizes overlap.
    
    We treat the matrix as a cost matrix = -normalized_matrix (because
    linear_sum_assignment does a minimum assignment).
    
    Then row_ind[k], col_ind[k] says that ground_truth label at row_ind[k]
    is matched to predicted label at col_ind[k].
    
    Because row_ind, col_ind are each length = min(#gt, #pred),
    any leftover GT labels get mapped to -1.
    """
    # The shape is (#gt, #pred). We want to find a pairing that
    # maximizes the sum of matrix entries => minimize negative
    cost_matrix = -normalized_matrix
    
    # Solve
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build dictionary: ground_truth_label -> predicted_label
    mapping = {}
    
    # For the matched pairs
    for i in range(len(row_ind)):
        gt_idx  = row_ind[i]
        pred_idx = col_ind[i]
        gt_label  = unique_gt_labels[gt_idx]
        pred_label = unique_pred_labels[pred_idx]
        mapping[gt_label] = pred_label
    
    # For any ground_truth label not in row_ind => map to -1
    matched_gt_set = set(unique_gt_labels[row_ind])
    for label in unique_gt_labels:
        if label not in matched_gt_set:
            mapping[label] = -1
    
    return mapping


def calculate_average_phrase_entropy(labels, target_label):
    """
    For all contiguous runs of `target_label`,
    see what label we transition to next.
    Then compute the Shannon entropy of that distribution.
    """
    transition_counts = Counter()
    total_transitions = 0
    in_target_phrase  = False
    
    for i in range(len(labels) - 1):
        curr_label = labels[i]
        next_label = labels[i+1]
        if curr_label == target_label:
            in_target_phrase = True
        else:
            if in_target_phrase:
                # We ended a run of target_label, so count the next_label
                transition_counts[next_label] += 1
                total_transitions += 1
                in_target_phrase = False
    
    # Edge case: if the array ends in target_label, there's no next_label
    # so no transition there.
    
    if total_transitions == 0:
        return 0.0
    
    entropy = 0.0
    for count in transition_counts.values():
        probability = count / total_transitions
        entropy -= probability * np.log(probability)
    return entropy


def calculate_average_phrase_length(labels, target_label):
    """
    Average contiguous-run length for `target_label`.
    """
    lengths = []
    curr_len = 0
    
    for l in labels:
        if l == target_label:
            curr_len += 1
        else:
            if curr_len > 0:
                lengths.append(curr_len)
                curr_len = 0
    # If ended with a run
    if curr_len > 0:
        lengths.append(curr_len)
    
    return np.mean(lengths) if len(lengths) > 0 else 0.0


def calculate_weighted_pearson(y_true, y_pred, weights):
    """
    Weighted Pearson correlation, weighting each sample by `weights`.
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    weights = np.array(weights)
    
    if np.all(weights == 0) or len(y_true)==0:
        return 0.0
    
    mean_y_true = np.average(y_true, weights=weights)
    mean_y_pred = np.average(y_pred, weights=weights)
    
    cov = np.average((y_true - mean_y_true)*(y_pred - mean_y_pred), weights=weights)
    var_y_true = np.average((y_true - mean_y_true)**2, weights=weights)
    var_y_pred = np.average((y_pred - mean_y_pred)**2, weights=weights)
    
    denom = np.sqrt(var_y_true * var_y_pred)
    if denom == 0:
        return 0.0
    return cov / denom


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
def process_all_folds(files: List[str], smoothing_window: int, labels_path: str):
    """
    For each fold/file in `files`:
      1) Convert GT labels -> phrase
      2) Fill + smooth predicted, BUT only within each separate "song" 
         according to dataset_indices (so no cross-song bleeding).
      3) Build normalized matrix
      4) Diagonalize to find GT->pred mapping
      5) Map ground_truth phrases => predicted label => compute FER, phrase metrics
      6) Return arrays for correlation scatter plots.
      
    Returns:
        (gt_entropies, hdbscan_entropies,
         gt_lengths,    hdbscan_lengths,
         bird_ids, fold_ids, phrase_counts, frame_error_rates,
         v_measure_scores)
    """
    ground_truth_entropies = []
    hdbscan_entropies      = []
    ground_truth_lengths   = []
    hdbscan_lengths        = []
    bird_ids               = []
    fold_ids               = []
    phrase_counts          = []
    frame_error_rates      = []
    v_measure_scores       = []
    
    analyzer = SequenceAnalyzer()
    
    # ----------------------------------------
    # HELPER: smooth labels per unique "song"
    # ----------------------------------------
    def smooth_per_song(labels: np.ndarray,
                        dataset_indices: np.ndarray,
                        window_size: int,
                        computer_cluster: ComputerClusterPerformance):
        """
        Applies majority_vote to 'labels' for each unique song in dataset_indices
        so that smoothing is *only* within that song, never across boundaries.
        
        Returns a new np.ndarray of the same shape as 'labels'.
        """
        if window_size == 1:
            # No smoothing, just return as-is
            return labels.copy()
        
        # Prepare an output array
        smoothed = np.zeros_like(labels)
        unique_song_ids = np.unique(dataset_indices)
        
        for song_id in unique_song_ids:
            # Gather the frames belonging to this song_id
            mask = (dataset_indices == song_id)
            
            # Extract that subset
            sub_arr = labels[mask]
            
            # Apply majority_vote on that subset - FIXED HERE
            # Only pass sub_arr as data, window_size as kwarg
            try:
                sub_smoothed = computer_cluster.majority_vote(sub_arr, window_size=window_size)
                smoothed[mask] = sub_smoothed
            except ValueError as e:
                print(f"Warning: majority_vote failed for song_id {song_id}: {str(e)}")
                smoothed[mask] = sub_arr  # Use unsmoothed data on failure
        
        return smoothed
    
    for fpath in tqdm(files, desc=f"Processing folds (window={smoothing_window})"):
        fname = os.path.basename(fpath)
        bird_id, fold_id = get_bird_id_and_fold(fname)
        
        data = np.load(fpath)
        ground_truth_labels = data['ground_truth_labels']
        hdbscan_labels      = data['hdbscan_labels']
        
        # This array has same shape as hdbscan/GT
        # Tells which "song" each frame belongs to
        dataset_indices     = data['dataset_indices']  # shape=(N,)
        
        # 1) Instantiate your local analysis class
        computer_cluster = ComputerClusterPerformance(labels_path=labels_path)
        
        # 2) Convert ground-truth => phrase
        gt_phrase = computer_cluster.syllable_to_phrase_labels(
            ground_truth_labels, 
            silence=0
        )
        
        # 3a) Fill noise in predicted (so -1 frames get replaced)
        pred_filled = computer_cluster.fill_noise_with_nearest_label(hdbscan_labels)
        
        # 3b) Now do "per-song" smoothing, 
        #     ensuring we never blur across dataset_indices boundaries.
        try:
            pred_smooth = smooth_per_song(
                labels=pred_filled,
                dataset_indices=dataset_indices,
                window_size=smoothing_window,
                computer_cluster=computer_cluster
            )
        except ValueError:
            # If the user code has a known bug with majority_vote, skip
            continue
        
        # If everything is -1 or no frames left, skip
        if np.all(pred_smooth == -1):
            continue
        
        # 4) Build normalized matrix from these final arrays
        M_norm, uniq_gt, uniq_pred = analyzer.create_shared_area_matrix(
            gt_phrase, 
            pred_smooth
        )
        
        # If either dimension is empty, skip
        if len(uniq_gt) == 0 or len(uniq_pred) == 0:
            continue
        
        # 5) Diagonalize => create a ground_truth->pred label mapping
        mapping = create_diagonal_label_mapping(M_norm, uniq_gt, uniq_pred)
        
        # 6) Map each ground_truth phrase label to its assigned predicted label
        mapped_gt = np.array([mapping[g] for g in gt_phrase])
        
        # 7) Compute FER:
        #    We'll ignore frames where mapped_gt == -1 or pred_smooth == -1
        valid_mask = (mapped_gt != -1) & (pred_smooth != -1)
        if np.sum(valid_mask) == 0:
            fer = np.nan
        else:
            mismatches = np.sum(mapped_gt[valid_mask] != pred_smooth[valid_mask])
            fer = (mismatches / np.sum(valid_mask)) * 100.0
        
        # 8) For phrase metrics: we look at each assigned predicted label
        uniq_mapped = np.unique(mapped_gt)
        for label in uniq_mapped:
            if label == -1:
                continue
            
            # average phrase entropy in mapped-gt
            gt_entropy = calculate_average_phrase_entropy(mapped_gt, label)
            # average phrase entropy in predicted
            hd_entropy = calculate_average_phrase_entropy(pred_smooth, label)
            
            # average phrase length in mapped-gt
            gt_length  = calculate_average_phrase_length(mapped_gt, label)
            # average phrase length in predicted
            hd_length  = calculate_average_phrase_length(pred_smooth, label)
            
            # how many frames in mapped_gt => label
            pcount = np.sum(mapped_gt == label)
            
            ground_truth_entropies.append(gt_entropy)
            hdbscan_entropies.append(hd_entropy)
            ground_truth_lengths.append(gt_length)
            hdbscan_lengths.append(hd_length)
            phrase_counts.append(pcount)
            bird_ids.append(bird_id)
            fold_ids.append(fold_id)
            frame_error_rates.append(fer)
        
        # After computing pred_smooth, add V-measure calculation:
        try:
            # Compute V-measure only on valid frames (where neither is -1)
            valid_mask = (gt_phrase != -1) & (pred_smooth != -1)
            if np.sum(valid_mask) > 0:
                v_score = v_measure_score(
                    gt_phrase[valid_mask],
                    pred_smooth[valid_mask]
                )
            else:
                v_score = 0.0
            v_measure_scores.append(v_score)
        except ValueError:
            v_measure_scores.append(0.0)
    
    return (ground_truth_entropies, hdbscan_entropies,
            ground_truth_lengths, hdbscan_lengths,
            bird_ids, fold_ids, phrase_counts, frame_error_rates,
            v_measure_scores)



def calculate_max_values_for_folder(files: List[str], labels_path: str) -> Tuple[float,float]:
    """
    Find max possible phrase entropy and phrase length in the GT across all folds
    for consistent axis ranges in your correlation plots.
    """
    max_entropy = 0.0
    max_length  = 0.0
    
    # We'll use your ComputerClusterPerformance
    # (if it needs a labels_path, provide it; else skip)
    computer_cluster = ComputerClusterPerformance(labels_path=labels_path)
    
    for fpath in tqdm(files, desc="Calculating max values"):
        data = np.load(fpath)
        gt_labels = data['ground_truth_labels']
        
        # Convert to phrase
        gt_phrase = computer_cluster.syllable_to_phrase_labels(gt_labels, silence=0)
        
        for label in np.unique(gt_phrase):
            # skip silence if that is your convention
            if label == 0:
                continue
            
            entropy = calculate_average_phrase_entropy(gt_phrase, label)
            length  = calculate_average_phrase_length(gt_phrase, label)
            max_entropy = max(max_entropy, entropy)
            max_length  = max(max_length,  length)
    
    return max_entropy, max_length


def plot_diagonalized_matrix(matrix: np.ndarray, frame_error_rate: float, output_prefix=""):
    """
    For display: reorder the matrix with Hungarian LSA 
    so it looks more diagonal if there's a strong mapping.
    """
    cost_mat = -matrix.T  # shape (len(pred), len(gt)) if matrix is (len(gt), len(pred))
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    # reorder
    diag_mat = matrix[np.ix_(col_ind, row_ind)]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(diag_mat, cmap='viridis', aspect='auto')
    plt.colorbar(label='Normalized Shared Area')
    plt.title(f'Diagonalized Mapping Matrix\nFER: {frame_error_rate:.2f}%')
    plt.xlabel('Reordered Predicted Labels')
    plt.ylabel('Reordered Ground Truth Labels')
    plt.tight_layout()
    
    plt.savefig(f'{output_prefix}diagonalized_matrix.png', dpi=300)
    plt.savefig(f'{output_prefix}diagonalized_matrix.svg', format='svg')
    plt.close()


def plot_correlation_by_window(window_results):
    """
    Plot correlation vs. window size.  window_results = [(w, r_e, r_l, r_avg), ...]
    """
    ws = [x[0] for x in window_results]
    re=  [x[1] for x in window_results]
    rl=  [x[2] for x in window_results]
    ra=  [x[3] for x in window_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ws, re, 'o-', label='Entropy', color='#3498db')
    plt.plot(ws, rl, 'o-', label='Length', color='#f39c12')
    plt.plot(ws, ra, 'o--', label='Avg', color='#2ecc71')
    
    plt.xlabel("Smoothing Window Size")
    plt.ylabel("Pearson Correlation (weighted)")
    plt.title("Pearson Correlation vs Window Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('correlation_by_window.png', dpi=300)
    plt.savefig('correlation_by_window.svg', format='svg')
    plt.close()


def plot_correlation_comparisons(
    ground_truth_entropies, hdbscan_entropies,
    ground_truth_lengths,   hdbscan_lengths,
    bird_ids, fold_ids,
    phrase_counts, frame_error_rates,
    smoothing_window, max_entropy, max_length
):
    """
    Scatter plots + per‐bird stats. Color by bird. Each fold is its own point.
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
    fers    = np.array(frame_error_rates)
    
    unique_birds = np.unique(birds)
    palette = sns.color_palette("husl", n_colors=len(unique_birds))
    bird_color_map = dict(zip(unique_birds, palette))
    
    # For scatter size, mild log transform
    min_size = 40
    sizes = [max(min_size, np.log(x+1)*50) for x in pcounts]
    
    # ---------------------------
    #  ENTROPY CORRELATION
    # ---------------------------
    plt.figure(figsize=(10, 8))
    line_x = np.linspace(0, max_entropy, 100)
    plt.plot(line_x, line_x, 'k--', alpha=0.5, zorder=1)
    
    for i in range(len(gte)):
        clr = bird_color_map[birds[i]]
        plt.scatter(gte[i], hte[i], 
                    color=clr, s=sizes[i],
                    alpha=0.8, edgecolor='black',
                    linewidth=0.5, zorder=2)
    
    r_pearson = calculate_weighted_pearson(gte, hte, pcounts)
    r_sq = r_pearson**2
    
    plt.text(0.05, 0.95,
             f'Weighted Pearson r={r_pearson:.3f}\nr^2={r_sq:.3f}',
             transform=plt.gca().transAxes,
             fontsize=FONT_SIZE,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=4))
    
    plt.xlabel("GT Phrase Entropy", fontsize=FONT_SIZE)
    plt.ylabel("HDBSCAN Phrase Entropy", fontsize=FONT_SIZE)
    plt.title(f"Phrase Entropy Comparison\n(Smoothing={smoothing_window})")
    plt.xlim(0, max_entropy)
    plt.ylim(0, max_entropy)
    
    from matplotlib.lines import Line2D
    legend_elems = []
    for b in unique_birds:
        legend_elems.append(Line2D([0],[0], marker='o', color='black',
                                   label=b, markersize=10,
                                   markerfacecolor=bird_color_map[b]))
    plt.legend(handles=legend_elems, title="Bird ID", fontsize=FONT_SIZE-2)
    plt.tight_layout()
    plt.savefig(f'phrase_entropy_correlation_{smoothing_window}.png', dpi=300)
    plt.savefig(f'phrase_entropy_correlation_{smoothing_window}.svg', format='svg')
    plt.close()
    
    # ---------------------------
    #  LENGTH CORRELATION
    # ---------------------------
    plt.figure(figsize=(10, 8))
    line_x = np.linspace(0, max_length, 100)
    plt.plot(line_x, line_x, 'k--', alpha=0.5, zorder=1)
    
    for i in range(len(gtl)):
        clr = bird_color_map[birds[i]]
        plt.scatter(gtl[i], htl[i],
                    color=clr, s=sizes[i],
                    alpha=0.8, edgecolor='black',
                    linewidth=0.5, zorder=2)
    
    r_pearson = calculate_weighted_pearson(gtl, htl, pcounts)
    r_sq = r_pearson**2
    
    plt.text(0.05, 0.95,
             f'Weighted Pearson r={r_pearson:.3f}\nr^2={r_sq:.3f}',
             transform=plt.gca().transAxes,
             fontsize=FONT_SIZE,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=4))
    
    plt.xlabel("GT Phrase Length", fontsize=FONT_SIZE)
    plt.ylabel("HDBSCAN Phrase Length", fontsize=FONT_SIZE)
    plt.title(f"Phrase Length Comparison\n(Smoothing={smoothing_window})")
    plt.xlim(0, max_length)
    plt.ylim(0, max_length)
    
    plt.legend(handles=legend_elems, title="Bird ID", fontsize=FONT_SIZE-2)
    plt.tight_layout()
    plt.savefig(f'phrase_length_correlation_{smoothing_window}.png', dpi=300)
    plt.savefig(f'phrase_length_correlation_{smoothing_window}.svg', format='svg')
    plt.close()
    
    # ---------------------------
    #   PER-BIRD SUMMARY
    # ---------------------------
    print("\n================= PER-BIRD SUMMARY STATS =================\n")
    # We'll gather arrays by bird
    by_bird = defaultdict(lambda: {
        "gt_entropy": [],
        "hd_entropy": [],
        "gt_length":  [],
        "hd_length":  [],
        "counts":     [],
        "fer":        []
    })
    
    for i in range(len(birds)):
        b = birds[i]
        by_bird[b]["gt_entropy"].append(gte[i])
        by_bird[b]["hd_entropy"].append(hte[i])
        by_bird[b]["gt_length"].append(gtl[i])
        by_bird[b]["hd_length"].append(htl[i])
        by_bird[b]["counts"].append(pcounts[i])
        by_bird[b]["fer"].append(fers[i])
    
    for b in by_bird:
        gte_arr  = np.array(by_bird[b]["gt_entropy"])
        hte_arr  = np.array(by_bird[b]["hd_entropy"])
        gtl_arr  = np.array(by_bird[b]["gt_length"])
        htl_arr  = np.array(by_bird[b]["hd_length"])
        c_arr    = np.array(by_bird[b]["counts"])
        fer_arr  = np.array(by_bird[b]["fer"])
        
        r_e  = calculate_weighted_pearson(gte_arr, hte_arr, c_arr)
        r2_e = r_e**2
        r_l  = calculate_weighted_pearson(gtl_arr, htl_arr, c_arr)
        r2_l = r_l**2
        avg_r2 = (r2_e + r2_l)/2.0
        
        mean_fer = np.nanmean(fer_arr)
        std_fer  = np.nanstd(fer_arr)
        
        print(f"Bird: {b}")
        print(f"  Weighted r^2(Entropy) = {r2_e:.3f}")
        print(f"  Weighted r^2(Length)  = {r2_l:.3f}")
        print(f"  Average r^2           = {avg_r2:.3f}")
        print(f"  FER(%) = {mean_fer:.2f} ± {std_fer:.2f} (SD)\n")


def plot_v_measure_by_window(window_results):
    """
    Plot V-measure score vs. window size.
    window_results = [(window_size, v_score), ...]
    """
    ws = [x[0] for x in window_results]
    vs = [x[1] for x in window_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ws, vs, 'o-', color='#e74c3c', linewidth=2)
    
    plt.xlabel("Smoothing Window Size")
    plt.ylabel("V-measure Score")
    plt.title("V-measure Score vs Window Size")
    plt.grid(True, alpha=0.3)
    
    # Add mean v-score as text
    mean_v = np.mean(vs)
    plt.text(0.05, 0.95,
             f'Mean V-score: {mean_v:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('v_measure_by_window.png', dpi=300)
    plt.savefig('v_measure_by_window.svg', format='svg')
    plt.close()


##############################################################################
#                               MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    # Change path as needed
    folder_path = "/media/george-vengrovski/George-SSD/folds_for_paper_llb"
    
    # Range of smoothing windows to test
    window_start = 0
    window_end   = 100
    window_step  = 25
    window_sizes = list(range(window_start, window_end+window_step, window_step))
    
    # 1) Gather all fold files
    all_npz_files = find_files_in_folder(folder_path)
    
    # 2) Global maxima for plotting
    max_entropy, max_phrase_len = calculate_max_values_for_folder(all_npz_files, labels_path=folder_path)
    
    # 3) Evaluate multiple smoothing windows
    all_results = []
    v_measure_results = []  # New list for V-measure results
    
    for wsize in tqdm(window_sizes, desc="Testing smoothing windows"):
        (gt_e, hd_e, gt_l, hd_l, 
         birds, folds, counts, fers,
         v_scores) = process_all_folds(
            all_npz_files, wsize, labels_path=folder_path
        )
        
        # Weighted correlation across all points
        r_e = calculate_weighted_pearson(np.array(gt_e), np.array(hd_e), np.array(counts))
        r_l = calculate_weighted_pearson(np.array(gt_l), np.array(hd_l), np.array(counts))
        r_avg = (r_e + r_l)/2.0
        
        all_results.append((wsize, r_e, r_l, r_avg))
        
        # Add mean V-measure score for this window
        mean_v = np.mean(v_scores)
        v_measure_results.append((wsize, mean_v))
        
        # Optionally show a diagonalized matrix for the first window, using the first file
        if wsize == window_start and len(all_npz_files)>0:
            # We'll do a quick example
            first_file = all_npz_files[0]
            data = np.load(first_file)
            gt_labels = data['ground_truth_labels']
            pred_labels = data['hdbscan_labels']
            
            # Build the post‐processed arrays
            comp_cluster = ComputerClusterPerformance(labels_path=folder_path)
            gt_phrase   = comp_cluster.syllable_to_phrase_labels(gt_labels, silence=0)
            pred_filled = comp_cluster.fill_noise_with_nearest_label(pred_labels)
            pred_smooth = comp_cluster.majority_vote(pred_filled, window_size=wsize)
            
            # Build matrix
            analyzer = SequenceAnalyzer()
            M_norm, _, _ = analyzer.create_shared_area_matrix(gt_phrase, pred_smooth)
            
            # Suppose we just use an average FER = mean of all folds
            mean_fer = np.nanmean(fers) if len(fers)>0 else 0.0
            
            plot_diagonalized_matrix(M_norm, mean_fer, output_prefix=f"window{wsize}_")
    
    # 4) Plot correlation vs window
    plot_correlation_by_window(all_results)
    
    # 5) Plot V-measure vs window
    plot_v_measure_by_window(v_measure_results)
    
    # 6) Print summary including V-measure scores
    print("\n========== SMOOTHING WINDOW RESULTS ==========")
    for i, (ws, re_, rl_, ra_) in enumerate(all_results):
        v_score = v_measure_results[i][1]
        print(f"Window={ws}:  r_e={re_:.3f},  r_l={rl_:.3f},  avg={ra_:.3f},  v_score={v_score:.3f}")
    print("==============================================\n")
    
    # 7) Pick best window by average correlation
    best_wsize = sorted(all_results, key=lambda x: x[3])[-1][0]
    print(f"BEST WINDOW SIZE = {best_wsize}\n")
    
    # 8) Final pass + correlation plots
    (gt_e, hd_e, gt_l, hd_l, birds, folds, counts, fers, v_scores) = process_all_folds(
        all_npz_files, best_wsize, labels_path=folder_path
    )
    
    plot_correlation_comparisons(
        gt_e, hd_e, 
        gt_l, hd_l,
        birds, folds,
        counts, fers,
        best_wsize,
        max_entropy,
        max_phrase_len
    )
    
    print("DONE! Check your plots and printed summaries.")
