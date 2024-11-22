import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from itertools import combinations
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

def load_embedding_data(npz_file_path):
    """Load embedding data and metadata from NPZ file."""
    data = np.load(npz_file_path, allow_pickle=True)
    
    # Extract metadata from filename
    file_base = os.path.basename(npz_file_path)
    file_parts = file_base.split('_')
    bird_id = file_parts[0]  # e.g., "USA5508"
    
    # Load the data
    embedding_outputs = data['embedding_outputs']
    dataset_indices = data['dataset_indices']
    
    # Try to load file indices and map if they exist
    file_indices = data.get('file_indices', None)
    file_map = data.get('file_map', None)
    
    return {
        'embedding_outputs': embedding_outputs,
        'dataset_indices': dataset_indices,
        'bird_id': bird_id,
        'file_path': npz_file_path,
        'file_indices': file_indices,
        'file_map': file_map
    }


def calculate_distribution_similarity(heatmap1, heatmap2):
   """Calculate Bhattacharyya coefficient between two distributions."""
   eps = 1e-10
   p = heatmap1 / (np.sum(heatmap1) + eps)
   q = heatmap2 / (np.sum(heatmap2) + eps)
   
   similarity = np.sum(np.sqrt(p * q))
   
   return similarity

def calculate_statistical_significance(similarity_matrix):
    """Calculate statistical significance using exact permutation test."""
    # Extract within and between period similarities
    within_periods = np.array([
        similarity_matrix[0,1],  # Before1 vs Before2
        similarity_matrix[2,3]   # After1 vs After2
    ])
    
    between_periods = np.array([
        similarity_matrix[0,2], similarity_matrix[0,3],
        similarity_matrix[1,2], similarity_matrix[1,3]
    ])
    
    # Observed difference in means
    observed_diff = np.mean(within_periods) - np.mean(between_periods)
    
    # Get all possible combinations
    all_similarities = np.concatenate([within_periods, between_periods])
    all_possible_within = list(combinations(all_similarities, 2))
    
    # Calculate difference for each possible combination
    perm_diffs = []
    for within_combo in all_possible_within:
        within = np.array(within_combo)
        between = np.array([x for x in all_similarities if x not in within])
        perm_diffs.append(np.mean(within) - np.mean(between))
    
    # Calculate exact p-value
    p_value = np.mean(np.array(perm_diffs) >= observed_diff)
    
    return {
        'within_mean': np.mean(within_periods),
        'within_std': np.std(within_periods),
        'between_mean': np.mean(between_periods),
        'between_std': np.std(between_periods),
        'p_value': p_value
    }

def plot_similarity_comparison(similarity_matrix, stats_results, save_dir, bird_id):
    """Create summary visualization of similarity structure."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Matrix with block structure
    sns.heatmap(similarity_matrix, 
                annot=True,  # Show numbers in cells
                fmt='.3f',   # Format numbers to 3 decimal places
                xticklabels=['Before1', 'Before2', 'After1', 'After2'],
                yticklabels=['Before1', 'Before2', 'After1', 'After2'],
                ax=ax1, 
                cmap='RdPu',
                vmin=0,      # Force scale to start at 0
                vmax=1)      # Force scale to end at 1
    ax1.set_title('Similarity Matrix')
    
    # Add block labels for visual grouping
    ax1.add_patch(plt.Rectangle((0, 0), 2, 2, fill=False, color='blue', linewidth=2))  # Before block
    ax1.add_patch(plt.Rectangle((2, 2), 2, 2, fill=False, color='blue', linewidth=2))  # After block
    ax1.add_patch(plt.Rectangle((0, 2), 2, 2, fill=False, color='red', linewidth=2))   # Between block
    ax1.add_patch(plt.Rectangle((2, 0), 2, 2, fill=False, color='red', linewidth=2))   # Between block
    
    # Plot 2: Bar plot comparison
    means = [stats_results['within_mean'], stats_results['between_mean']]
    stds = [stats_results['within_std'], stats_results['between_std']]
    
    bars = ax2.bar(['Within Periods', 'Between Periods'], means, yerr=stds,
                   capsize=5, color=['blue', 'red'])
    ax2.set_ylabel('Similarity (Cosine Similarity)')
    ax2.set_title(f'Similarity Comparison\np = {stats_results["p_value"]:.3e}')
    
    # Add exact values on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.suptitle(f'Bird {bird_id} - Recording Group Similarities', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'bird_{bird_id}_similarity_analysis.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, f'bird_{bird_id}_similarity_analysis.svg'))
    plt.close()

def plot_pairwise_comparisons(embedding_outputs, dataset_indices, save_dir, bird_id, file_indices, file_map):
    """Plot and calculate similarities between all recording pairs."""
    # Generate heatmaps
    bins = 300
    heatmaps = []
    labels = ['Before1', 'Before2', 'After1', 'After2']
    
    # Create first heatmap to get consistent binning
    mask = dataset_indices == 0
    data = embedding_outputs[mask]
    heatmap_1, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    heatmap_1 = heatmap_1 / heatmap_1.max()
    heatmaps.append(heatmap_1)
    
    # Create other heatmaps with same binning
    for i in range(1, 4):
        mask = dataset_indices == i
        data = embedding_outputs[mask]
        heatmap, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=[xedges, yedges])
        heatmap = heatmap / heatmap.max()
        heatmaps.append(heatmap)
    
    # Calculate similarity matrix
    n_maps = len(heatmaps)
    similarities = np.zeros((n_maps, n_maps))
    for i in range(n_maps):
        for j in range(n_maps):
            similarities[i,j] = calculate_distribution_similarity(heatmaps[i], heatmaps[j])
    
    # Calculate statistical significance
    stats_results = calculate_statistical_significance(similarities)
    
    # Plot similarity matrix with statistical analysis
    plot_similarity_comparison(similarities, stats_results, save_dir, bird_id)
    
    # Generate separate and combined UMAP plots
    plot_separate_and_combined_umaps(
        embedding_outputs,
        dataset_indices,
        save_dir,
        bird_id,
        file_indices,
        file_map,
        labels=['Before1', 'Before2', 'After1', 'After2'],
        before_indices=[0, 1],
        after_indices=[2, 3]
    )
    
    # Calculate summary statistics
    within_before = similarities[0,1]
    within_after = similarities[2,3]
    between_periods = np.mean([
        similarities[0,2], similarities[0,3],
        similarities[1,2], similarities[1,3]
    ])
    
    print(f"\nBird {bird_id} Similarity Summary:")
    print(f"Within Before Period: {within_before:.3f}")
    print(f"Within After Period: {within_after:.3f}")
    print(f"Between Periods: {between_periods:.3f}")
    print(f"Statistical Test: p = {stats_results['p_value']:.3e}")
    
    return {
        'within_before': within_before,
        'within_after': within_after,
        'between_periods': between_periods,
        'similarity_matrix': similarities,
        'p_value': stats_results['p_value'],
        'within_mean': stats_results['within_mean'],
        'within_std': stats_results['within_std'],
        'between_mean': stats_results['between_mean'],
        'between_std': stats_results['between_std']
    }

def analyze_multiple_birds(npz_files, save_dir):
    """Analyze multiple birds from separate NPZ files."""
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    for npz_file in npz_files:
        # Load data
        data = load_embedding_data(npz_file)
        embedding_outputs = data['embedding_outputs']
        dataset_indices = data['dataset_indices']
        bird_id = data['bird_id']
        file_indices = data['file_indices']
        file_map = data['file_map']
        
        print(f"\nProcessing bird {bird_id} from {npz_file}")
        
        # Calculate similarities and plot
        result = plot_pairwise_comparisons(embedding_outputs, dataset_indices, save_dir, bird_id, file_indices, file_map)
        result['bird_id'] = bird_id
        result['file_path'] = npz_file
        result['file_indices'] = data['file_indices']
        result['file_map'] = data['file_map']
        result['dataset_indices'] = dataset_indices
        results.append(result)
        
        # Add the combined heatmap plot
        plot_all_heatmaps_combined(embedding_outputs, dataset_indices, save_dir, bird_id, file_indices, file_map)
    
    # Compile results
    if results:
        # Create DataFrame and plot similarity summary
        results_df = pd.DataFrame({
            'Bird': [r['bird_id'] for r in results],
            'File_Path': [r['file_path'] for r in results],
            'Within_Before': [r['within_before'] for r in results],
            'Within_After': [r['within_after'] for r in results],
            'Between_Periods': [r['between_periods'] for r in results],
            'P_value': [r['p_value'] for r in results],
            'Within_Mean': [r['within_mean'] for r in results],
            'Within_Std': [r['within_std'] for r in results],
            'Between_Mean': [r['between_mean'] for r in results],
            'Between_Std': [r['between_std'] for r in results],
            'file_indices': [r['file_indices'] for r in results],
            'file_map': [r['file_map'] for r in results],
            'dataset_indices': [r['dataset_indices'] for r in results]
        })
        
        # Print detailed file information
        print("\nDetailed File Information:")
        for idx, row in results_df.iterrows():
            print(f"\nBird: {row['Bird']}")
            print(f"File: {os.path.basename(row['File_Path'])}")
            print(f"Within Before similarity: {row['Within_Before']:.3f}")
            print(f"Within After similarity: {row['Within_After']:.3f}")
            print(f"Between Periods similarity: {row['Between_Periods']:.3f}")
            print(f"P-value: {row['P_value']:.3e}")
        
        plot_similarity_summary(results_df, save_dir)
        results_df.to_csv(os.path.join(save_dir, 'all_birds_results.csv'))
        return results_df
    else:
        print("No valid results to report")
        return None

def plot_separate_and_combined_umaps(embedding_outputs, dataset_indices, save_dir, bird_id, 
                                     file_indices, file_map,
                                     labels=['Before1', 'Before2', 'After1', 'After2'],
                                     before_indices=[0, 1], after_indices=[2, 3]):
    """
    Plot individual UMAPs and combined before/after comparison.
    """
    bins = 300
    
    # Create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get consistent binning using all data
    all_data = embedding_outputs
    _, xedges, yedges = np.histogram2d(all_data[:, 0], all_data[:, 1], bins=bins)
    
    # Prepare file map dictionary
    if file_map is not None:
        file_map_dict = file_map.item()
    else:
        file_map_dict = {}
    
    # Plot individual heatmaps
    individual_heatmaps = []
    for i, label in enumerate(labels):
        mask = dataset_indices == i
        current_data = embedding_outputs[mask]
        
        # Create heatmap
        heatmap, _, _ = np.histogram2d(current_data[:, 0], current_data[:, 1], 
                                       bins=[xedges, yedges])
        heatmap = heatmap / heatmap.max()
        individual_heatmaps.append(heatmap)
        
        # Get dates and number of points
        group_file_indices = file_indices[mask]
        dates = []
        unique_file_indices = np.unique(group_file_indices)
        for idx in unique_file_indices:
            idx = int(idx)
            if idx in file_map_dict:
                file_path = file_map_dict[idx][0]
                date_time, _ = parse_date_time(file_path)
                if date_time:
                    dates.append(date_time)
        
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            date_str = min_date.strftime('%Y-%m-%d')
            num_points = len(current_data)
            ax_title = f"{label}\n{date_range_str}\nPoints: {num_points}"
            filename = f"{bird_id}_{label}_{date_str}_umap"
        else:
            num_points = len(current_data)
            ax_title = f"{label}\nPoints: {num_points}"
            filename = f"{bird_id}_{label}_umap"
        
        # Plot individual map with appropriate colormap
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = 'Greens' if i in after_indices else 'Purples'  # Use green for after periods
        im = ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                       origin='lower', cmap=cmap, vmax=0.1, aspect='equal')
        ax.set_title(ax_title, fontsize=16)
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300)
        fig.savefig(os.path.join(save_dir, f'{filename}.svg'))
        plt.close(fig)
    
    # Create combined before
    before_mask = np.isin(dataset_indices, before_indices)
    before_data = embedding_outputs[before_mask]
    
    before_file_indices = file_indices[before_mask]
    dates = []
    unique_file_indices = np.unique(before_file_indices)
    for idx in unique_file_indices:
        idx = int(idx)
        if idx in file_map_dict:
            file_path = file_map_dict[idx][0]
            date_time, _ = parse_date_time(file_path)
            if date_time:
                dates.append(date_time)
    
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        date_str = min_date.strftime('%Y-%m-%d')
    else:
        date_range_str = ""
        date_str = ""
    num_points = len(before_data)
    
    before_heatmap = np.zeros_like(individual_heatmaps[0])
    # Sum heatmaps for before periods
    for idx in before_indices:
        before_heatmap += individual_heatmaps[idx]
    before_heatmap /= len(before_indices)
    
    # Normalize combined heatmaps
    before_heatmap = before_heatmap / before_heatmap.max()
    
    # Plot combined before
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(before_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   origin='lower', cmap='Purples', vmax=0.1, aspect='equal')
    ax.set_title(f"Combined Before\n{date_range_str}\nPoints: {num_points}", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    fig.tight_layout()
    filename = f"{bird_id}_combined_before_{date_str}"
    fig.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, f'{filename}.svg'))
    plt.close(fig)
    
    # Create combined after
    after_mask = np.isin(dataset_indices, after_indices)
    after_data = embedding_outputs[after_mask]
    
    after_file_indices = file_indices[after_mask]
    dates = []
    unique_file_indices = np.unique(after_file_indices)
    for idx in unique_file_indices:
        idx = int(idx)
        if idx in file_map_dict:
            file_path = file_map_dict[idx][0]
            date_time, _ = parse_date_time(file_path)
            if date_time:
                dates.append(date_time)
    
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        date_str = min_date.strftime('%Y-%m-%d')
    else:
        date_range_str = ""
        date_str = ""
    num_points = len(after_data)
    
    after_heatmap = np.zeros_like(individual_heatmaps[0])
    # Sum heatmaps for after periods
    for idx in after_indices:
        after_heatmap += individual_heatmaps[idx]
    after_heatmap /= len(after_indices)
    
    # Normalize combined heatmaps
    after_heatmap = after_heatmap / after_heatmap.max()
    
    # Plot combined after
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(after_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   origin='lower', cmap='Greens', vmax=0.1, aspect='equal')
    ax.set_title(f"Combined After\n{date_range_str}\nPoints: {num_points}", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    fig.tight_layout()
    filename = f"{bird_id}_combined_after_{date_str}"
    fig.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, f'{filename}.svg'))
    plt.close(fig)
    
    # Plot overlap
    brightness_factor = 4
    rgb_image = np.zeros((before_heatmap.shape[0], before_heatmap.shape[1], 3))
    rgb_image[..., 0] = np.clip(before_heatmap.T * brightness_factor, 0, 1)  # Red
    rgb_image[..., 1] = np.clip(after_heatmap.T * brightness_factor, 0, 1)   # Green
    rgb_image[..., 2] = np.clip(before_heatmap.T * brightness_factor, 0, 1)  # Blue
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
              origin='lower', aspect='equal')
    ax.set_title(f"Before vs After Overlap", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    fig.tight_layout()
    filename = f"{bird_id}_overlap_{date_str}"
    fig.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, f'{filename}.svg'))
    plt.close(fig)

def plot_all_heatmaps_combined(embedding_outputs, dataset_indices, save_dir, bird_id, file_indices, file_map, 
                               labels=['Before1', 'Before2', 'After1', 'After2']):
    """Create a single figure with all heatmaps arranged in a grid."""
    bins = 300
    
    # Set up the figure with subplots in a 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Get consistent binning using all data
    _, xedges, yedges = np.histogram2d(embedding_outputs[:, 0], embedding_outputs[:, 1], bins=bins)
    
    # Prepare file map dictionary
    if file_map is not None:
        file_map_dict = file_map.item()
    else:
        file_map_dict = {}
    
    # Store heatmaps for saving
    all_heatmaps = {}
    
    # Plot individual heatmaps in first 4 spots
    for i, label in enumerate(labels):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        mask = dataset_indices == i
        current_data = embedding_outputs[mask]
        
        heatmap, _, _ = np.histogram2d(current_data[:, 0], current_data[:, 1], 
                                       bins=[xedges, yedges])
        all_heatmaps[label] = heatmap  # Save unnormalized heatmap
        heatmap_norm = heatmap / heatmap.max()
        
        # Use green colormap for after periods (indices 2 and 3)
        cmap = 'Greens' if i >= 2 else 'Purples'
        ax.imshow(heatmap_norm.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                  origin='lower', cmap=cmap, vmax=0.1, aspect='equal')
        
        # Get dates and number of points
        group_file_indices = file_indices[mask]
        dates = []
        unique_file_indices = np.unique(group_file_indices)
        for idx in unique_file_indices:
            idx = int(idx)
            if idx in file_map_dict:
                file_path = file_map_dict[idx][0]
                date_time, _ = parse_date_time(file_path)
                if date_time:
                    dates.append(date_time)
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            num_points = len(current_data)
            ax.set_title(f"{label}\n{date_range_str}\nPoints: {num_points}", fontsize=14)
        else:
            num_points = len(current_data)
            ax.set_title(f"{label}\nPoints: {num_points}", fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
    
    # Create and plot combined before (top right)
    ax_before = fig.add_subplot(gs[0, 2])
    before_mask = np.isin(dataset_indices, [0, 1])
    before_data = embedding_outputs[before_mask]
    before_heatmap, _, _ = np.histogram2d(before_data[:, 0], before_data[:, 1], 
                                          bins=[xedges, yedges])
    all_heatmaps['Combined_Before'] = before_heatmap  # Save unnormalized heatmap
    before_heatmap_norm = before_heatmap / before_heatmap.max()
    
    # Get dates and number of points
    before_file_indices = file_indices[before_mask]
    dates = []
    unique_file_indices = np.unique(before_file_indices)
    for idx in unique_file_indices:
        idx = int(idx)
        if idx in file_map_dict:
            file_path = file_map_dict[idx][0]
            date_time, _ = parse_date_time(file_path)
            if date_time:
                dates.append(date_time)
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        num_points = len(before_data)
        ax_before.set_title(f"Combined Before\n{date_range_str}\nPoints: {num_points}", fontsize=14)
    else:
        num_points = len(before_data)
        ax_before.set_title(f"Combined Before\nPoints: {num_points}", fontsize=14)
    ax_before.imshow(before_heatmap_norm.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     origin='lower', cmap='Purples', vmax=0.1, aspect='equal')
    ax_before.set_xlabel('UMAP 1')
    ax_before.set_ylabel('UMAP 2')
    
    # Create and plot combined after (bottom right)
    ax_after = fig.add_subplot(gs[1, 2])
    after_mask = np.isin(dataset_indices, [2, 3])
    after_data = embedding_outputs[after_mask]
    after_heatmap, _, _ = np.histogram2d(after_data[:, 0], after_data[:, 1], 
                                         bins=[xedges, yedges])
    all_heatmaps['Combined_After'] = after_heatmap  # Save unnormalized heatmap
    after_heatmap_norm = after_heatmap / after_heatmap.max()
    
    # Get dates and number of points
    after_file_indices = file_indices[after_mask]
    dates = []
    unique_file_indices = np.unique(after_file_indices)
    for idx in unique_file_indices:
        idx = int(idx)
        if idx in file_map_dict:
            file_path = file_map_dict[idx][0]
            date_time, _ = parse_date_time(file_path)
            if date_time:
                dates.append(date_time)
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        date_range_str = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        num_points = len(after_data)
        ax_after.set_title(f"Combined After\n{date_range_str}\nPoints: {num_points}", fontsize=14)
    else:
        num_points = len(after_data)
        ax_after.set_title(f"Combined After\nPoints: {num_points}", fontsize=14)
    ax_after.imshow(after_heatmap_norm.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    origin='lower', cmap='Greens', vmax=0.1, aspect='equal')
    ax_after.set_xlabel('UMAP 1')
    ax_after.set_ylabel('UMAP 2')
    
    plt.suptitle(f'Bird {bird_id} - All UMAP Projections', fontsize=16)
    filename = f'{bird_id}_all_umaps'
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{filename}.svg'), bbox_inches='tight')
    plt.close()
    
    # Save unnormalized heatmaps
    np.savez(os.path.join(save_dir, f'bird_{bird_id}_heatmaps.npz'), **all_heatmaps)

def parse_date_time(file_path, format="standard"):
    """Parse the date and time from the file name."""
    # Example: 'USA5508_45570.28601193_10_5_7_56_41_segment_1.npz'
    try:
        file_base = os.path.basename(file_path)
        parts = file_base.split('_')
        month = int(parts[2])  # 10
        day = int(parts[3])    # 5
        hour = int(parts[4])   # 7
        minute = int(parts[5]) # 56
        second = int(parts[6]) # 41
        
        # You might need to adjust the year based on your data
        # For example, if data is from 2023
        year = 2023
        file_date = datetime(year, month, day, hour, minute, second)
        return file_date, parts[0]  # returns datetime object and bird ID
    except (ValueError, IndexError) as e:
        print(f"Error parsing date from {file_path}: {e}")
        return None, None

def plot_similarity_summary(results_df, save_dir):
    """Create summary plots of similarities across birds."""
    plt.rcParams.update({'font.size': 24, 'text.color': 'black', 'axes.labelcolor': 'black',
                        'xtick.color': 'black', 'ytick.color': 'black'})
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    data_to_plot = pd.melt(results_df[['Within_Before', 'Within_After', 'Between_Periods']], 
                          var_name='Comparison', value_name='Similarity')
    
    # Create bar plot
    sns.barplot(x='Comparison', y='Similarity', data=data_to_plot, ax=ax, 
                capsize=0.1, errwidth=2, ci=68)
    
    # Process each bird's data
    print("\nDetailed Information by Bird:")
    for idx, row in results_df.iterrows():
        bird_id = row['Bird']
        dataset_indices = row['dataset_indices']
        file_map_dict = row['file_map'].item()
        file_indices = row['file_indices']
        
        print(f"\nBird: {bird_id}")
        
        # Process each group
        for group_idx in range(4):
            mask = dataset_indices == group_idx
            point_count = np.sum(mask)
            unique_indices = np.unique(file_indices[mask])
            
            # Get dates from files
            dates = []
            for idx in unique_indices:
                idx = int(idx)
                if idx in file_map_dict:
                    file_path = file_map_dict[idx][0]
                    date_time, _ = parse_date_time(file_path)
                    if date_time:
                        dates.append(date_time)
            
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                
                # Create descriptive group name based on date range
                group_name = f"group_{min_date.strftime('%Y-%m-%d')}_to_{max_date.strftime('%Y-%m-%d')}"
                
                print(f"\n  {group_name}:")
                print(f"    Points: {point_count:,}")
                print(f"    Date Range: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")
                print(f"    Number of unique files: {len(unique_indices)}")
                print(f"    Number of files with valid dates: {len(dates)}")
                print(f"    Recording hours: {min([d.hour for d in dates]):02d}:00 to {max([d.hour for d in dates]):02d}:00")
                print(f"    Days spanned: {(max_date - min_date).days + 1}")
            else:
                print(f"\n  Group {group_idx}:")
                print(f"    Points: {point_count:,}")
                print("    Date Range: Could not determine dates")
                print(f"    Number of unique files: {len(unique_indices)}")
    
    # Calculate combined p-value using Fisher's method
    chi_square = -2 * np.sum(np.log(results_df['P_value']))
    dof = 2 * len(results_df)
    combined_p = 1 - stats.chi2.cdf(chi_square, dof)
    plt.title(f'p = {combined_p:.2e}', pad=20)
    
    # Update x-axis labels with point counts
    new_labels = []
    for comp in ['Within_Before', 'Within_After', 'Between_Periods']:
        total_points = 0
        for _, row in results_df.iterrows():
            dataset_indices = row['dataset_indices']
            if comp == 'Within_Before':
                mask = np.isin(dataset_indices, [0, 1])
            elif comp == 'Within_After':
                mask = np.isin(dataset_indices, [2, 3])
            else:  # Between_Periods
                mask = np.ones_like(dataset_indices, dtype=bool)
            total_points += np.sum(mask)
        
        label = f"{comp.replace('_', ' ')}\n(n={total_points:,})"
        new_labels.append(label)
    
    ax.set_xticklabels(new_labels, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarity_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'similarity_summary.svg'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # test on two birds
    npz_files = ["/home/george-vengrovski/Downloads/USA5508_Seasonality_4_Groups.npz", "/home/george-vengrovski/Downloads/USA5494_Seasonality_4_Groups.npz"]
    save_dir = "imgs/seasonality_analysis_results"
    results_df = analyze_multiple_birds(npz_files, save_dir)
