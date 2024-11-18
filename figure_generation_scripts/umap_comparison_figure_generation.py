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

def load_embedding_data(npz_file_path):
    """Load embedding data and metadata from NPZ file."""
    data = np.load(npz_file_path, allow_pickle=True)
    return {
        'embedding_outputs': data['embedding_outputs'],
        'dataset_indices': data['dataset_indices']
    }


def calculate_distribution_similarity(heatmap1, heatmap2):
   """Calculate Bhattacharyya coefficient between two distributions."""
   eps = 1e-10
   p = heatmap1 / (np.sum(heatmap1) + eps)
   q = heatmap2 / (np.sum(heatmap2) + eps)
   
   similarity = np.sum(np.sqrt(p * q))
   
   return similarity

def calculate_statistical_significance(similarity_matrix):
    """Calculate statistical significance of within vs between period differences."""
    # Extract within and between period similarities
    within_periods = [
        similarity_matrix[0,1],  # Before1 vs Before2
        similarity_matrix[2,3]   # After1 vs After2
    ]
    
    between_periods = [
        similarity_matrix[0,2], similarity_matrix[0,3],  # Before1 vs After
        similarity_matrix[1,2], similarity_matrix[1,3]   # Before2 vs After
    ]
    
    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(within_periods, between_periods,
                                            alternative='greater')
    
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

def plot_pairwise_comparisons(embedding_outputs, dataset_indices, save_dir, bird_id):
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
    
    for bird_id, npz_file in enumerate(npz_files, 1):
        print(f"\nProcessing bird {bird_id} from {npz_file}")
        
        # Load data
        data = load_embedding_data(npz_file)
        embedding_outputs = data['embedding_outputs']
        dataset_indices = data['dataset_indices']
        
        # Calculate similarities and plot
        result = plot_pairwise_comparisons(embedding_outputs, dataset_indices, save_dir, bird_id)
        result['bird_id'] = bird_id
        results.append(result)
        
        # Add the combined heatmap plot
        plot_all_heatmaps_combined(embedding_outputs, dataset_indices, save_dir, bird_id)
    
    # Compile results
    if results:
        # Create DataFrame and plot similarity summary
        results_df = pd.DataFrame({
            'Bird': [r['bird_id'] for r in results],
            'Within_Before': [r['within_before'] for r in results],
            'Within_After': [r['within_after'] for r in results],
            'Between_Periods': [r['between_periods'] for r in results],
            'P_value': [r['p_value'] for r in results],
            'Within_Mean': [r['within_mean'] for r in results],
            'Within_Std': [r['within_std'] for r in results],
            'Between_Mean': [r['between_mean'] for r in results],
            'Between_Std': [r['between_std'] for r in results]
        })
        
        plot_similarity_summary(results_df, save_dir)
        results_df.to_csv(os.path.join(save_dir, 'all_birds_results.csv'))
        print("\nDetailed Results Summary:")
        print(results_df.to_string(index=False))
        return results_df
    else:
        print("No valid results to report")
        return None

def plot_separate_and_combined_umaps(embedding_outputs, dataset_indices, save_dir, bird_id, 
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
        
        # Plot individual map with appropriate colormap
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = 'Greens' if i in after_indices else 'Purples'  # Use green for after periods
        im = ax.imshow(heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                       origin='lower', cmap=cmap, vmax=0.1)
        ax.set_title(f"{label}", fontsize=16)
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_{label}_umap.png'), dpi=300)
        fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_{label}_umap.svg'))
        plt.close(fig)
    
    # Create combined before/after comparison
    before_heatmap = np.zeros_like(individual_heatmaps[0])
    after_heatmap = np.zeros_like(individual_heatmaps[0])
    
    # Sum heatmaps for before and after periods
    for idx in before_indices:
        before_heatmap += individual_heatmaps[idx]
    before_heatmap /= len(before_indices)
    
    for idx in after_indices:
        after_heatmap += individual_heatmaps[idx]
    after_heatmap /= len(after_indices)
    
    # Normalize combined heatmaps
    before_heatmap = before_heatmap / before_heatmap.max()
    after_heatmap = after_heatmap / after_heatmap.max()
    
    # Create RGB overlay
    rgb_image = np.zeros((before_heatmap.shape[0], before_heatmap.shape[1], 3))
    brightness_factor = 4
    
    # Purple for before, Green for after
    rgb_image[..., 0] = np.clip(before_heatmap.T * brightness_factor, 0, 1)  # Red
    rgb_image[..., 1] = np.clip(after_heatmap.T * brightness_factor, 0, 1)   # Green
    rgb_image[..., 2] = np.clip(before_heatmap.T * brightness_factor, 0, 1)  # Blue
    
    # Plot combined before
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(before_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   origin='lower', cmap='Purples', vmax=0.1)
    ax.set_title(f"Combined Before", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_combined_before.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_combined_before.svg'))
    plt.close(fig)
    
    # Plot combined after
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(after_heatmap.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   origin='lower', cmap='Greens', vmax=0.1)
    ax.set_title(f"Combined After", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_combined_after.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_combined_after.svg'))
    plt.close(fig)
    
    # Plot overlap
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_image, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
              origin='lower')
    ax.set_title("Before vs After Overlap", fontsize=16)
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_overlap.png'), dpi=300)
    fig.savefig(os.path.join(save_dir, f'bird_{bird_id}_overlap.svg'))
    plt.close(fig)

def plot_all_heatmaps_combined(embedding_outputs, dataset_indices, save_dir, bird_id, 
                               labels=['Before1', 'Before2', 'After1', 'After2']):
    """Create a single figure with all heatmaps arranged in a grid."""
    bins = 300
    
    # Set up the figure with subplots in a 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Get consistent binning using all data
    _, xedges, yedges = np.histogram2d(embedding_outputs[:, 0], embedding_outputs[:, 1], bins=bins)
    
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
                  origin='lower', cmap=cmap, vmax=0.1)
        ax.set_title(f"{label}", fontsize=14)
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
    
    ax_before.imshow(before_heatmap_norm.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     origin='lower', cmap='Purples', vmax=0.1)
    ax_before.set_title("Combined Before", fontsize=14)
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
    
    ax_after.imshow(after_heatmap_norm.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    origin='lower', cmap='Greens', vmax=0.1)
    ax_after.set_title("Combined After", fontsize=14)
    ax_after.set_xlabel('UMAP 1')
    ax_after.set_ylabel('UMAP 2')
    
    plt.suptitle(f'Bird {bird_id} - All UMAP Projections', fontsize=16)
    plt.savefig(os.path.join(save_dir, f'bird_{bird_id}_all_umaps.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'bird_{bird_id}_all_umaps.svg'), bbox_inches='tight')
    plt.close()
    
    # Save unnormalized heatmaps
    np.savez(os.path.join(save_dir, f'bird_{bird_id}_heatmaps.npz'), **all_heatmaps)

def plot_similarity_summary(results_df, save_dir):
    """
    Create summary plots of similarities across birds.
    """
    # Set the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Within vs Between Period Similarities
    data_to_plot = pd.melt(results_df[['Within_Before', 'Within_After', 'Between_Periods']], 
                           var_name='Comparison', value_name='Similarity')
    
    sns.boxplot(x='Comparison', y='Similarity', data=data_to_plot, ax=ax1)
    sns.swarmplot(x='Comparison', y='Similarity', data=data_to_plot, color='0.25', ax=ax1)
    
    ax1.set_title('Distribution of Similarities Across Birds', fontsize=12)
    ax1.set_xlabel('Comparison Type')
    ax1.set_ylabel('Similarity (Cosine Similarity)')
    
    # Plot 2: P-values
    sns.histplot(data=results_df['P_value'], ax=ax2, bins=20)
    ax2.axvline(0.05, color='r', linestyle='--', label='p=0.05')
    ax2.set_title('Distribution of P-values', fontsize=12)
    ax2.set_xlabel('P-value')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    # Add overall p-value using Fisher's method
    chi_square = -2 * np.sum(np.log(results_df['P_value']))
    dof = 2 * len(results_df)
    combined_p = 1 - stats.chi2.cdf(chi_square, dof)
    plt.suptitle(f'Similarity Analysis Summary (Combined p = {combined_p:.2e})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarity_summary.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'similarity_summary.svg'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    npz_files = ["files/USA5508_1million_test.npz"]
    save_dir = "seasonal_analysis_results"
    results_df = analyze_multiple_birds(npz_files, save_dir) 