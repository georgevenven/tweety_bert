def plot_dataset_comparison_from_file(npz_file_path, data_dirs, save_name):
    """
    Generate comparison plots between two datasets using saved UMAP embeddings.
    
    Parameters:
    - npz_file_path: path to the saved .npz file
    - data_dirs: list of original data directory paths (needed for titles)
    - save_name: name for saving the plots
    """
    # Load the saved data
    data = np.load(npz_file_path)
    embedding_outputs = data['embedding_outputs']
    dataset_indices = data['dataset_indices']

    # Create experiment-specific directory
    experiment_dir = os.path.join("imgs", "umap_plots", save_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Split data by dataset
    dataset_1_mask = dataset_indices == 0
    dataset_2_mask = dataset_indices == 1
    
    data_1 = embedding_outputs[dataset_1_mask]
    data_2 = embedding_outputs[dataset_2_mask]

    # Create 2D histograms
    bins = 300
    heatmap_1, xedges, yedges = np.histogram2d(data_1[:, 0], data_1[:, 1], bins=bins)
    heatmap_2, _, _ = np.histogram2d(data_2[:, 0], data_2[:, 1], bins=[xedges, yedges])

    # Normalize heatmaps
    heatmap_1 = heatmap_1 / heatmap_1.max()
    heatmap_2 = heatmap_2 / heatmap_2.max()

    # Create RGB image for overlap visualization
    rgb_image = np.zeros((heatmap_1.shape[0], heatmap_1.shape[1], 3))
    brightness_factor = 4
    
    # Purple for dataset 1, Green for dataset 2
    rgb_image[..., 0] = np.clip(heatmap_1.T * brightness_factor, 0, 1)  # Red channel
    rgb_image[..., 1] = np.clip(heatmap_2.T * brightness_factor, 0, 1)  # Green channel
    rgb_image[..., 2] = np.clip(heatmap_1.T * brightness_factor, 0, 1)  # Blue channel

    # Plot dataset 1
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(heatmap_1.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
               origin='lower', cmap='Purples', vmax=0.1)
    ax1.set_title(f"Dataset: {os.path.basename(data_dirs[0])}", fontsize=16)
    ax1.set_xlabel('UMAP Dimension 1')
    ax1.set_ylabel('UMAP Dimension 2')
    fig1.tight_layout()
    fig1.savefig(os.path.join(experiment_dir, "dataset_1.png"), dpi=300)
    fig1.savefig(os.path.join(experiment_dir, "dataset_1.svg"))
    plt.close(fig1)

    # Plot dataset 2
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.imshow(heatmap_2.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
               origin='lower', cmap='Greens', vmax=0.1)
    ax2.set_title(f"Dataset: {os.path.basename(data_dirs[1])}", fontsize=16)
    ax2.set_xlabel('UMAP Dimension 1')
    fig2.tight_layout()
    fig2.savefig(os.path.join(experiment_dir, "dataset_2.png"), dpi=300)
    fig2.savefig(os.path.join(experiment_dir, "dataset_2.svg"))
    plt.close(fig2)

    # Plot overlap
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.imshow(rgb_image, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
               origin='lower')
    ax3.set_title("Overlapping Datasets", fontsize=16)
    ax3.set_xlabel('UMAP Dimension 1')
    fig3.tight_layout()
    fig3.savefig(os.path.join(experiment_dir, "overlap.png"), dpi=300)
    fig3.savefig(os.path.join(experiment_dir, "overlap.svg"))
    plt.close(fig3)