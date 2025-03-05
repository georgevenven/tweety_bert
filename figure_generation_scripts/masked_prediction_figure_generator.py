import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

# Now import from src
from data_class import SongDataSet_Image, CollateFunction
from utils import load_model, get_device

# Configuration - modify these paths as needed
MODEL_DIR = "experiments/TweetyBERT_Paper_Yarden_Model"  # Directory containing the model checkpoint
DATA_DIR = "/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_specs"  # Directory containing spectrogram files
OUTPUT_DIR = "imgs/masked_predictions_for_figure_2"  # Output directory for visualizations
CONTEXT_LENGTH = 1000  # Context length for the model
NUM_SAMPLES = 100  # Number of samples to process
MASK_PERCENTAGE = 0.25  # 25% masking as mentioned

class MaskedPredictionVisualizer:
    def __init__(self, model_dir, data_dir, output_dir, context_length=1000, num_samples=100, mask_percentage=0.25):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.context_length = context_length
        self.num_samples = num_samples
        self.mask_percentage = mask_percentage
        self.device = get_device()
        
        # Create output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        self.model = self.load_model()
        
        # Create data loader
        self.data_loader = self.create_dataloader()
    
    def load_model(self):
        print(f"Loading model from {self.model_dir}")
        model = load_model(self.model_dir)
        model.to(self.device)
        model.eval()
        return model
    
    def create_dataloader(self):
        dataset = SongDataSet_Image(
            self.data_dir, 
            num_classes=40,  # This doesn't matter for reconstruction
            infinite_loader=False, 
            segment_length=None  # We'll handle segmentation manually
        )
        
        # No need for collate function as we'll process one sample at a time
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        return loader
    
    def z_score_spectrogram(self, spec):
        spec_mean = np.mean(spec)
        spec_std = np.std(spec)
        return (spec - spec_mean) / spec_std
    
    def generate_visualizations(self):
        processed_samples = 0
        
        with torch.no_grad():
            for i, (spectrogram, _, _, file_name) in enumerate(tqdm(self.data_loader, desc="Generating visualizations")):
                if processed_samples >= self.num_samples:
                    break
                
                # Get the original spectrogram shape
                spectrogram = spectrogram.squeeze(0)  # Remove batch dimension
                
                # Save the full high-resolution spectrogram
                self.save_full_spectrogram(spectrogram, file_name[0], i)
                
                # Process segments if the spectrogram is longer than context_length
                if spectrogram.shape[0] > self.context_length:umba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  @numba.jit()
Traceback (most recent call last):
  File "/home/george-vengrovski/Documents/projects/tweety_bert_paper/figure_generation_scripts/visualizing_hdb_scan_labels.py", line 10, in <module>
    from src.analysis import ComputerClusterPerformance
  File "/home/george-vengrovski/Documents/projects/tweety_bert_paper/src/analysis.py", line 8, in <module>
    from data_class import SongDataSet_Image, CollateFunction
ModuleNotFoundError: No module named 'data_class'
(base) george-vengrovski@gardner-lambda:~/Documents/projects/tweety_bert_paper$ 
                    # Choose a random segment
                    start_idx = torch.randint(0, spectrogram.shape[0] - self.context_length, (1,)).item()
                    segment = spectrogram[start_idx:start_idx + self.context_length]
                    
                    # Generate and save the masked prediction visualization
                    self.process_segment(segment, file_name[0], i, start_idx)
                    
                    processed_samples += 1
                else:
                    print(f"Skipping {file_name[0]} - too short ({spectrogram.shape[0]} < {self.context_length})")
    
    def save_full_spectrogram(self, spectrogram, file_name, index):
        plt.figure(figsize=(30, 10))
        # Transpose the spectrogram for visualization (time on x-axis, frequency on y-axis)
        plt.imshow(spectrogram.cpu().numpy().T, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Full Spectrogram - {file_name}', fontsize=24)
        plt.xlabel('Time Bins', fontsize=18)
        plt.ylabel('Frequency Bins', fontsize=18)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/full_spec_{index}_{os.path.splitext(file_name)[0]}.png", dpi=300)
        plt.close()
    
    def process_segment(self, segment, file_name, index, start_idx):
        # Prepare input for the model - use the model's masking operation
        segment_input = segment.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).to(self.device)
        
        # Use the model's masking operation directly
        # Calculate m based on the percentage (25% of the context length)
        m = int(self.context_length * self.mask_percentage)
        
        # Get a copy of the original input before masking
        original_input = segment_input.clone()
        
        # Forward pass through the model using train_forward which applies masking
        output, mask, _, _ = self.model.train_forward(segment_input)
        
        # Create visualization
        self.visualize_masked_prediction(original_input, mask, output, file_name, index, start_idx)
    
    def visualize_masked_prediction(self, segment, mask, output, file_name, index, start_idx):
        # Convert tensors to numpy arrays for visualization
        segment_np = segment.squeeze().cpu().numpy()  # Shape: (196, 1000)
        mask_np = mask.squeeze().cpu().numpy()
        output_np = output.squeeze().cpu().numpy()  # Shape: (1000, 196)
        
        # Create a figure with 2 subplots with more space between them
        fig = plt.figure(figsize=(20, 18))  # Increased figure height
        
        # Create two axes with more space between them
        ax1 = fig.add_axes([0.1, 0.55, 0.8, 0.35])  # Original spectrogram
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.35])   # Reconstruction
        
        # Plot original spectrogram
        ax1.imshow(segment_np, aspect='auto', origin='lower', cmap='viridis')
        # Remove x-axis label and ticks from top spectrogram
        ax1.set_xlabel('')
        ax1.set_xticks([])
        ax1.set_ylabel('Frequency Bins', fontsize=44)  # 2x bigger
        ax1.tick_params(axis='y', which='major', labelsize=32)  # 2x bigger
        
        # Plot full model reconstruction
        ax2.imshow(output_np.T, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_xlabel('Time Bins', fontsize=44)  # 2x bigger
        ax2.set_ylabel('Frequency Bins', fontsize=44)  # 2x bigger
        ax2.tick_params(axis='both', which='major', labelsize=32)  # 2x bigger
        
        # Create mask indicator arrays (1 for masked, 0 for unmasked)
        mask_indicator = np.zeros((1, mask_np.shape[1]))
        for i in range(mask_np.shape[1]):
            if np.any(mask_np[:, i] > 0.5):
                mask_indicator[0, i] = 1
        
        # Create a custom colormap with pure white for unmasked and bright red for masked regions
        cmap = plt.cm.colors.ListedColormap(['#FFFFFF', '#990000'])
        
        # Add mask indicators completely outside the spectrograms
        # For the first spectrogram
        mask_ax1 = fig.add_axes([0.1, 0.91, 0.8, 0.03])  # Above the first spectrogram
        mask_ax1.imshow(mask_indicator, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        mask_ax1.set_xticks([])
        mask_ax1.set_yticks([])
        mask_ax1.axis('off')  # Remove the box outline
        
        # For the second spectrogram
        mask_ax2 = fig.add_axes([0.1, 0.46, 0.8, 0.03])  # Above the second spectrogram
        mask_ax2.imshow(mask_indicator, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        mask_ax2.set_xticks([])
        mask_ax2.set_yticks([])
        mask_ax2.axis('off')  # Remove the box outline
        
        # Add titles above the mask indicators
        fig.text(0.5, 0.96, 'Original Spectrogram with Mask', ha='center', fontsize=42)
        fig.text(0.5, 0.51, 'Full Model Reconstruction', ha='center', fontsize=42)
        
        plt.savefig(f"{self.output_dir}/masked_pred_{index}_{os.path.splitext(file_name)[0]}.png", dpi=300)
        plt.close()

def main():
    visualizer = MaskedPredictionVisualizer(
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        context_length=CONTEXT_LENGTH,
        num_samples=NUM_SAMPLES,
        mask_percentage=MASK_PERCENTAGE
    )
    
    visualizer.generate_visualizations()

if __name__ == "__main__":
    main()
