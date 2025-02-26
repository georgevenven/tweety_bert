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
                if spectrogram.shape[0] > self.context_length:
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
        
        # Print shapes for debugging
        print(f"Segment shape: {segment_np.shape}")
        print(f"Output shape: {output_np.shape}")
        print(f"Mask shape: {mask_np.shape}")
        
        # Create a figure with 2 subplots (original with mask and prediction with mask)
        fig, axs = plt.subplots(2, 1, figsize=(20, 16))
        
        # Plot original spectrogram
        axs[0].imshow(segment_np, aspect='auto', origin='lower', cmap='viridis')
        axs[0].set_title('Original Spectrogram with Mask', fontsize=28)
        axs[0].set_xlabel('Time Bins', fontsize=22)
        axs[0].set_ylabel('Frequency Bins', fontsize=22)
        axs[0].tick_params(axis='both', which='major', labelsize=16)
        
        # Plot full model reconstruction
        axs[1].imshow(output_np.T, aspect='auto', origin='lower', cmap='viridis')
        axs[1].set_title('Full Model Reconstruction', fontsize=28)
        axs[1].set_xlabel('Time Bins', fontsize=22)
        axs[1].set_ylabel('Frequency Bins', fontsize=22)
        axs[1].tick_params(axis='both', which='major', labelsize=16)
        
        # Add mask indicators as solid red bars above both spectrograms
        # Find the y-limits of the plots
        y_min0, y_max0 = axs[0].get_ylim()
        y_min1, y_max1 = axs[1].get_ylim()
        height0 = 0.05 * (y_max0 - y_min0)
        height1 = 0.05 * (y_max1 - y_min1)
        
        # Create a separate figure to visualize the mask for debugging
        plt.figure(figsize=(20, 5))
        plt.imshow(mask_np, aspect='auto', cmap='Reds')
        plt.title('Mask Visualization (Red = Masked)', fontsize=24)
        plt.colorbar(label='Mask Value')
        plt.savefig(f"{self.output_dir}/mask_debug_{index}_{os.path.splitext(file_name)[0]}.png", dpi=300)
        plt.close()
        
        # Add mask indicators to both plots
        # The mask shape might be different from the spectrogram shape due to model architecture
        # We need to ensure we're mapping the mask correctly to the spectrogram time bins
        
        # For each time bin in the mask
        for i in range(mask_np.shape[1]):  # Assuming mask_np shape is [freq, time]
            # Check if any frequency bin at this time position is masked
            if np.any(mask_np[:, i] > 0.5):
                # Add red bar above the original spectrogram
                axs[0].add_patch(plt.Rectangle((i, y_max0), 1, height0, color='red', alpha=1.0, clip_on=False))
                # Add red bar above the reconstruction
                axs[1].add_patch(plt.Rectangle((i, y_max1), 1, height1, color='red', alpha=1.0, clip_on=False))
        
        # Adjust the y-limits to make room for the mask indicators
        axs[0].set_ylim(y_min0, y_max0 + height0)
        axs[1].set_ylim(y_min1, y_max1 + height1)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)  # Add some space between subplots
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
