import torch
import os
import sys

figure_generation_dir = os.path.dirname(__file__)
project_root = os.path.dirname(figure_generation_dir)
os.chdir(project_root)

print(project_root)

sys.path.append("src")

from utils import load_model
from analysis import plot_umap_projection

# THIS SHOULD ALWAYS BE CPU SO YOU CAN FIT SUPER LONG SONGS IN MODEL UNLESS YOU HAVE A100 or BETTER!
device = torch.device("cpu")

weights_path = "experiments/LLB16_Whisperseg_NoNormNoThresholding/saved_weights/model_step_21000.pth"
config_path = "experiments/LLB16_Whisperseg_NoNormNoThresholding/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

plot_umap_projection(
    model=model, 
    device=device, 
    data_dir="/media/george-vengrovski/Extreme SSD1/yarden_data/llb16_no_threshold_no_norm",
    samples=1e6, 
    category_colors_file="/home/george-vengrovski/Downloads/category_colors_llb3.pkl", 
    layer_index=-2, 
    dict_key="attention_output", 
    context=1000, 
    raw_spectogram=False,
    save_dict_for_analysis=True,
    save_name="LLB16NONORM"
)
