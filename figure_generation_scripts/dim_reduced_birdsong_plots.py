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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path = "/media/george-vengrovski/Extreme SSD/YardenLLB3_PreExtracted_WithPitch_Shift/saved_weights/model_step_26500.pth"
config_path = "/media/george-vengrovski/Extreme SSD/YardenLLB3_PreExtracted_WithPitch_Shift/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

plot_umap_projection(
    model=model, 
    device=device, 
    data_dir="/media/george-vengrovski/Extreme SSD/yarden_data/llb3_data_matrices",
    samples=5e4, 
    file_path="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
    layer_index=-2, 
    dict_key="attention_output", 
    context=1000, 
    raw_spectogram=False,
    save_dict_for_analysis = False,
    save_name="heatmap_test",
)
