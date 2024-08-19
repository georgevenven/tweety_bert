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

weights_path = "experiments/LLB3_Whisperseg/saved_weights/model_step_27000.pth"
config_path = "experiments/LLB3_Whisperseg/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

plot_umap_projection(
    model=model, 
    device=device, 
    data_dirs=["/media/george-vengrovski/Extreme SSD/yarden_data/llb3_test", "/media/george-vengrovski/Extreme SSD/yarden_data/llb3_train"],
    samples=5e3, 
    category_colors_file="/home/george-vengrovski/Documents/projects/tweety_bert_paper/files/category_colors_llb3.pkl", 
    layer_index=-2, 
    dict_key="attention_output", 
    context=1000, 
    raw_spectogram=False,
    save_dict_for_analysis=False,
    save_name="test",
    plot_comparison=True,
)