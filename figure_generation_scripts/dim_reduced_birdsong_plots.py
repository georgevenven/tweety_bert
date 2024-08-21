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
    samples=1e6, 
    category_colors_file="/home/george-vengrovski/Downloads/category_colors_llb3.pkl", 
    layer_index=-2, 
    dict_key="attention_output", 
    context=1000, 
    raw_spectogram=False,
    save_dict_for_analysis=True,
    save_name="Yarden_LLB3_Whisperseg",
    plot_comparison=True,
    truncate_to_smallest_group= False
)


weights_path = "experiments/Sham_Lesion_5271/saved_weights/model_step_23500.pth"
config_path = "experiments/Sham_Lesion_5271/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

plot_umap_projection(
    model=model, 
    device=device, 
    data_dirs=["/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5271_before_3-07", "/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5271_after_3-07"],
    samples=1e6, 
    category_colors_file="/home/george-vengrovski/Downloads/category_colors_llb3.pkl", 
    layer_index=-2, 
    dict_key="attention_output", 
    context=1000, 
    raw_spectogram=False,
    save_dict_for_analysis=True,
    save_name="SHAM_USA5271",
    plot_comparison=True,
    truncate_to_smallest_group= False
)

weights_path = "experiments/Sham_Lesion_5283/saved_weights/model_step_36500.pth"
config_path = "experiments/Sham_Lesion_5283/config.json"

model = load_model(config_path, weights_path)
model = model.to(device)

plot_umap_projection(
    model=model, 
    device=device, 
    data_dirs=["/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5283_before_3-05", "/media/george-vengrovski/Extreme SSD/sham lesioned birds/USA5283_after_3-05"],
    samples=1e6, 
    category_colors_file="/home/george-vengrovski/Downloads/category_colors_llb3.pkl", 
    layer_index=-2, 
    dict_key="attention_output", 
    context=1000, 
    raw_spectogram=False,
    save_dict_for_analysis=True,
    save_name="SHAM_USA5283",
    plot_comparison=True,
    truncate_to_smallest_group= False
)