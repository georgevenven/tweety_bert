import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("src")

os.chdir('/home/george-vengrovski/Documents/projects/tweety_bert_paper')

# For removing mat files 
folders = ['/media/george-vengrovski/disk2/canary_temp/combined_yarden_specs']

for folder in folders:
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith('.mat'):
                os.remove(os.path.join(folder, filename))
                print(f'Removed {filename} from {folder}')
    else:
        print(f'Folder does not exist: {folder}')

