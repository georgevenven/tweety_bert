import os

import sys
sys.path.append("src")


from spectogram_generator import WavtoSpec


wav_to_spec = WavtoSpec('/media/george-vengrovski/disk2/budgie/T5_ssd_combined', '/media/george-vengrovski/disk2/budgie/T5_ssd_combined_specs')
wav_to_spec.process_directory()