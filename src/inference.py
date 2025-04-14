import os
import json
import torch
import subprocess
from tqdm import tqdm
from torch.nn import functional as F
from spectogram_generator import WavtoSpec
from linear_probe import LinearProbeModel, LinearProbeTrainer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import numpy as np
from utils import load_model
from pathlib import Path
from datetime import datetime

def get_creation_date(path):
    stat = path.stat()
    if hasattr(stat, 'st_birthtime'):
        return stat.st_birthtime  # macos
    elif os.name == 'nt':
        return stat.st_ctime      # windows
    else:
        return stat.st_mtime      # linux/unix fallback

def tweety_net_detector_inference(input_file, return_json=True, mode="local_file"):
   os.environ["MKL_THREADING_LAYER"] = "GNU"
   command = [
       'python', '/home/george-vengrovski/Documents/projects/tweety_net_song_detector/src/inference.py',
       '--mode', mode,
       '--input', input_file
   ]
   if return_json:
       command.append('--return_json')
   result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   if result.returncode == 0:
       try:
           output_json = json.loads(result.stdout)
           return output_json
       except json.JSONDecodeError:
           print("Error parsing JSON output.")
           return None
   else:
       print(f"Error executing command: {result.stderr}")
       return None

class TweetyBertInference:
  def __init__(self, classifier_path, spec_dst_folder, output_path, song_detection_json=None, visualize=False, dump_interval=500, apply_post_processing=True):
      print(output_path)
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.classifier = self.load_decoder_state(classifier_path)
      self.wav_to_spec = None
      self.spec_dst_folder = spec_dst_folder
      self.output_path = output_path
      self.visualize = visualize
      self.dump_interval = dump_interval
      self.apply_post_processing = apply_post_processing
      if song_detection_json is not None:
          with open(song_detection_json, 'r') as f:
              self.song_detection_data = json.load(f)
      else:
          self.song_detection_data = None
      os.makedirs(self.spec_dst_folder, exist_ok=True)
      base_colors = plt.colormaps['tab20'](np.linspace(0, 1, 20))
      additional_colors = plt.colormaps['Set2'](np.linspace(0, 1, 8))
      colors = np.vstack((base_colors, additional_colors))
      colors = colors[np.random.permutation(len(colors))]
      colors = np.vstack(([1, 1, 1, 1], colors))
      self.cmap = mcolors.ListedColormap(colors[:self.num_classes])

  def load_decoder_state(self, linear_decoder_dir):
      save_dir = os.path.join(linear_decoder_dir, "decoder_state")
      with open(os.path.join(save_dir, "decoder_config.json"), "r") as f:
          config = json.load(f)
      self.num_classes = config["num_classes"]
      self.data_file = config["data_file"]
      self.model_dir = config["model_dir"]
      tweety_bert_json = os.path.join(config["model_dir"], "config.json")
      with open(tweety_bert_json, "r") as f:
          config = json.load(f)
      self.context_length = config["context"]
      tweety_bert_model = load_model(self.model_dir)
      self.create_classifier(tweety_bert_model)
      model = self.classifier_model.load_state_dict(torch.load(os.path.join(save_dir, "decoder_weights.pth")))
      print(f"Decoder state loaded from {save_dir}")
      return model

  def create_classifier(self, tweety_bert_model):
      if self.num_classes is None:
          raise ValueError("Number of classes is not set. Run prepare_data first.")
      self.classifier_model = LinearProbeModel(
          num_classes=self.num_classes,
          model_type="neural_net",
          model=tweety_bert_model,
          freeze_layers=True,
          layer_num=-1,
          layer_id="attention_output",
          TweetyBERT_readout_dims=196,
          classifier_type="decoder"
      ).to(self.device)

  def setup_wav_to_spec(self, folder, csv_file_dir=None):
      self.wav_to_spec = WavtoSpec(folder, self.spec_dst_folder, csv_file_dir)

  def smooth_labels(self, labels, window_size=50):
      labels = np.array(labels)
      for i in range(len(labels)):
          if labels[i] == -1:
              left = right = i
              while left >= 0 or right < len(labels):
                  if left >= 0 and labels[left] != -1:
                      labels[i] = labels[left]
                      break
                  if right < len(labels) and labels[right] != -1:
                      labels[i] = labels[right]
                      break
                  left -= 1
                  right += 1
      smoothed_labels = np.zeros_like(labels)
      for i in range(len(labels)):
          start = max(0, i - window_size // 2)
          end = min(len(labels), i + window_size // 2 + 1)
          window = labels[start:end]
          unique, counts = np.unique(window, return_counts=True)
          smoothed_labels[i] = unique[np.argmax(counts)]
      return smoothed_labels

  def process_file(self, file_path):
      # Print detection method being used
      if self.song_detection_data is None:
          print(f"Using TweetyNET detection for {os.path.basename(file_path)}")
      else:
          print(f"Using pre-designated JSON for {os.path.basename(file_path)}")

      file_creation_timestamp = get_creation_date(Path(file_path))
      creation_date_str = datetime.fromtimestamp(file_creation_timestamp).isoformat()

      spec, vocalization, labels = self.wav_to_spec.multiprocess_process_file(file_path, self.wav_to_spec, json_path=self.song_detection_data, save_npz=False)
      if spec is None:
          return {
              "file_name": os.path.basename(file_path),
              "creation_date": creation_date_str,
              "song_present": False,
              "syllable_onsets_offsets_ms": {},
              "syllable_onsets_offsets_timebins": {}
          }
      if self.song_detection_data is None:
          vocalization_data = tweety_net_detector_inference(input_file=file_path)
          if not vocalization_data or vocalization_data.get('segments') == []:
              return {
                  "file_name": os.path.basename(file_path),
                  "creation_date": creation_date_str,
                  "song_present": False,
                  "syllable_onsets_offsets_ms": {},
                  "syllable_onsets_offsets_timebins": {}
              }
      else:
          file_name = os.path.basename(file_path)
          vocalization_data = next((item for item in self.song_detection_data if item["filename"] == file_name), None)
          if vocalization_data is None:
              vocalization_data = tweety_net_detector_inference(input_file=file_path)
          if vocalization_data is not None and not vocalization_data.get("song_present", False):
              return {
                  "file_name": file_name,
                  "creation_date": creation_date_str,
                  "song_present": False,
                  "syllable_onsets_offsets_ms": {},
                  "syllable_onsets_offsets_timebins": {}
              }
      vocalization_data = vocalization_data['segments'][0]
      song_spec = spec[:, vocalization_data['onset_timebin']:vocalization_data['offset_timebin']]
      spectogram, pad_amount = self.inference_data_class(song_spec)
      spec_tensor = torch.Tensor(spectogram).to(self.device).unsqueeze(1)
      logits = self.classifier_model(spec_tensor.permute(0,1,3,2))
      logits = logits.reshape(logits.shape[0] * logits.shape[1], -1)
      predicted_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
      if self.apply_post_processing:
          post_processed_labels = self.smooth_labels(predicted_labels, window_size=50)
          post_processed_labels[-pad_amount:] = -1
          post_processed_labels = post_processed_labels[:-pad_amount]
      else:
          post_processed_labels = predicted_labels
      onsets_offsets_ms, onsets_offsets_timebins = self.convert_to_onset_offset(post_processed_labels)
      song_present = len(onsets_offsets_ms) > 0
      if self.visualize:
          self.visualize_spectrogram(spectogram.flatten(0,1)[:-pad_amount].T, post_processed_labels, os.path.basename(file_path))
      return {
          "file_name": os.path.basename(file_path),
          "creation_date": creation_date_str,
          "song_present": song_present,
          "syllable_onsets_offsets_ms": onsets_offsets_ms,
          "syllable_onsets_offsets_timebins": onsets_offsets_timebins
      }

  def process_folder(self, folder_path, save_interval=100):
      results = []
      file_count = 0
      processed_files = set()
      if os.path.exists(self.output_path):
        print(self.output_path)
        with open(self.output_path, 'r') as f:
            loaded_data = json.load(f)
            results = loaded_data.get('results', [])
        processed_files = set(result['file_name'] for result in results)
        file_count = len(results)
      wav_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.lower().endswith('.wav')]
      for file_path in tqdm(wav_files, desc="Processing files"):
          if os.path.basename(file_path) in processed_files:
              continue
          try:
              result = self.process_file(file_path)
              results.append(result)
              file_count += 1
              if file_count % self.dump_interval == 0:
                  self.save_results(results, self.output_path)
          except Exception as e:
              print(f"Error processing {file_path}: {e}")
      self.save_results(results, self.output_path)
      return results

  def save_results(self, results, output_path):
      metadata = {
          "classifier_path": self.model_dir,
          "spec_dst_folder": self.spec_dst_folder,
          "output_path": self.output_path,
          "visualize": self.visualize,
          "dump_interval": self.dump_interval,
          "apply_post_processing": self.apply_post_processing
      }
      with open(output_path, 'w') as f:
          json.dump({"metadata": metadata, "results": results}, f, indent=4)

  def inference_data_class(self, data):
      recording_length = data.shape[1]
      spectogram = data[20:216]
      spec_mean = np.mean(spectogram)
      spec_std = np.std(spectogram)
      spectogram = (spectogram - spec_mean) / spec_std
      spectogram = torch.from_numpy(spectogram).float().permute(1, 0)
      pad_amount = self.context_length - (recording_length % self.context_length)
      if recording_length < self.context_length:
          pad_amount = self.context_length - recording_length
      if recording_length > self.context_length and pad_amount != 0:
          pad_amount = self.context_length - (spectogram.shape[0] % self.context_length)
      spectogram = F.pad(spectogram, (0, 0, 0, pad_amount), 'constant', 0)
      spectogram = spectogram.reshape(spectogram.shape[0] // self.context_length, self.context_length, spectogram.shape[1])
      return spectogram, pad_amount

  def convert_to_onset_offset(self, labels):
      sampling_rate = 44100
      NFFT = 1024
      hop_length = 119
      ms_per_timebin = (hop_length / sampling_rate) * 1000
      syllable_dict = {}
      syllable_dict_no_ms = {}
      current_label = -1
      start_time_label = None
      for i, label in enumerate(labels):
          if label != current_label:
              if current_label != -1 and start_time_label is not None:
                  end_time = i * ms_per_timebin
                  end_time_no_ms = i
                  if current_label not in syllable_dict:
                      syllable_dict[current_label] = []
                  if current_label not in syllable_dict_no_ms:
                      syllable_dict_no_ms[current_label] = []
                  syllable_dict[current_label].append([start_time_label, end_time])
                  syllable_dict_no_ms[current_label].append([start_time_label / ms_per_timebin, end_time_no_ms])
              if label != -1:
                  start_time_label = i * ms_per_timebin
              current_label = label
      if current_label != -1 and start_time_label is not None:
          end_time = len(labels) * ms_per_timebin
          end_time_no_ms = len(labels)
          if current_label not in syllable_dict:
              syllable_dict[current_label] = []
          if current_label not in syllable_dict_no_ms:
              syllable_dict_no_ms[current_label] = []
          syllable_dict[current_label].append([start_time_label, end_time])
          syllable_dict_no_ms[current_label].append([start_time_label / ms_per_timebin, end_time_no_ms])
      syllable_dict = {int(key): value for key, value in syllable_dict.items()}
      syllable_dict_no_ms = {int(key): value for key, value in syllable_dict_no_ms.items()}
      return syllable_dict, syllable_dict_no_ms

  def visualize_spectrogram(self, spec, predicted_labels, file_name):
      plt.figure(figsize=(15, 10))
      plt.subplot(2, 1, 1)
      plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
      plt.title('Spectrogram', fontsize=24)
      plt.colorbar(format='%+2.0f dB')
      plt.subplot(2, 1, 2)
      im = plt.imshow([predicted_labels], aspect='auto', origin='lower', cmap=self.cmap, vmin=0, vmax=self.num_classes-1)
      plt.title('Predicted Labels', fontsize=24)
      cbar = plt.colorbar(im, ticks=range(self.num_classes))
      cbar.set_label('Syllable Class')
      cbar.set_ticklabels(['Silence'] + [f'Class {i}' for i in range(1, self.num_classes)])
      plt.tight_layout()
      output_path = os.path.join(self.spec_dst_folder, f"{os.path.splitext(file_name)[0]}_visualization.png")
      plt.savefig(output_path, dpi=300, bbox_inches='tight')
      plt.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run TweetyBert Inference")
  parser.add_argument('--bird_name', type=str, required=True, help='Name of the bird')
  parser.add_argument('--wav_dir', type=str, required=True, help='Path to the input wav file')
  parser.add_argument('--song_detection_json', type=str, default=None, help='Path to the song detection JSON file')
  parser.add_argument('--visualize', type=bool, default=False, help='Enable visualization')
  parser.add_argument('--apply_post_processing', type=bool, default=False, help='Apply post-processing')
  parser.add_argument('--window_size', type=int, default=50, help='Window size for label smoothing')
  args = parser.parse_args()
  classifier_path = f"experiments/{args.bird_name}_linear_decoder"
  inference_spec_dst_folder = f"imgs/decoder_specs_inference_test"
  output_path = f"files/{args.bird_name}_decoded_database.json"
  inference = TweetyBertInference(
      classifier_path,
      inference_spec_dst_folder,
      output_path,
      song_detection_json=args.song_detection_json,
      visualize=args.visualize,
      apply_post_processing=args.apply_post_processing
  )
  inference.setup_wav_to_spec(args.wav_dir)
  results = inference.process_folder(args.wav_dir)
  inference.save_results(results, output_path)