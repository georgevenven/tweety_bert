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
from post_processing import majority_vote

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
  def __init__(self, classifier_path, spec_dst_folder, output_path, song_detection_json=None, visualize=False, dump_interval=500, apply_post_processing=True, window_size=50):
      print(output_path)
      # Force CPU usage
      self.device = torch.device("cpu")
      self.classifier = self.load_decoder_state(classifier_path)
      self.wav_to_spec = None
      self.spec_dst_folder = spec_dst_folder
      self.output_path = output_path
      self.visualize = visualize
      self.dump_interval = dump_interval
      self.apply_post_processing = apply_post_processing
      self.window_size = window_size
      self.song_detection_json = song_detection_json
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
      tweety_bert_model = load_model(self.model_dir, device='cpu')
      self.create_classifier(tweety_bert_model)
      model = self.classifier_model.load_state_dict(torch.load(os.path.join(save_dir, "decoder_weights.pth"), map_location=self.device))
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
      self.wav_to_spec = WavtoSpec(folder, self.spec_dst_folder, song_detection_json_path=self.song_detection_json)

  def smooth_labels(self, labels, window_size=50):
      return majority_vote(labels, window_size=window_size)

  def process_file(self, file_path):
      # Print detection method being used
      if self.song_detection_data is None:
          print(f"Using TweetyNET detection for {os.path.basename(file_path)}")
      else:
          print(f"Using pre-designated JSON for {os.path.basename(file_path)}")

      file_creation_timestamp = get_creation_date(Path(file_path))
      creation_date_str = datetime.fromtimestamp(file_creation_timestamp).isoformat()

      # Ensure WavtoSpec instance exists
      if self.wav_to_spec is None:
           print(f"Error: WavtoSpec not initialized before processing {os.path.basename(file_path)}. Skipping.")
           # Return an empty/error structure
           return {"file_name": os.path.basename(file_path), "creation_date": creation_date_str, "song_present": False, "error": "WavtoSpec not initialized", "syllable_onsets_offsets_ms": {}, "syllable_onsets_offsets_timebins": {}}

      try:
          spec_data_tuple = self.wav_to_spec.multiprocess_process_file(
              file_path,
              song_detection_json_path=self.song_detection_json, # Pass the path here
              save_npz=False
          )
      except Exception as e:
          print(f"ERROR [{os.path.basename(file_path)}]: Exception during spec generation! Error: {e}")
          return {"file_name": os.path.basename(file_path), "creation_date": creation_date_str, "song_present": False, "error": f"Spec gen exception: {e}", "syllable_onsets_offsets_ms": {}, "syllable_onsets_offsets_timebins": {}}

      # --- Unpack the spectrogram data ---
      # Expecting (Sxx_log, results, labels) or (None, None, None)
      if spec_data_tuple is None or spec_data_tuple[0] is None:
            print(f"Spectrogram generation failed or skipped for {os.path.basename(file_path)}")
            return {
                "file_name": os.path.basename(file_path), "creation_date": creation_date_str, "song_present": False,
                "error": "Spectrogram generation failed or skipped",
                "syllable_onsets_offsets_ms": {}, "syllable_onsets_offsets_timebins": {}
            }
      # Assign the actual spectrogram array (first element of the tuple)
      spec = spec_data_tuple[0]
      # We might not need vocalization_segments_info or labels_from_specgen directly here
      # vocalization_segments_info = spec_data_tuple[1]
      # labels_from_specgen = spec_data_tuple[2]
      # --- End Unpacking ---

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
      # --- DEBUG: Check the segment being used and spec_data ---
      # Use the first segment for now, assuming single segment processing logic
      segment_to_process = vocalization_data['segments'][0]
      print(f"DEBUG [{file_name}]: Segment used for slicing: Type={type(segment_to_process)}, Value={segment_to_process}")
      print(f"DEBUG [{file_name}]: spec (spectrogram array) before slicing: Type={type(spec)}, Shape={spec.shape if isinstance(spec, np.ndarray) else 'N/A'}")
      # --- END DEBUG ---
      try:
          # Slice the unpacked 'spec' array, not the original tuple 'spec_data'
          song_spec = spec[:, segment_to_process['onset_timebin']:segment_to_process['offset_timebin']]
      except TypeError as e:
          print(f"ERROR [{file_name}]: TypeError during slicing! Error: {e}")
          print(f"  spec type: {type(spec)}") # Print type of spec array
          print(f"  segment_to_process type: {type(segment_to_process)}")
          print(f"  segment_to_process value: {segment_to_process}")
          # Handle error, e.g., return an error dict or skip
          return {"file_name": file_name, "creation_date": creation_date_str, "song_present": False, "error": f"TypeError slicing: {e}", "syllable_onsets_offsets_ms": {}, "syllable_onsets_offsets_timebins": {}}
      except IndexError as e:
           print(f"ERROR [{file_name}]: IndexError during slicing! Error: {e}")
           print(f"  spec shape: {spec.shape if isinstance(spec, np.ndarray) else 'N/A'}") # Print shape of spec array
           print(f"  Indices attempted: {segment_to_process.get('onset_timebin', 'N/A')}:{segment_to_process.get('offset_timebin', 'N/A')}")
           return {"file_name": file_name, "creation_date": creation_date_str, "song_present": False, "error": f"IndexError slicing: {e}", "syllable_onsets_offsets_ms": {}, "syllable_onsets_offsets_timebins": {}}

      spectogram, pad_amount = self.inference_data_class(song_spec)
      # Handle potential None return from inference_data_class
      if spectogram is None:
           print(f"Warning: inference_data_class returned None for {file_name}. Skipping inference.")
           return {"file_name": file_name, "creation_date": creation_date_str, "song_present": False, "error": "Data class preparation failed", "syllable_onsets_offsets_ms": {}, "syllable_onsets_offsets_timebins": {}}
      spec_tensor = torch.Tensor(spectogram).to(self.device).unsqueeze(1)
      logits = self.classifier_model(spec_tensor.permute(0,1,3,2))
      logits = logits.reshape(logits.shape[0] * logits.shape[1], -1)
      predicted_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
      if self.apply_post_processing:
          post_processed_labels = self.smooth_labels(predicted_labels, window_size=self.window_size)
          post_processed_labels[-pad_amount:] = -1
          post_processed_labels = post_processed_labels[:-pad_amount]
      else:
          post_processed_labels = predicted_labels
      onsets_offsets_ms, onsets_offsets_timebins = self.convert_to_onset_offset(post_processed_labels)
      song_present = bool(onsets_offsets_ms) # Song is present if any non -1 labels resulted in segments

      # --- Visualize ---
      if self.visualize:
          # Ensure spec is a 2D numpy array before passing
          spec_np = spec if isinstance(spec, np.ndarray) else spec.cpu().numpy()
          labels_np = post_processed_labels if isinstance(post_processed_labels, np.ndarray) else post_processed_labels.cpu().numpy()
          if spec_np.ndim == 2: # Only visualize if spec is 2D
              self.visualize_spectrogram(spec_np, labels_np, os.path.basename(file_path))
          else:
              print(f"Warning: Skipping visualization for {os.path.basename(file_path)} due to unexpected spec shape: {spec_np.shape}")
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
      # spec is now guaranteed to be a 2D numpy array [freq, time]
      # predicted_labels is a 1D numpy array [time]

      if spec.ndim != 2:
          print(f"ERROR [{file_name}]: Spectrogram passed to visualize is not 2D! Shape: {spec.shape}. Skipping visualization.")
          return

      plt.figure(figsize=(25, 10)) # Wider figure
      # Adjust GridSpec: make label plot height 1/8th of spectrogram plot height (approx)
      gs = plt.GridSpec(2, 1, height_ratios=[8, 1], hspace=0.05) # Changed height ratio

      # Plot Spectrogram
      ax1 = plt.subplot(gs[0])
      # Display with freq on y-axis, time on x-axis
      im = ax1.imshow(spec, aspect='auto', origin='lower', cmap='viridis') # Keep original orientation
      ax1.set_title(f'Spectrogram - {file_name}', fontsize=16)
      ax1.set_ylabel('Frequency Bins', fontsize=14) # Slightly larger label
      ax1.tick_params(axis='y', labelsize=10)
      ax1.tick_params(axis='x', bottom=False, labelbottom=False) # Hide x-axis ticks and labels for top plot
      # REMOVED Spectrogram colorbar
      # cbar_spec = plt.colorbar(im, ax=ax1, format='%+2.0f dB', pad=0.01)
      # cbar_spec.ax.tick_params(labelsize=10)

      # Plot Predicted Labels
      ax2 = plt.subplot(gs[1], sharex=ax1) # Share x-axis with spectrogram
      # Ensure labels are 2D for imshow: shape (1, time)
      # Adjust vmin/vmax based on actual labels present + background
      min_label_val = -1 # Assume -1 for background/noise
      max_label_val = self.num_classes - 1
      im_labels = ax2.imshow([predicted_labels], aspect='auto', origin='lower', cmap=self.cmap,
                             vmin=min_label_val, vmax=max_label_val)
      ax2.set_title('Predicted Syllable Labels', fontsize=16)
      ax2.set_xlabel('Time Bins', fontsize=14) # Slightly larger label
      ax2.set_yticks([]) # Hide y-axis ticks
      ax2.tick_params(axis='x', labelsize=10)

      # Remove the colorbar for the predicted label plot
      # (No plt.colorbar for im_labels)

      plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap
      output_path = os.path.join(self.spec_dst_folder, f"{os.path.splitext(file_name)[0]}_visualization.png")

      try:
          plt.savefig(output_path, dpi=150, bbox_inches='tight') # Lower DPI if needed
      except Exception as e:
          print(f"ERROR [{file_name}]: Failed during visualization/saving! Error: {e}") # Catch plotting/saving errors
      finally:
          plt.close() # Ensure plot is closed even if error occurs

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run TweetyBert Inference")
  parser.add_argument('--bird_name', type=str, required=True, help='Name of the bird')
  parser.add_argument('--wav_dir', type=str, required=True, help='Path to the input wav file')
  parser.add_argument('--song_detection_json', type=str, default=None, help='Path to the song detection JSON file')
  parser.add_argument('--visualize', type=lambda x: (str(x).lower() == 'true'), default=False, help='Enable visualization')
  parser.add_argument('--apply_post_processing', type=lambda x: (str(x).lower() == 'true'), default=False, help='Apply post-processing')
  parser.add_argument('--window_size', type=int, default=50, help='Window size for label smoothing if post-processing is enabled.')
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
      apply_post_processing=args.apply_post_processing,
      window_size=args.window_size
  )
  inference.setup_wav_to_spec(args.wav_dir)
  results = inference.process_folder(args.wav_dir)
  inference.save_results(results, output_path)