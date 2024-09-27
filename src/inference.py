import os
import json
import torch
from tqdm import tqdm
from torch.nn import functional as F
from spectogram_generator import WavtoSpec
from train_decoder import TweetyBertClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

class TweetyBertInference:
    def __init__(self, classifier_path, spec_dst_folder, output_path, song_detection_json=None, visualize=False, dump_interval=1, apply_post_processing=True):
        self.classifier = TweetyBertClassifier.load_decoder_state(classifier_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.cmap = mcolors.ListedColormap(colors[:self.classifier.num_classes])

    def setup_wav_to_spec(self, folder, csv_file_dir=None):
        self.wav_to_spec = WavtoSpec(folder, self.spec_dst_folder, csv_file_dir)

    def process_file(self, file_path):
        spec, vocalization, labels = self.wav_to_spec.process_file(self.wav_to_spec, file_path=file_path)

        if spec is None:
            return {
                "file_name": os.path.basename(file_path),
                "song_present": False,
                "syllable_onsets_offsets_ms": {},
                "syllable_onsets_offsets_timebins": {}
            }

        if self.song_detection_data is None:
            vocalization_data = tweety_net_detector_inference(input_file=file_path)

            if vocalization_data['segments'] == []:
                return {
                    "file_name": os.path.basename(file_path),
                    "song_present": False, 
                    "syllable_onsets_offsets_ms": {},
                    "syllable_onsets_offsets_timebins": {}
                }
        else:
            file_name = os.path.basename(file_path)
            vocalization_data = next((item for item in self.song_detection_data if item["filename"] == file_name), None)

            if vocalization_data is None:
                vocalization_data = tweety_net_detector_inference(input_file=file_path)

            if vocalization_data is not None and not vocalization_data["song_present"]:
                return {
                    "file_name": file_name,
                    "song_present": False,
                    "syllable_onsets_offsets_ms": {},
                    "syllable_onsets_offsets_timebins": {}
                }

        vocalization_data = vocalization_data['segments'][0]
        song_spec = spec[:, vocalization_data['onset_timebin']:vocalization_data['offset_timebin']]

        spectogram, pad_amount = self.inference_data_class(song_spec)
        spec_tensor = torch.Tensor(spectogram).to(self.device).unsqueeze(1)
        logits = self.classifier.classifier_model(spec_tensor.permute(0,1,3,2))
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
            "song_present": song_present,
            "syllable_onsets_offsets_ms": onsets_offsets_ms,
            "syllable_onsets_offsets_timebins": onsets_offsets_timebins
        }

    def process_folder(self, folder_path):
        results = []
        file_count = 0
        processed_files = set()

        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                results = json.load(f)
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
            "classifier_path": self.classifier.model_dir,
            "spec_dst_folder": self.spec_dst_folder,
            "output_path": self.output_path,
            "visualize": self.visualize,
            "dump_interval": self.dump_interval,
            "apply_post_processing": self.apply_post_processing
        }
        with open(output_path, 'w') as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TweetyBert Inference")
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--song_detection_json', type=str, default=None, help='Path to the song detection JSON file')
    parser.add_argument('--visualize', type=bool, default=True, help='Enable visualization')
    parser.add_argument('--apply_post_processing', type=bool, default=False, help='Apply post-processing')

    args = parser.parse_args()

    classifier_path = f"../experiments/{args.experiment_name}/linear_decoder"
    inference_spec_dst_folder = f"../experiments/{args.experiment_name}/inference_specs"
    output_path = f"../experiments/{args.experiment_name}/decoder_test_database.json"

    inference = TweetyBertInference(
        classifier_path, 
        inference_spec_dst_folder, 
        output_path, 
        song_detection_json=args.song_detection_json,  
        visualize=args.visualize,  
        apply_post_processing=args.apply_post_processing
    )
    
    inference.setup_wav_to_spec(args.folder_path)
    results = inference.process_folder(args.folder_path)
    inference.save_results(results, output_path)
