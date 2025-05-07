```markdown
# TweetyBERT: Automated Parsing of Birdsong Through Self-Supervised Machine Learning

This repository contains the code and instructions to replicate the results and figures presented in the paper "TweetyBERT: Automated parsing of birdsong through self-supervised machine learning."

## 🐦 TweetyBERT Overview

TweetyBERT combines a convolutional front-end with a transformer architecture to learn representations of bird vocalizations. The model can be used for:
- Automated/Unsupervised labeling of songbird syllables
- Comparing embeddings before/after perturbation
- Visualizing song with dimensionality reduction

For questions or collaboration inquiries, please email: georgev [at] Uoregon.edu

## 🔧 Repository Structure

The repository is organized as follows:

<pre>
tweety_bert/
├── readme.md                     # This file
├── pretrain.py                   # Python script for pretraining TweetyBERT
├── decoding.py                   # Python script for UMAP generation and decoder training
├── run_inference.py              # Python script for running inference
├── figure_generation_scripts/    # Scripts for generating paper figures
│   ├── dim_reduced_birdsong_plots.py # Generates UMAP plots from embeddings
│   ├── generating_inference_fig2b.py # Script for Figure 2B
│   ├── inspecting_hdbscan_labels.py  # For visualizing HDBSCAN labels on spectrograms
│   ├── linear_probe_analysis.ipynb   # Jupyter notebook for linear probe analysis and plots (Fig 6)
│   ├── masked_prediction_figure_generator.py # Generates visualizations for masked predictions (Fig 2)
│   ├── paper_loss_plots.py           # Generates training and validation loss plots
│   ├── spec_image_generator.py       # Generates spectrogram images
│   ├── umap_comparison_figure_generation.py # For seasonality analysis plots (Fig 7)
│   ├── umap_eval.py                  # Evaluates UMAP clustering and generates Fig 5D,E,F
│   ├── UMAP_plots_from_npz.py        # Generates UMAP plots from .npz files (Fig 3B,C, Fig 4A,B)
│   ├── visualizing_hdb_scan_labels.py # Visualizes HDBSCAN labels (Fig 4C,D,E, Fig 5A,B,C)
│   └── visualizing_song_cluster_phase.py # Interactive UMAP region selection (Fig 3A)
├── scripts/                      # Helper scripts and utilities for one-off tasks
│   ├── convert_yarden_to_phrase_labels.py # Converts Yarden annotations to phrase labels for JSON
│   ├── fold_v_measure_calculation.py # Calculates V-Measure for UMAP folds (Fig 4A,B)
│   ├── Layer_Wise_FER.py             # Performs layer-wise Frame Error Rate analysis
│   ├── layer_wise_v_measure.py       # Calculates V-Measure layer by layer
│   ├── linear_probe_automated_analysis.py # Automates linear probe analysis (Fig 6 data)
│   ├── merge_output_label_jsons.py   # Merges multiple JSON label files into one
│   ├── move_non_zero_files.py        # Moves WAV files that contain audio above a threshold
│   ├── multiple_dir_to_single_folder.py # Combines files from multiple source directories into one
│   ├── plot_all_umaps.py             # Generates UMAPs for multiple layers of the model
│   ├── plot_num_songs_detected.py    # Plots the number of songs detected per bird/day from a JSON
│   ├── remove_empty_yarden_data.py   # Removes Yarden data files with no labels
│   ├── segmenting_script.py          # Interactive tool for thresholding and segmenting spectrograms (PyQtGraph)
│   ├── split_based_on_canary_date.py # Splits files into groups based on date relative to an experiment date
│   ├── split_files_into_train_and_test.py # Splits files into training and testing sets
│   ├── tape_archive_extractor.py     # Extracts data from tape archive .raw files to .wav
│   ├── temp_process_aws_data_formats.ipynb # Jupyter notebook for processing AWS data formats
│   ├── vast_ai_unzipper.py           # Utility for unzipping files on Vast.ai
│   └── whisperseg.py                 # Processes WAV files using WhisperSeg for vocalization detection
├── shell_scripts/                # Shell scripts for automation (Alternative workflow)
│   ├── batch_decoding.sh             # Script for batch processing of multiple birds (UMAP, Decoder Training, Inference)
│   ├── fold_umap_generation.sh       # Generates UMAPs for different data folds
│   ├── inference.sh                  # Runs inference on new WAV files
│   ├── llb_umap_generations.sh       # Generates UMAPs for specific LLB birds
│   ├── pretrain.sh                   # Script for pretraining TweetyBERT
│   ├── train_decoder.sh              # Script for training a decoder for a single bird
│   └── train_decoder_multiple_dir.sh # Script for training a decoder using data from multiple directories
├── src/                          # Core model implementation and primary codebase
│   ├── __init__.py                   # Makes src a Python package
│   ├── analysis.py                   # Contains functions for UMAP plotting and performance metrics
│   ├── data_class.py                 # Defines Dataset and Dataloader classes
│   ├── decoder.py                    # Handles decoder training and saving
│   ├── experiment_manager.py         # Manages experiment configurations and runs
│   ├── inference.py                  # Script for running inference with a trained model
│   ├── linear_probe.py               # Implements linear probe model and trainer
│   ├── model.py                      # Defines the TweetyBERT model architecture
│   ├── spectogram_generator.py       # Generates spectrograms from WAV files
│   ├── trainer.py                    # Handles model training loop and metrics
│   ├── TweetyBERT.py                 # Main script for running TweetyBERT experiments
│   ├── TweetyNET.py                  # Implementation of TweetyNET model (likely for song detection)
│   └── utils.py                      # Utility functions (e.g., loading models, configs)
├── temp.ipynb                    # Temporary Jupyter notebook
├── test.py                       # Test script, e.g., for loading and inspecting .npz files
├── files/                        # Stores NPZ files and JSON annotation databases [User must populate or adjust paths in scripts]
│   ├── LLB3_Untrained.npz            # Example NPZ file for UMAP plots from npz script
│   ├── category_colors_llb3.pkl      # Example category colors file
│   └── llb3_predictions_for_fig2b.npz # NPZ file for generating Figure 2b
├── experiments/                  # Stores model checkpoints and training logs [User must populate or adjust paths in scripts]
│   └── TweetyBERT_Paper_Yarden_Model/  # Example model directory for linear probe & pretraining
│       ├── train_files.txt             # List of training files
│       └── test_files.txt              # List of test files
├── imgs/                         # Stores generated images, plots, and visualizations [Output directory]
│   ├── umap_plots/                   # Output directory for UMAP_plots_from_npz.py
│   ├── verifying_ground_truth_labels/ # Output for visualizing_ground_truth_labels.py
│   └── seasonality_analysis/         # Output directory for umap_comparison_figure_generation.py
├── results/                      # Stores output data from model computations and analysis [Output directory]
│   └── proxy_metrics/                # Stores results from umap_eval.py (Fig 5D,E,F)
│       ├── all_windows_summary.txt     # Summary of stats for each smoothing window
│       └── metrics_by_window.png       # Plot for Figure 5F
└── LLB_Embedding/                # Example directory containing NPZ files for Figure 3 [User must populate]
└── LLB_Fold_Data/                # Example directory containing NPZ files for Figure 4A,B V-Measure calculation [User must populate]
</pre>

## 🚀 Installation & Environment Setup

The following steps assume you have Conda installed and are using a CUDA-capable GPU (e.g., NVIDIA RTX 4090). Adjust as necessary for your system.

```bash
# 1. Create and activate a new Conda environment
conda create -n tweetybert python=3.11
conda activate tweetybert

# 2. Install core scientific packages (including librosa)
conda install -c conda-forge \
    numpy \
    matplotlib \
    tqdm \
    umap-learn \
    hdbscan \
    scikit-learn \
    pandas \
    seaborn \
    jupyter \
    ipykernel \
    librosa

# 3. Install additional dependencies via pip
pip install soundfile shutil-extra glasbey pyqtgraph PyQt5 hmmlearn

# 4. (Optional) Install PyTorch if not already installed (adjust CUDA version if needed)
# Example for CUDA 12.x:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
# For other versions, visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

# 5. Clone this repository
git clone [https://github.com/georgevenven/tweety_bert.git](https://github.com/georgevenven/tweety_bert.git) # Replace with your actual repo URL if different
cd tweety_bert
```

**Important Notes:**
* Ensure you have Python 3.11+ and PyTorch >= 2.0 installed.
* The primary workflow now uses the Python scripts (`pretrain.py`, `decoding.py`, `run_inference.py`) located in the root directory.
* The shell scripts in `shell_scripts/` provide an alternative workflow.

## 💾 Storage Requirements

Depending on the size of your audio dataset, storage requirements can range from **50 GB** to **1 TB** or more. Ensure you have sufficient disk space.

## ⚡ GPU & Training Times

Pretraining can take several hours to days on a single NVIDIA RTX 4090 GPU, depending on your dataset size and hyperparameters.

## 🎶 Song Detection & JSON Format

TweetyBERT uses a JSON file to identify segments of bird song within audio recordings. This file can also optionally include syllable labels for validation.

**Example JSON Structure:**
```json
{
  "filename": "bird_XXXX_YYYY_MM_DD_HH_MM_SS.wav",
  "song_present": true,
  "segments": [
    {
      "onset_timebin": 100,
      "offset_timebin": 500,
      "onset_ms": 1234.56,
      "offset_ms": 5678.90
    }
  ],
  "spec_parameters": {
    "step_size": 119,
    "nfft": 1024
  },
  "syllable_labels": {
    "1": [
      [1.00, 2.50]
    ]
  }
}
```
* **`filename`**: Name of the WAV file.
* **`song_present`**: Boolean indicating if song is detected.
* **`segments`**: List of detected song segments with onset/offset times (in timebins and milliseconds).
* **`spec_parameters`**: Parameters used for spectrogram generation (e.g., `step_size`, `nfft`).
* **`syllable_labels` (optional)**: Time intervals for each labeled syllable, keyed by label ID.

Typically, a separate song detection tool (like the one potentially in `src/TweetyNET.py` or `scripts/whisperseg.py`) is used to generate this JSON.

## 🗄️ NPZ File Format

The `.npz` files used for embeddings and analysis generally contain the following arrays:

| Array Name            | Example Shape        | Data Type   | Description                                                                 |
| :-------------------- | :------------------- | :---------- | :-------------------------------------------------------------------------- |
| `embedding_outputs`   | `(N, 2)`             | `float32`   | 2D UMAP embedding coordinates for N timebins.                               |
| `hdbscan_labels`      | `(N,)`               | `int64`     | Cluster labels assigned by HDBSCAN for each timebin (-1 for noise).         |
| `ground_truth_labels` | `(N,)`               | `int64`     | Human-annotated syllable labels for each timebin.                           |
| `predictions`         | `(N, 196)`           | `float32`   | Raw output (neural activations) from a TweetyBERT layer before UMAP.        |
| `s`                   | `(N, 196)`           | `float32`   | Spectrogram data for the N timebins (frequency bins = 196).                 |
| `hdbscan_colors`      | `(C_hdbscan, 3)`     | `float64`   | RGB color values for each HDBSCAN cluster.                                  |
| `ground_truth_colors` | `(C_gt, 3)`          | `float64`   | RGB color values for each ground truth syllable class.                      |
| `original_spectogram` | `(N, 196)`           | `float32`   | Original full spectrogram corresponding to the N timebins.                  |
| `vocalization`        | `(N,)`               | `int64`     | Binary array indicating if a timebin contains vocalization (1) or not (0). |
| `file_indices`        | `(N,)`               | `int64`     | Index mapping each timebin to its original source file in `file_map`.       |
| `dataset_indices`     | `(N,)`               | `int64`     | Index indicating which dataset or group a timebin belongs to (e.g., for seasonality). |
| `file_map`            | `()` (scalar object) | `object`    | A dictionary mapping integer file indices to actual file path strings.      |

*N = Total number of timebins in the NPZ file (e.g., 1,001,549 in the example).*
*C_hdbscan = Number of unique HDBSCAN clusters.*
*C_gt = Number of unique ground truth syllable classes.*

## 🏋️ Training the Model (Pretraining)

To pretrain TweetyBERT using the Python script:
1.  Navigate to the root directory of the repository (`tweety_bert/`).
2.  Run `pretrain.py` with appropriate arguments. Key arguments include:
    * `--input_dir`: Path to the folder containing WAV files.
    * `--song_detection_json_path`: Path to the song detection JSON (optional, uses internal detection if not provided).
    * `--experiment_name`: Name for your training run (e.g., "MyTweetyBERTModel"). Results saved to `experiments/<experiment_name>`.
    * `--test_percentage`: Percentage of data for the test set (default: 20).
    * `--batch_size`, `--learning_rate`, `--context`, `--m`, etc.: Model and training hyperparameters. Use `--help` to see all options.

**Example:**
```bash
python pretrain.py \
    --input_dir "/path/to/your/wav/files" \
    --song_detection_json_path "/path/to/your/song_detection.json" \
    --experiment_name "MyTweetyBERTModel" \
    --test_percentage 20 \
    --batch_size 42 \
    --learning_rate 3e-4 \
    --context 250 \
    --m 100 \
    --multi_thread # Add this flag to use multi-threading for spec gen
```

## 💡 Generating Embeddings & Training a Decoder

After pretraining, use `decoding.py` to generate UMAP embeddings and train a decoder.
1.  Navigate to the root directory (`tweety_bert/`).
2.  Run `decoding.py`. This script has two main modes controlled by `--mode`:
    * **`--mode single`**: Processes a single dataset (potentially a random subset).
    * **`--mode grouping`**: Processes data split into temporal groups (relies on `scripts/copy_files_from_wavdir_to_multiple_event_dirs.py` for grouping logic).
3.  Key arguments:
    * `--mode`: `single` or `grouping`.
    * `--bird_name`: Descriptive name for UMAP/decoder outputs (e.g., "my_canary").
    * `--model_name`: Name of the pretrained experiment (must match `experiment_name` used in `pretrain.py`).
    * `--wav_folder`: Path to the bird's/dataset's WAV files.
    * `--song_detection_json_path`: Path to the detection JSON for these files.
    * `--num_samples_umap`: Number of samples for UMAP (e.g., "5e5").
    * `--num_random_files_spec` (for `--mode single`): Number of random WAVs to use for spectrogram generation.

**Example (`single` mode):**
```bash
python decoding.py \
    --mode single \
    --bird_name "my_canary_decoder" \
    --model_name "MyTweetyBERTModel" \
    --wav_folder "/path/to/this_birds/wav/files" \
    --song_detection_json_path "/path/to/this_birds/song_detection.json" \
    --num_random_files_spec 100 \
    --num_samples_umap 5e5
```

**Example (`grouping` mode):**
```bash
python decoding.py \
    --mode grouping \
    --bird_name "my_canary_grouped_decoder" \
    --model_name "MyTweetyBERTModel" \
    --wav_folder "/path/to/this_birds/wav/files" \
    --song_detection_json_path "/path/to/this_birds/song_detection.json" \
    --num_samples_umap 1e5
```
*(Note: The grouping mode currently relies on the interaction with `scripts/copy_files_from_wavdir_to_multiple_event_dirs.py`, which might require specific setup or be interactive).*

## 🔍 Inference

Run inference on new WAV files using a trained decoder.
1.  Navigate to the root directory (`tweety_bert/`).
2.  Run `run_inference.py`. Key arguments:
    * `--bird_name`: Name used when training the decoder (used to find the saved decoder state, e.g., "my_canary_decoder").
    * `--wav_folder`: Directory of new WAV files for inference.
    * `--song_detection_json`: Path to the detection JSON for these new files (optional, uses internal detection if not provided).
    * `--apply_post_processing`: Apply smoothing (True/False, default True).
    * `--window_size`: Smoothing window size (default 200).
    * `--visualize`: Generate output plots (True/False, default False).

**Example:**
```bash
python run_inference.py \
    --bird_name "my_canary_decoder" \
    --wav_folder "/path/to/new/wav/files" \
    --song_detection_json "/path/to/new/song_detection.json" \
    --apply_post_processing True \
    --visualize True
```
The output will be a JSON database (`files/<bird_name>_decoded_database.json`) summarizing detected syllables. Visualizations (if enabled) are saved in `imgs/inference_specs_<bird_name>/`.

## 📄 Regenerating Figures from the Paper

The following instructions outline how to regenerate the figures presented in the paper.
**Note:** You will need to adjust file paths within the scripts to point to your local data locations. The original paths mentioned (e.g., `/media/george-vengrovski/...`) are specific to the original development environment. Placeholder paths like `[Path_to_your_data_here]` should be replaced.

**General Setup:**
1.  Ensure your Conda environment (`tweetybert`) is activated.
2.  Navigate to the root of the cloned `tweety_bert` repository.
3.  Organize your data files (`.npz`, `.wav`, `.json`) as referenced by the scripts, or update the paths within the scripts accordingly.

---

**Figures 1 & 2:**
These are cartoon schematics, and their direct replication from code is not applicable. Figure 2's masked prediction visualizations can be conceptually generated using `figure_generation_scripts/masked_prediction_figure_generator.py`.
    - **To generate similar masked prediction examples:**
      1. Edit `figure_generation_scripts/masked_prediction_figure_generator.py`:
         - Set `MODEL_DIR` to your trained model directory (e.g., `"experiments/TweetyBERT_Paper_Yarden_Model"`).
         - Set `DATA_DIR` to the directory containing spectrogram NPZ files for visualization (e.g., `"[Path_to_your_data_here]/llb3_specs"`).
         - Set `OUTPUT_DIR` to where visualizations will be saved (e.g., `"imgs/masked_predictions_for_figure_2"`).
      2. Run the script from the `tweety_bert` root directory:
         ```bash
         python figure_generation_scripts/masked_prediction_figure_generator.py
         ```

---

**Figure 3: TweetyBERT and Spectrogram UMAP Embeddings**

* **Data:** Prepare NPZ files containing TweetyBERT embeddings and raw spectrogram embeddings, along with ground truth labels. Assume you place them in `files/LLB_Embedding_Paper/`.
* **Figure 3B & 3C (UMAP plots):**
    1.  Edit `figure_generation_scripts/UMAP_plots_from_npz.py`:
        * Set `input_path` to the path of an NPZ file (e.g., `"files/LLB_Embedding_Paper/Your_Bird_Embeddings.npz"`).
        * Set `output_dir` (e.g., `"imgs/umap_plots_fig3"`).
    2.  Run from the `tweety_bert` root directory:
        ```bash
        python figure_generation_scripts/UMAP_plots_from_npz.py
        ```
* **Figure 3A (Interactive UMAP region visualization):**
    1.  Edit `figure_generation_scripts/visualizing_song_cluster_phase.py`:
        * Set `file_path` to the path of an NPZ file (e.g., `"files/LLB_Embedding_Paper/Your_Bird_Embeddings.npz"`).
    2.  Run from the `tweety_bert` root directory:
        ```bash
        python figure_generation_scripts/visualizing_song_cluster_phase.py
        ```
        This script is interactive and requires a GUI; use the Lasso tool to select a UMAP region. Saved images appear in `imgs/selected_regions/`.

---

**Figure 4: Machine-derived vs. Human-derived Clusters**

* **Data:** Prepare NPZ files from UMAP folds (e.g., in `files/LLB_Fold_Data_Paper/`). Each file should contain embeddings and labels for a data fold.
* **Figure 4A & 4B (V-Measure Calculation and UMAP Plots):**
    1.  **Calculate V-Measure scores:**
        * Edit `scripts/fold_v_measure_calculation.py`:
            * Set `npz_directory` to your folder of UMAP fold data (e.g., `"files/LLB_Fold_Data_Paper/"`).
        * Run from the `tweety_bert` root directory:
            ```bash
            python scripts/fold_v_measure_calculation.py
            ```
            This will print V-measure scores for each fold.
    2.  **Generate UMAP plots** (will plot both syllable and phrase labels for comparison):
        * Edit `figure_generation_scripts/UMAP_plots_from_npz.py`:
            * Set `input_path` to one of the NPZ files from your fold data (e.g., `"files/LLB_Fold_Data_Paper/fold1.npz"`).
            * Set `output_dir` (e.g., `"imgs/umap_plots_fig4"`).
        * Run from the `tweety_bert` root directory:
            ```bash
            python figure_generation_scripts/UMAP_plots_from_npz.py
            ```
* **Figure 4C, 4D, 4E (Spectrograms with HDBSCAN and Ground Truth Labels):**
    1.  Edit `figure_generation_scripts/visualizing_hdb_scan_labels.py`:
        * Set `file_path` to an NPZ file from your UMAP fold data (e.g., `"files/LLB_Fold_Data_Paper/fold1.npz"`).
        * Adjust `segment_length` and `start_idx` or rely on the random selection. The script saves to `output_dir = "imgs/all_spec_plus_labels"`.
    2.  Run from the `tweety_bert` root directory:
        ```bash
        python figure_generation_scripts/visualizing_hdb_scan_labels.py
        ```
        This script generates many spectrogram fragments. You will need to manually select one that clearly shows interesting phrases and mostly aligned labels, similar to the paper's figure.

---

**Figure 5: Comparing Human and Automated Labels for Sequence Analysis**

* **Figure 5A, 5B, 5C (Spectrogram with Spurious Insertions):**
    Generated similarly to Figure 4C,D,E using `figure_generation_scripts/visualizing_hdb_scan_labels.py`. You'll need to manually find/select a sample NPZ file and segment that exhibits significant spurious insertions.
* **Figure 5D, 5E, 5F (UMAP Evaluation and Smoothing Window Analysis):**
    1.  Edit `figure_generation_scripts/umap_eval.py`:
        * Set `folder_path` to your directory containing UMAP fold data (e.g., `"[Path_to_your_data_here]/LLB_Fold_Data"`).
        * Set `labels_path` argument in `ComputerClusterPerformance` calls if it differs from `folder_path`.
        * Set `output_dir` (e.g., `"results/proxy_metrics_fig5"`).
    2.  Run from the `tweety_bert` root directory:
        ```bash
        python figure_generation_scripts/umap_eval.py
        ```
    * This script will generate:
        * `all_windows_summary.txt` in the `output_dir`, containing statistics for each smoothing window and identifying the optimal one.
        * `metrics_by_window.png` (used for Fig 5F) in the `output_dir`.
        * Plots for Fig 5D and 5E (normalized confusion matrices) will be in the `output_dir/best_window/` subdirectory (e.g., `0X_M_norm_diag.png` and `0X_M_norm_fullreorder.png`).

---

**Figure 6: Evaluating TweetyBERT Embeddings Using Linear Probes**

* **Data Generation:**
    1.  Edit `scripts/linear_probe_automated_analysis.py`:
        * Ensure the list `experiment_configs` has the correct `experiment_path` for your pretrained TweetyBERT model (e.g., `"experiments/TweetyBERT_Paper_Yarden_Model"`).
        * Update `train_dir` and `test_dir` within `experiment_configs` to point to your linear probe datasets (e.g., for llb3, llb11, llb16). Adjust paths like `"/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/{dataset}_train"`.
        * `results_path` is set to `"results"`.
    2.  Run from the `tweety_bert` root directory:
        ```bash
        python scripts/linear_probe_automated_analysis.py
        ```
        This will create subdirectories in `results/` like `TweetyBERT_linear_probe_llb3/`, etc., containing `results.json`.
* **Plot Generation:**
    1.  Open the Jupyter Notebook: `figure_generation_scripts/linear_probe_analysis.ipynb`.
    2.  In the first cell, update `base_path` to point to the parent directory of the results generated above (e.g., `base_path = 'results'`).
    3.  Run the first cell of the notebook.

---

**Figure 7: Seasonal Vocal Plasticity in Canaries**

* **Data:** You need NPZ files containing embeddings for different birds across breeding and non-breeding seasons. Place your equivalent files (e.g., `5494_Seasonality_Final.npz`, `5508_Seasonality_Final.npz`) in a directory like `files/seasonality_embeddings/`.
* **Plot Generation:**
    1.  Edit `figure_generation_scripts/umap_comparison_figure_generation.py`:
        * In the `if __name__ == "__main__":` block, update the `npz_files` list to point to your two seasonality embedding NPZ files (e.g., `["files/seasonality_embeddings/5494_Seasonality_Final.npz", "files/seasonality_embeddings/5508_Seasonality_Final.npz"]`).
        * Set `save_dir` to your desired output directory (e.g., `"imgs/seasonality_analysis_fig7"`).
    2.  Run from the `tweety_bert` root directory:
        ```bash
        python figure_generation_scripts/umap_comparison_figure_generation.py
        ```
    * The script will generate various plots in `save_dir/bird_{bird_id}/`. The specific PNG files used for the paper are titled `bird_{bird_id}_all_before_vs_all_after_overlap.png`.

---