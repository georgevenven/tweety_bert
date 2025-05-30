# TweetyBERT: Automated Parsing of Birdsong Through Self-Supervised Machine Learning

This repository contains the code and instructions to replicate the results and figures presented in the paper [TweetyBERT: Automated parsing of birdsong through self-supervised machine learning (bioRxiv preprint)](https://www.biorxiv.org/content/10.1101/2025.04.09.648029v1)

## 🐦 TweetyBERT Overview

TweetyBERT combines a convolutional front-end with a transformer architecture to learn representations of bird vocalizations. The model can be used for:
- Automated/Unsupervised labeling of songbird syllables
- Comparing embeddings before/after perturbation
- Visualizing song with dimensionality reduction

For questions or collaboration inquiries, please email: georgev [at] Uoregon.edu

## 🔧 Repository Structure

The repository is organized as follows:

```
tweety_bert/
├── readme.md                     # This file
├── pretrain.py                   # Python script for pretraining TweetyBERT
├── decoding.py                   # Python script for UMAP generation and decoder training
├── run_inference.py              # Python script for running inference
├── figure_generation_scripts/    # Scripts for generating paper figures
├── scripts/                      # Helper scripts and utilities for data processing and analysis
├── shell_scripts/                # Shell scripts for automation (Alternative workflow / deprecated)
├── src/                          # Core model implementation and primary codebase
│   ├── model.py                      # Defines the TweetyBERT model architecture
│   ├── spectogram_generator.py       # Generates spectrograms from WAV files
│   ├── trainer.py                    # Handles model pretraining loop and metrics
│   ├── decoder.py                    # Handles decoder training and saving
│   ├── inference.py                  # Core script for running inference with a trained model
│   ├── linear_probe.py               # Implements linear probe model and trainer
│   ├── analysis.py                   # Contains functions for UMAP plotting and performance metrics
│   ├── data_class.py                 # Defines Dataset and Dataloader classes
│   └── utils.py                      # Utility functions (e.g., loading models, configs)
├── files/                        # Stores NPZ files and JSON annotation databases [User must populate or adjust paths]
├── experiments/                  # Stores model checkpoints and training logs [User must populate or adjust paths]
├── imgs/                         # Stores generated images, plots, and visualizations [Output directory]
└── results/                      # Stores output data from model computations and analysis [Output directory]
├── detect_song.py                # Python script for running song detection
```
* **Root Directory:** Contains the main workflow scripts (`pretrain.py`, `decoding.py`, `run_inference.py`) and this README.
* **`figure_generation_scripts/`**: Contains Python scripts specifically designed to reproduce the figures shown in the associated publication. Edit paths within these scripts as needed.
* **`scripts/`**: A collection of utility Python scripts for various tasks like data conversion, splitting, merging, plotting specific metrics, etc.
* **`shell_scripts/`**: Contains the original bash scripts for running workflows (now largely superseded by the root Python scripts). Kept for reference or alternative use cases.
* **`src/`**: Holds the core Python source code for the TweetyBERT model, data handling, training, inference logic, and analysis functions.
* **`files/`**: Intended location for input data like annotation files (`.json`) and embedding files (`.npz`). You will need to place your data here or modify paths in the scripts.
* **`experiments/`**: Default location where trained model checkpoints (`.pth`), configuration files (`config.json`), and training logs (`training_statistics.json`, `train_files.txt`, `test_files.txt`) are saved.
* **`imgs/`**: Default output directory for generated images, such as UMAP plots, spectrogram visualizations, and other figures.
* **`results/`**: Default output directory for non-image results, like performance metrics (`.txt`, `.csv`).
* **`detect_song.py`**: Python script for running song detection.

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
#
# =====================
# ⚠️ WARNING: Be VERY careful to match the correct CUDA version to your system and GPU drivers! ⚠️
#   - If you do NOT have an NVIDIA GPU or do not need GPU acceleration, install the CPU-only version:
#       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#   - If you have a CUDA-capable GPU, install the version matching your CUDA toolkit (see https://pytorch.org/get-started/locally/)
#   - Mismatched CUDA versions can cause import errors or silent failures.
# =====================
# Example for CUDA 12.x:
pip install torch torchvision torchaudio

# 5. Clone this repository
git clone https://github.com/georgevenven/tweety_bert.git

cd tweety_bert
```

**Important Notes:**
* Ensure you have Python 3.11+ and PyTorch >= 2.0 installed.
* The primary workflow now uses the Python scripts (`pretrain.py`, `decoding.py`, `run_inference.py`) located in the root directory.
* The shell scripts in `shell_scripts/` provide an alternative workflow.

## 💾 Storage Requirements

Depending on the size of your audio dataset, storage requirements can range from **50 GB** to **1 TB** or more. Ensure you have sufficient disk space.

## ⚡ GPU & Training Times

Pretraining can take 3-4 hours on a single NVIDIA RTX 4090 GPU, and 100s of hours on a CPU, depending on your dataset size and hyperparameters.
The speed of training is highly dependent on disk I/O operations; for best performance, keep your data and this repository on the fastest available drive (preferably NVMe or SATA SSD).

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
* **(Important):** Even if `song_present` is true, the file might be skipped later if the `segments` list is empty or contains only very short segments.
* **`segments`**: List of detected song segments with onset/offset times (in timebins and milliseconds).
* **`spec_parameters`**: Parameters used for spectrogram generation (e.g., `step_size`, `nfft`).
* **`syllable_labels` (optional)**: Time intervals for each labeled syllable, keyed by label ID.

### Generating the Song Detection JSON (Recommended)

This repository includes a wrapper script `detect_song.py` to simplify the process of generating the required JSON file using the [TweetyNet Song Detector](https://github.com/georgevenven/tweety_net_song_detector).

1.  **Prerequisite:** Ensure `git` is installed on your system.
2.  **Navigate:** Go to the root directory of the `tweety_bert` repository.
3.  **Run:** Execute the `detect_song.py` script, providing the path to your WAV files.

**Example:**
```bash
python detect_song.py --input_dir "/path/to/your/wav/files"
```

* The first time you run it, the script will automatically clone the detector repository into a local folder named `song_detector`.
* It will then process all `.wav` files found in the specified `--input_dir` (including subdirectories).
* The output is a single JSON file (defaulting to `files/song_detection.json`) containing entries for each processed WAV file, indicating detected song segments.
* **Use the path to this generated JSON file** for the `--song_detection_json_path` argument when running `pretrain.py`, `decoding.py`, or `run_inference.py`.


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
    --context 1000 \
    --m 250 \
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
*(Note: The grouping mode currently relies on the interaction with `scripts/copy_files_from_wavdir_to_multiple_event_dirs.py`, which is interactive).*

## 🔍 Inference (Not Part of the TweetyBERT Paper)

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

*N = Total number of timebins in the NPZ file.*
*C_hdbscan = Number of unique HDBSCAN clusters.*
*C_gt = Number of unique ground truth syllable classes.*

## 📄 Regenerating Figures from the Paper

The following instructions outline how to regenerate the figures presented in the paper.
**Note:** You will need to adjust file paths within the scripts to point to your local data locations. 

**General Setup:**
1.  Ensure your Conda environment (`tweetybert`) is activated.
2.  Navigate to the root of the cloned `tweety_bert` repository.
3.  Organize your data files (`.npz`, `.wav`, `.json`) as referenced by the scripts, or update the paths within the scripts accordingly.

---

**Figures 1 & 2:**
These are cartoon schematics, and their direct replication from code is not applicable. Figure 2's masked prediction visualizations can be conceptually generated using `figure_generation_scripts/masked_prediction_figure_generator.py`.
<!-- - **To generate similar masked prediction examples:**
      1. Run the `masked_prediction_figure_generator.py` script from the `tweety_bert` root directory. Provide the paths to your trained model directory, the directory containing spectrogram NPZ files for visualization, and the desired output directory using command-line arguments.

         ```bash
         python figure_generation_scripts/masked_prediction_figure_generator.py \
             --model-dir experiments/TweetyBERT_Paper_Yarden_Model \
             --data-dir [Path_to_your_data_here]/llb3_specs \
             --output-dir imgs/masked_predictions_for_figure_2 \
             --num-samples 10 # Optional: specify number of samples
         ```

      2. The script will generate visualization images in the specified output directory. -->

---

**Figure 3: TweetyBERT and Spectrogram UMAP Embeddings**

* **Data:** Prepare NPZ files containing TweetyBERT embeddings and raw spectrogram embeddings, along with ground truth labels. Assume you place them in `files/LLB_Embedding_Paper/`.
* **Figure 3B & 3C (UMAP plots):**
    1. Run from the `tweety_bert` root directory, providing the input NPZ file and output directory as arguments:
        ```bash
        python figure_generation_scripts/UMAP_plots_from_npz.py \
            "placeholder_input_npz_dir" \
            "placeholder_output_dir"
        ```
       *(This script may open an interactive window for cropping if processing a single file.)*
* **Figure 3A (Interactive UMAP region visualization):**
    1. Run from the `tweety_bert` root directory, providing the input NPZ file as an argument:
        ```bash
        python figure_generation_scripts/visualizing_song_cluster_phase.py \
            "placeholder_embedding_npz_path"
        ```
       This script requires a GUI; use the Lasso tool in the interactive plot to select a UMAP region. Saved images appear in `imgs/selected_regions/`.
       *(Optional arguments like `--collage_mode`, `--max_length`, `--used_group_coloring` can be added.)*

---

**Figure 4: Machine-derived vs. Human-derived Clusters**

* **Data:** Prepare NPZ files from UMAP folds (e.g., in `files/LLB_Fold_Data_Paper/`). Each file should contain embeddings and labels for a data fold.
* **Figure 4A & 4B (V-Measure Calculation and UMAP Plots):**
    1.  **Calculate V-Measure scores:**
        * Run from the `tweety_bert` root directory, providing the path to your fold data:
            ```bash
            python scripts/fold_v_measure_calculation.py "placeholder_dir_for_folds"
            ```
            This will print V-measure scores for each fold[cite: 113].
    2.  **Generate UMAP plots** (will plot both syllable and phrase labels for comparison):
        * Run from the `tweety_bert` root directory, providing the input NPZ file and output directory:
            ```bash
            python figure_generation_scripts/UMAP_plots_from_npz.py \
                "placeholder_input_npz_dir" \
                "placeholder_output_dir"
            ```
           *(This script may open an interactive window for cropping if processing a single file, you can close it)*.

* **Figure 4C, 4D, 4E (Spectrograms with HDBSCAN and Ground Truth Labels):**
    1. Run from the `tweety_bert` root directory, providing the path to an NPZ file from your fold data:
        ```bash
        # Example: Generate 10 random segments of default length (1000)
        python figure_generation_scripts/visualizing_hdb_scan_labels.py \
            --file_path "npz_file_path" # from which npz file you would like to generate examples \ 
            --output_dir "imgs/specs_plus_labels"

        ```
       This script generates spectrogram fragments (defaulting to 100 random ones if `--start_idx` is not provided). You will need to manually select one that clearly shows interesting phrases and mostly aligned labels, similar to the paper's figure. 
       `--output_dir`.

---

**Figure 5: Comparing Human and Automated Labels for Sequence Analysis**

* **Figure 5A, 5B, 5C (Spectrogram with Spurious Insertions):**
    Generated similarly to Figure 4C,D,E using `figure_generation_scripts/visualizing_hdb_scan_labels.py`. You'll need to manually find/select a sample NPZ file and segment that exhibits significant spurious insertions by running the script with appropriate arguments.
    Example:
    ```bash
    # add no smoothing parameter
    python figure_generation_scripts/visualizing_hdb_scan_labels.py \
        --file_path "npz_file_path" # from which npz file you would like to generate examples \ 
        --output_dir "imgs/specs_plus_labels" \
        --no_smoothing
    ```
* **Figure 5D, 5E, 5F (UMAP Evaluation and Smoothing Window Analysis):**
    1.  Run from the `tweety_bert` root directory, providing the path to the UMAP fold data directory and the desired output directory:
        ```bash
        python figure_generation_scripts/umap_eval.py \
            --folder_path "path_to_umap_eval" \
            --output_dir "/results/test"

        ```

    2.  This script will generate:
        * `all_windows_summary.txt` in the `output_dir`, containing statistics for each smoothing window and identifying the optimal one.
        * `metrics_by_window.png` (used for Fig 5F) in the `output_dir`.
        * Plots for Fig 5D and 5E (normalized confusion matrices) will be in the `output_dir/best_window/` subdirectory (e.g., `06_M_norm_fullreorder.png` and `04_M_norm_diag.png` - *note: the exact filenames might vary slightly based on internal plotting choices*).

---

**Figure 6: Evaluating TweetyBERT Embeddings Using Linear Probes**

* **Data Generation:**
    1.  Edit `scripts/linear_probe_automated_analysis.py`:
        * Ensure the list `experiment_configs` has the correct `experiment_path` for your pretrained TweetyBERT model (e.g., `"experiments/TweetyBERT_Paper_Yarden_Model"`).
        * Update `train_dir` and `test_dir` within `experiment_configs` to point to your linear probe datasets (e.g., for llb3, llb11, llb16). Adjust paths like `"/media/george-vengrovski/Desk SSD/TweetyBERT/linear_probe_dataset/{dataset}_train"`.
        * `results_path` is set to `"results"`. If you would like to train the linear probes from scratch, you will have to generate your own linear probe dataset, contact me for more details. 
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

* **Data:** You need NPZ files containing embeddings for different birds across breeding and non-breeding seasons (e.g., `5494_Seasonality_Final.npz`, `5508_Seasonality_Final.npz`). Place these files in a suitable location, for example, `files/seasonality_embeddings/`.
* **Plot Generation:**
    1.  Run the `umap_comparison_figure_generation.py` script from the `tweety_bert` root directory. Provide the desired output directory *first*, followed by the paths to one or more NPZ files containing the seasonality embeddings.
        ```bash
        python figure_generation_scripts/umap_comparison_figure_generation.py \
            "imgs/seasonality_analysis_fig7" \
            "files/seasonality_embeddings/5494_Seasonality_Final.npz" \
            "files/seasonality_embeddings/5508_Seasonality_Final.npz"
        ```
    2.  The script will generate various plots in the specified output directory (e.g., `imgs/seasonality_analysis_fig7/`), creating subdirectories for each bird ID found in the NPZ filenames (e.g., `bird_5494/`, `bird_5508/`). The specific PNG files used for the paper are titled `bird_{bird_id}_all_before_vs_all_after_overlap.png`.

---

