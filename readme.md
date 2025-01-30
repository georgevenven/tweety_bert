# TweetyBERT

A self-supervised transformer-based model for analyzing and decoding bird vocalizations, with a focus on canary song analysis.

## Installation & Environment Setup

Below is a sample workflow using Conda. This includes creating a dedicated environment, installing necessary packages, and cloning the TweetyBERT repository:

```bash
# 1. Create and activate a new Conda environment
conda create -n tweetybert python=3.11
conda activate tweetybert

# 2. Install core scientific packages
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
    ipykernel

# 3. Install audio processing libraries
conda install -c conda-forge \
    librosa \
    soundfile

# 4. Install additional dependencies via pip
pip install shutil-extra

# 5. (Optional) Install PyTorch if not already installed
#    Note: Adjust CUDA version if necessary.
#    For more details, see https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12

# 6. Clone the TweetyBERT repository
git clone https://github.com/yourusername/TweetyBERT.git
cd TweetyBERT
```

---

## Overview

TweetyBERT combines a convolutional front-end with a transformer architecture to learn representations of bird vocalizations. The model can be used for:

- Automated/Unsupervised labeling of songbird syllables  
- Comparing embeddings before / after perturbation  
- Visualizing song with dimensionality reduction  

### Prerequisites Recap

- Python 3.11+  
- PyTorch >= 2.0  
- CUDA 12.x (for GPU acceleration)  
- Required packages: `numpy`, `matplotlib`, `tqdm`, `umap-learn`, `hdbscan`, `scikit-learn`, `pandas`, `seaborn`, `jupyter`, `ipykernel`, `librosa`, `soundfile`, `shutil-extra`

---

## Song Detection & JSON Format

TweetyBERT uses a song detection file (JSON) to locate regions within each recording where bird song is present. A single recording file can contain multiple songs, and each song is stored in a separate list or segment. The JSON also supports **optional** syllable labels for performance analysis / validation.

Below is a **simplified** example of how a single entry in the song detection JSON might look. Placeholder values are used here:

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
    },
    {
      "onset_timebin": 600,
      "offset_timebin": 900,
      "onset_ms": 7890.12,
      "offset_ms": 12345.67
    }
  ],
  "spec_parameters": {
    "step_size": 119,
    "nfft": 1024
  },
  "syllable_labels": {
    "1": [
      [1.00, 2.50],
      [3.75, 4.10]
    ],
    "7": [
      [5.20, 5.45]
    ]
  }
}
```

- **filename**: The WAV file name.  
- **song_present**: Whether any song was detected in this file.  
- **segments**: Each detected song segment with onset/offset times (in timebins and milliseconds).  
- **spec_parameters**: Parameters (like `step_size` and `nfft`) used for spectrogram generation.  
- **syllable_labels (optional)**: Time intervals for each labeled syllable, keyed by the label ID. Used for evaluating the unsupervised analysis. 

Typically, you will use a separate **Song Detection tool** to generate this JSON. TweetyBERT just needs to know **where** the songs occur (and optionally any known syllable labels).

---

## Pretraining

TweetyBERT can be pretrained on any set of WAV files where the bird songs are marked in a corresponding JSON. Below are the **only** parameters you must edit for the pretraining script:

```bash
INPUT_DIR="/home/george-vengrovski/Documents/testost_pretrain"
SONG_DETECTION_JSON_PATH=None
TEST_PERCENTAGE=20
EXPERIMENT_NAME="TESTOSTERONE_MODEL"
```

- **INPUT_DIR**: Path to the folder containing WAV files  
- **SONG_DETECTION_JSON_PATH**: Path to the JSON file (or `None` if you don't have one)  
- **TEST_PERCENTAGE**: Percentage of data to reserve for testing (20% by default)  
- **EXPERIMENT_NAME**: Name of the training run; results will go to `experiments/<EXPERIMENT_NAME>`  

No other changes are required in the script.

---

## Training a Decoder

After pretraining, you can train a decoder (linear classifier) to label syllables (or cluster IDs, etc.).  
You can use either `train_decoder.sh` or `train_decoder_multiple_dir.sh` (for selecting subfolders/dates).

Below is an example of **key variables** to edit:

```bash
BIRD_NAME="LLb3_test_with_modification_toscrtipt"
MODEL_NAME="LLB_Model_For_Paper"
WAV_FOLDER="/media/george-vengrovski/George-SSD/llb_stuff/llb_birds/yarden_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json"
NUM_SAMPLES=15
```

- **BIRD_NAME**: A short descriptive name used in UMAP and output logs  
- **MODEL_NAME**: The same model name (`EXPERIMENT_NAME`) used during pretraining  
- **WAV_FOLDER**: Path to the bird's WAV files  
- **SONG_DETECTION_JSON_PATH**: Path to the detection JSON  
- **NUM_SAMPLES**: (Optional) Number of WAVs to sample for embedding or training  

---

## Inference

Once the decoder is trained, you can run inference on new WAV files:

```bash
WAV_FOLDER="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/llb3_songs"
SONG_DETECTION_JSON_PATH="/media/george-vengrovski/disk2/canary/yarden_data/llb3_data/onset_offset_results.json"
BIRD_NAME="llb3"
APPLY_POST_PROCESSING="True"
```

- **WAV_FOLDER**: Directory of WAV files  
- **SONG_DETECTION_JSON_PATH**: The detection JSON for these files  
- **BIRD_NAME**: The bird name (for consistent logging)  
- **APPLY_POST_PROCESSING**: If `"True"`, merges or cleans up very short segments in the final output  

The inference step will produce a JSON database summarizing whether each file had song, plus the detected syllables (onset/offset times). Example:

```json
{
  "metadata": {
    "classifier_path": "experiments/LLB_Model_For_Paper",
    "spec_dst_folder": "imgs/decoder_specs_inference_test",
    "output_path": "files/llb3_decoded_database.json",
    "visualize": false,
    "dump_interval": 500,
    "apply_post_processing": true
  },
  "results": [
    {
      "file_name": "llb3_1688_2018_04_27_12_27_27.wav",
      "creation_date": "2018-05-07T09:08:30",
      "song_present": false,
      "syllable_onsets_offsets_ms": {},
      "syllable_onsets_offsets_timebins": {}
    },
    {
      "file_name": "llb3_3251_2018_05_01_05_48_59.wav",
      "creation_date": "2018-05-07T09:09:59",
      "song_present": true,
      "syllable_onsets_offsets_ms": {
        "1": [
          [0.0, 2579.6825396825398]
        ],
        "3": [
          [2579.6825396825398, 2663.3333333333335],
          [3834.444444444445, 6508.571428571429],
          ...
        ]
      },
      "syllable_onsets_offsets_timebins": {
        "1": [[0.0, 956]],
        "3": [[956.0, 987], [1421.0, 2412], ...]
      }
    }
  ]
}
```

The `metadata` block describes the inference configuration, while each entry in `results` provides decoding details for each WAV file.

---

## Project Structure

```
tweety_bert_paper/
├── src/              # Core model implementation and primary codebase
├── scripts/          # Helper scripts and utilities for one-off tasks
├── results/          # Output data from model computations and analysis
├── imgs/             # Generated images, plots, and visualizations
├── files/            # NPZ files and JSON annotation databases
├── figure_generation_scripts/  # Scripts for generating paper figures
└── shell_scripts/    # Shell scripts for automation and deployment
```

### Directory Details

- **src/**: Contains the core TweetyBERT model implementation, training logic, and essential components
- **scripts/**: Helper utilities and standalone scripts for specific tasks or data processing
- **results/**: Storage for computation outputs, evaluation metrics, and analysis results
- **imgs/**: Generated visualizations, spectrograms, and other image outputs
- **files/**: Storage for NPZ data files and JSON databases containing song annotations
- **figure_generation_scripts/**: Specialized scripts for generating publication-ready figures
- **shell_scripts/**: Automation scripts for running experiments and deployment

---

### Happy TweetyBERTing!

If you have any questions, suggestions, or would like to contribute, feel free to open an issue or pull request on our repository.
