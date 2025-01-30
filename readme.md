/*
OVERVIEW
--------
TweetyBERT combines a convolutional front-end with a transformer architecture to learn representations of bird vocalizations. The model can be used for:
 - Automated/Unsupervised labeling of songbird syllables
 - Comparing embeddings before / after perturbation
 - Visualizing song with dimensionality reduction

INSTALLATION & ENVIRONMENT SETUP
--------------------------------
Below is a sample workflow using Conda. This includes creating a dedicated environment, installing necessary packages, and cloning the TweetyBERT repository.

NOTE: The steps below assume a CUDA-capable GPU (e.g., NVIDIA RTX 4090). Adjust as necessary for your system.

1) Create and activate a new Conda environment:
   conda create -n tweetybert python=3.11
   conda activate tweetybert

2) Install core scientific packages (including librosa):
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

3) Install additional dependencies via pip (using pip for soundfile instead of conda):
   pip install soundfile shutil-extra

4) (Optional) Install PyTorch if not already installed (adjust CUDA version if needed):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12

5) Clone the TweetyBERT repository:
   git clone https://github.com/yourusername/TweetyBERT.git
   cd TweetyBERT

   IMPORTANT: If you want to run the shell scripts provided (e.g., for pretraining, training, etc.), you must:
   cd shell_scripts
   and then run the desired shell script (like ./train_decoder.sh).

STORAGE REQUIREMENTS
--------------------
Depending on the size of your audio dataset, the storage requirements can range from 50 GB to 1 TB (or more). Ensure you have sufficient disk space available.

GPU & TRAINING TIMES
--------------------
Pretraining can take several hours (or days) on a single NVIDIA RTX 4090 GPU, depending on your dataset size and chosen hyperparameters.

PREREQUISITES RECAP
-------------------
 - Python 3.11+
 - PyTorch >= 2.0
 - CUDA 12.x (for GPU acceleration)
 - Required packages: numpy, matplotlib, tqdm, umap-learn, hdbscan, scikit-learn, pandas, seaborn, jupyter, ipykernel, librosa, soundfile, shutil-extra

SONG DETECTION & JSON FORMAT
----------------------------
TweetyBERT uses a song detection file (JSON) to locate regions within each recording where bird song is present. A single recording file can contain multiple songs, and each song is stored in a separate list or segment. The JSON also supports optional syllable labels for performance analysis/validation.

Example of a simplified detection JSON entry (placeholder values):
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
 - filename: The WAV file name
 - song_present: Whether any song was detected in this file
 - segments: Each detected song segment with onset/offset times
 - spec_parameters: Parameters (like step_size and nfft) used for spectrogram generation
 - syllable_labels (optional): Time intervals for each labeled syllable, keyed by the label ID

Typically, you will use a separate Song Detection tool to generate this JSON. TweetyBERT just needs to know where the songs occur (and optionally any known syllable labels).

PRETRAINING
-----------
TweetyBERT can be pretrained on any set of WAV files where the bird songs are marked in a corresponding JSON. The only parameters you generally need to edit are:

INPUT_DIR="/path/to/wav/files"
SONG_DETECTION_JSON_PATH=None   # or "/path/to/song_detection.json"
TEST_PERCENTAGE=20
EXPERIMENT_NAME="TESTOSTERONE_MODEL"

 - INPUT_DIR: Path to the folder containing WAV files
 - SONG_DETECTION_JSON_PATH: Path to the JSON file (or None if you don't have one)
 - TEST_PERCENTAGE: Percentage of data to reserve for testing (20% by default)
 - EXPERIMENT_NAME: Name of the training run; results go to experiments/<EXPERIMENT_NAME>

TRAINING A DECODER
------------------
After pretraining, you can train a decoder (linear classifier) to label syllables (or cluster IDs, etc.). You can use train_decoder.sh or train_decoder_multiple_dir.sh.

Example of key variables to edit:
BIRD_NAME="example_bird_name"
MODEL_NAME="EXAMPLE_MODEL_FOR_PAPER"
WAV_FOLDER="/path/to/wav/files"
SONG_DETECTION_JSON_PATH="/path/to/song_detection.json"
NUM_SAMPLES=15

 - BIRD_NAME: A short descriptive name for logs
 - MODEL_NAME: The model name (EXPERIMENT_NAME) used during pretraining
 - WAV_FOLDER: Path to the bird's WAV files
 - SONG_DETECTION_JSON_PATH: Path to the detection JSON
 - NUM_SAMPLES: (Optional) Number of WAVs to sample for embedding/training

INFERENCE
---------
Once the decoder is trained, you can run inference on new WAV files:
WAV_FOLDER="/path/to/wav/files"
SONG_DETECTION_JSON_PATH="/path/to/song_detection.json"
BIRD_NAME="example_bird"
APPLY_POST_PROCESSING="True"

 - WAV_FOLDER: Directory of WAV files
 - SONG_DETECTION_JSON_PATH: The detection JSON
 - BIRD_NAME: The bird name (for consistent logging)
 - APPLY_POST_PROCESSING: If "True", merges or cleans up very short segments

During inference, TweetyBERT will produce a JSON database summarizing song presence and detected syllables for each WAV file. Example:
{
  "metadata": {
    "classifier_path": "experiments/EXAMPLE_MODEL_FOR_PAPER",
    "spec_dst_folder": "imgs/decoder_specs_inference_test",
    "output_path": "files/example_bird_decoded_database.json",
    "visualize": false,
    "dump_interval": 500,
    "apply_post_processing": true
  },
  "results": [
    {
      "file_name": "example_bird_1688_2018_04_27_12_27_27.wav",
      "creation_date": "2018-05-07T09:08:30",
      "song_present": false,
      "syllable_onsets_offsets_ms": {},
      "syllable_onsets_offsets_timebins": {}
    },
    {
      "file_name": "example_bird_3251_2018_05_01_05_48_59.wav",
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

PROJECT STRUCTURE
-----------------
tweety_bert_paper/
├── src/                (Core model implementation and primary codebase)
├── scripts/            (Helper scripts and utilities for one-off tasks)
├── results/            (Output data from model computations and analysis)
├── imgs/               (Generated images, plots, and visualizations)
├── files/              (NPZ files and JSON annotation databases)
├── figure_generation_scripts/  (Scripts for generating paper figures)
└── shell_scripts/      (Shell scripts for automation and deployment)

DIRECTORY DETAILS
-----------------
 - src/: Core TweetyBERT model code, training logic, essential components
 - scripts/: Helper utilities and standalone scripts for specific tasks
 - results/: Storage for computation outputs, evaluation metrics, analysis
 - imgs/: Generated visualizations, spectrograms, and other image outputs
 - files/: NPZ data files and JSON databases containing song annotations
 - figure_generation_scripts/: Scripts for generating publication-ready figures
 - shell_scripts/: Automation scripts for running experiments and deployment

HAPPY TWEETYBERTING!
--------------------
If you have any questions, suggestions, or would like to contribute, feel free to open an issue or pull request on our repository.
*/
