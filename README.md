# Professor Feathers

Professor Feathers is a voice-controlled parrot interaction system built around a **two-stage keyword pipeline**: a base classifier recognizes two commands (`dance` and `sing`), and a dynamic classifier learns new anonymous spoken words that map back to those same actions. The project includes randomized **parrot audio feedback** and offline experiment scripts for the base and dynamic classifier.

## System overview

The project has four main capabilities: data collection, feature extraction, offline evaluation, and live interaction. In the live system, microphone audio is segmented into utterances, passed to the base classifier first, and if the base command is rejected, the dynamic classifier attempts to match a learned anonymous keyword mapped to `dance` or `sing`.

The current design uses two feature spaces on purpose: the base classifier uses a speaker-leaning-invariant keyword representation, while the dynamic classifier uses a richer speaker-aware representation with optional delta-MFCC information to help distinguish user-specific learned words. This separation keeps the first classifier focused on the spoken command itself while allowing the second classifier to exploit speaker-specific cues when learning personalized words.

## Pipeline

1. **Collect recordings** for base keywords and save them as WAV files plus manifest metadata.
2. **Extract features** from each utterance, including MFCC statistics and, for the dynamic branch, optional pitch and delta-MFCC information.
3. **Train / evaluate models** offline with experiment scripts for KNN grid search and cross-validation.
4. **Run live inference** with unknown-word rejection, on-the-fly learning for new dynamic words, and randomized parrot feedback clips during recognition and training.

## Project structure

```text
Professor-Feathers/
├── src/
    ├── audio.py
    ├── collect.py
    ├── config.py
    ├── dual_knn.py
    ├── endpointer.py
    ├── parrot_feedback.py
    ├── storage.py
    ├── servo/
        ├── servo_rcv.ino
        ├── servo_snd.py
    ├── features/
        ├── feature_core.py
        ├── feature_loading.py
        ├── feature_plotting.py
        └── feature_spaces.py
    ├── knn_utils.py
├── live_dual_knn_main.py
├── run_knn_experiments.py
├── run_knn_experiments_dual.py
├── show_feature_space.py
├── start_recording.py
├── pyproject.toml
├── README.md
└── data/
    ├── base_keywords/
    ├── dynamic_keywords/
    ├── parrot_voice/
    ├── manifests/
    ├── knn_experiments/
    └── raw/
```

### Core modules

- `audio.py` — WAV I/O, microphone capture, rolling buffers, and `MicrophoneStream` for live audio input.
- `collect.py` — recording batches, hotkey support, sample saving, and helper functions used during training data collection and online learning.
- `endpointer.py` — speech segmentation that turns continuous microphone audio into utterance events.
- `storage.py` — storage layout and manifest handling for saved samples.
- `feature_core.py` — low-level feature extraction from raw waveforms.
- `feature_loading.py` — loads WAV trees or manifest-backed datasets into feature-ready structures.
- `feature_plotting.py` — PCA-based inspection of feature separation and clustering.
- `feature_spaces.py` — builds the separate feature vectors for the base and dynamic classifiers, including delta-MFCC support.
- `knn_utils.py` — standardization, neighbor search, and shared KNN evaluation helpers.
- `dual_knn.py` — model classes for the live dual-classifier flow.
- `parrot_feedback.py` — randomized playback of parrot feedback WAV files for recognized commands, failed recognition, and training prompts.

### Main scripts

- `live_dual_knn_main.py` — main live interaction loop for continuous listening, recognition, learning, and action triggering.
- `run_knn_experiments.py` — earlier KNN experiment script for feature variants and `k` search on a single classifier setup.
- `run_knn_experiments_dual.py` — offline evaluation for both base and dynamic classifiers with stratified folds, test splits, and confusion matrices.
- `start_recording.py` - live recording of utterances for the first classifier's training.

## Data layout

### Base keywords

The base classifier expects examples under a root such as:

```text
data/base_keywords/
├── dance/
│   ├── session_001/
│   └── session_002/
└── sing/
    ├── session_001/
    └── session_002/
```

These recordings are used to train the first-stage keyword recognizer for the core commands.

### Dynamic keywords

The dynamic classifier stores learned anonymous words under a root such as:

```text
data/dynamic_keywords/
├── dance_001/
├── dance_002/
├── sing_001/
└── sing_002/
```

Each anonymous folder corresponds to a spoken word that is internally mapped back to the `dance` or `sing` action by its prefix.

### Parrot feedback clips

Place the feedback WAV files here:

```text
data/parrot_voice/
├── dance_recognized.wav
├── sing1.wav
├── sing2.wav
├── training1.wav
├── training2.wav
├── not_recognized1.wav
├── not_recognized2.wav
└── not_recognized3.wav
```

The feedback module randomizes among multi-option groups, probabilistically plays training clips during learning iterations, and avoids triggering the dance confirmation every single time.

## Hardware components

- A microphone supported by `sounddevice` for live capture.
- A speaker or headphones for playing parrot feedback WAV files during training and recognition.
- Two servo motors to perform dancing.
- ESP32 Development Board for interaction between the laptop and the servos

## Software components

- Python 3 environment with the project installed in editable mode.
- `numpy` for numeric processing and feature arrays.
- `matplotlib` for experiment plots and confusion matrices.
- `sounddevice` for microphone capture and WAV playback in the live loop.
- Optional YAML configuration support via the project config loader.

## How the live system works

The live loop continuously reads microphone chunks, sends them through the endpointer, and processes complete utterances only when speech boundaries have been detected. A detected utterance is first checked by the base classifier using unknown rejection based on mean neighbor distance plus a minimum label-vote ratio, and only if that base path rejects the sample does the system attempt dynamic-word recognition.

When the user presses the learning hotkeys, the system creates a new anonymous keyword folder, records a batch of examples, writes them to disk, rebuilds the dynamic classifier, and resumes continuous listening. During this process, the feedback module can play randomized training prompts with a configurable probability on each training iteration.

## Installation

Install the package in editable mode:

```bash
python3 -m pip install -e .
```

## Usage

### 1. Collect keyword data

Use your existing collection entrypoint or collector script to record base keywords and dynamic examples into the expected folder layout. For live online learning, the main loop itself can create and fill new `dance_###` or `sing_###` folders when triggered by hotkeys.

### 2. Visualize feature distributions

Generate PCA plots to inspect class overlap and separation in the selected feature space. This is useful for debugging feature quality, although PCA overlap in 2D does not necessarily imply poor performance in the full-dimensional classifier space.

### 3. Run dual KNN experiments

Example:

```bash
python run_knn_experiments_dual.py \
  --base-source-root data/base_keywords \
  --dynamic-source-root data/dynamic_keywords \
  --output-root data/knn_experiments_dual \
  --test-ratio 0.2 \
  --seed 42 \
  --force
```

This script evaluates the base and dynamic classifiers separately, performs stratified validation over multiple `k` values, and saves CSV summaries, test-accuracy reports, and confusion matrices.

### 4. Run the live dual-classifier system

Example:

```bash
python live_dual_knn_main.py
```

In the live system:
- speak a known base command such as `dance` or `sing` to trigger a direct action
- press `d` to start learning a new anonymous word mapped to `dance`
- press `s` to start learning a new anonymous word mapped to `sing`
- press the configured quit key to stop the loop

## Unknown-word rejection

The base classifier uses a combination of mean `k`-nearest-neighbor distance and minimum label-vote ratio to decide whether an utterance should be accepted as a known base command. The dynamic classifier also uses an automatically derived distance threshold so that unrecognized learned words do not trigger actions too aggressively.

This rejection logic is important because the live system is open-set in practice: many incoming sounds are not valid commands and should fail safely rather than forcing a wrong label.

## Experiment outputs

The experiment scripts write structured outputs such as:

- validation CSV files
- held-out test accuracy summaries
- per-class metrics
- confusion matrices in CSV and PNG form
- feature-distribution and model-comparison plots
- summary JSON files with the selected best model settings

These outputs make it easier to track regressions, compare feature variants and justify hyperparameters choice.

## Configuration notes

The live script exposes important constants near the top of the file, including `BASE_K`, `DYNAMIC_K`, unknown-threshold settings, action-label prefixes, and toggles such as `USE_DELTA_FOR_DYNAMIC_KNN`. The feedback behavior can also be tuned through the parrot-feedback module, including the dance trigger probability and the per-iteration training feedback probability.


## Next improvements

Potential next steps include adding a cleaner config entry for feedback probabilities, supporting more actions than `dance` and `sing`, exposing the live interaction state in a small UI instead of the console and enabling the system to fully run on an edge device, rather than requiring serial connection to a laptop.