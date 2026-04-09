# Professor Feathers Audio Collection

This repository contains a focused audio collection workflow:

- `pf-collect`: arm recording with a hotkey, wait for speech, detect end-of-word, and save each sample.

The recorder uses a lightweight energy-based endpointer so one key press can start a batch and segment each spoken word into its own WAV file automatically.

## Project layout

```text
src/
  audio.py
  collect.py
  config.py
  endpointer.py
  features.py
  storage.py
data/
  raw/
  manifests/
```

## Quick start

1. Install the package in editable mode:

```bash
python3 -m pip install -e .
```

2. Collect samples for a keyword:

```bash
pf-collect --keyword hello_bird
```

Press `space` once to start a batch of 20 recordings, then say the keyword once per sample with a brief pause between repetitions. The collector will auto-arm for the next word until the batch completes. Press `esc` to quit.

If you want to start collection directly from the IDE, run [start_recording.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_recording.py) and edit the `KEYWORD` constant at the top of that file first.

3. Build and view a 2D feature-space plot from the saved samples:

```bash
pf-features
```

This extracts a per-sample feature vector made from MFCC statistics plus pitch (`f0`) statistics and builds a PCA dashboard. The saved figure includes multiple views such as exact-label clustering, base-keyword vs speaker comparisons, `PC1/PC2`, `PC1/PC3`, `PC2/PC3`, label centroids, and explained-variance bars. The dashboard is saved to `data/features/feature_space.png` and opened on screen. You can also run [show_feature_space.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/show_feature_space.py) directly from the IDE.

If you want to visualize only a hand-picked subset of the dataset, point the plotter at a folder tree of WAV files instead of the full manifest:

```bash
pf-features --samples-root data/feature_space_selection
```

The expected folder layout is either `<samples-root>/<keyword>/<session-id>/sample.wav` or `<samples-root>/<keyword>/sample.wav`. A helper folder with instructions is available at [data/feature_space_selection/README.txt](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/data/feature_space_selection/README.txt).

## Notes

- Audio is recorded as mono 16 kHz PCM WAV.
- Samples are saved under `data/raw/<keyword>/<session-id>/sample_XXXX.wav`.
- Metadata is appended to `data/manifests/samples.jsonl`.

## Detailed flow

**Overview**

This repo is a small speech-command pipeline with four main jobs:

1. record isolated spoken words/commands
2. save them as WAVs plus metadata
3. turn them into feature vectors and visualize them with PCA
4. train/use a simple KNN classifier offline and live

The code is split pretty cleanly between reusable logic in [src/](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src) and runnable entry scripts in the repo root.

**Main Pipelines**

1. **Collection pipeline**
   - IDE entry point: [start_recording.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_recording.py#L24)
   - CLI entry point: `pf-collect` from [pyproject.toml](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/pyproject.toml#L20)
   - Core flow lives in [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py#L188)

   Flow:
   - microphone blocks come from [src/audio.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/audio.py#L140)
   - word boundaries are detected by [src/endpointer.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/endpointer.py#L67)
   - batches, hotkeys, saving, and manifest writes are handled in [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py#L49), [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py#L104), and [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py#L188)
   - files land under `data/raw/<keyword>/<session-id>/sample_XXXX.wav` via [src/storage.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/storage.py#L29)

2. **Feature extraction + PCA visualization**
   - IDE entry point: [show_feature_space.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/show_feature_space.py#L22)
   - CLI entry point: `pf-features` from [pyproject.toml](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/pyproject.toml#L20)
   - Core flow lives in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L918)

   Flow:
   - WAVs are read by [src/audio.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/audio.py#L119)
   - MFCCs are built in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L154)
   - F0 contour is estimated in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L114)
   - raw feature parts are extracted in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L187)
   - feature vectors are assembled in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L350)
   - PCA is computed in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L651)
   - the dashboard is drawn in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L824)

   Current feature variants supported:
   - `no_f0`
   - `f0_mean_std`
   - `f0_contour`
   defined in [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L42)

3. **Offline KNN evaluation**
   - main script: [run_knn_classifier.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_classifier.py#L301)
   - experiment/grid search script: [run_knn_experiments.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_experiments.py#L212)
   - shared KNN math: [src/knn_utils.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/knn_utils.py#L9)

   Flow:
   - scan labeled WAV folders in [run_knn_classifier.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_classifier.py#L38)
   - split into train/test in [run_knn_classifier.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_classifier.py#L49)
   - copy files into `data/knn_split/train` and `data/knn_split/test`
   - build feature matrices with [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py#L405)
   - standardize and run KNN with [src/knn_utils.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/knn_utils.py#L9) and [src/knn_utils.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/knn_utils.py#L43)
   - save predictions, confusion matrices, scaler, and F0 stats from [run_knn_classifier.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_classifier.py#L363)

4. **Live command recognition + live learning**
   - entry point: [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L535)

   Flow:
   - load training data from `SOURCE_ROOT` in [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L40)
   - build a live KNN model in [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L179)
   - continuously listen and segment words in [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L391)
   - classify one utterance in [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L242)
   - reject unknown commands using distance/vote thresholds in [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L283)
   - press `d` or `s` to create new synthetic labels like `d_001` / `s_001` in [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py#L306)
   - those new recordings are saved with the shared collection helpers from [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py#L130) and [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py#L145)


What Each File Is For

- [src/audio.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/audio.py): low-level audio I/O, WAV read/write, microphone stream, rolling buffer.
- [src/endpointer.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/endpointer.py): energy-based VAD/endpointer that turns a stream into one utterance.
- [src/storage.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/storage.py): file naming, folder layout, manifest read/write.
- [src/config.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/config.py): app defaults and optional YAML override loading.
- [src/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/collect.py): reusable recording pipeline, hotkeys, batch recording, saving samples.
- [src/features.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/features.py): feature extraction, F0 normalization logic, sample loading, PCA, plotting.
- [src/knn_utils.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/knn_utils.py): shared KNN helpers used by both offline and live classification.
- [start_recording.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_recording.py): simple IDE launcher for collection.
- [show_feature_space.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/show_feature_space.py): simple IDE launcher for PCA plots.
- [run_knn_classifier.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_classifier.py): one-shot train/test split plus KNN evaluation and confusion matrices.
- [run_knn_experiments.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/run_knn_experiments.py): compare feature variants and multiple `k` values.
- [start_live_classification.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/start_live_classification.py): continuous listening, live classification, unknown rejection, and command learning.
- [src/pipelines/collect.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/src/pipelines/collect.py): compatibility wrapper only.
- [README.md](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/README.md): quick-start and project layout.

**Data / Outputs**

- `data/raw`: normal collected training data for keywords/commands.
- `data/live`: current live-classification training source if `SOURCE_ROOT` points there in start_live_classification.py
- `data/manifests/samples.jsonl`: manifest of saved samples from the collection pipeline.
- `data/feature_space_selection`: manual subset folder for PCA plotting.
- `data/features`: saved PCA dashboards and summaries.
- `data/knn_split`: offline train/test split outputs, predictions, confusion matrix, scaler, F0 stats.
- `data/knn_experiments`: grid-search results and comparison plots.

**How The Pieces Fit Together**

The reusable stack is:

`audio.py` -> `endpointer.py` -> `collect.py` / `start_live_classification.py`  
`storage.py` stores recordings and manifest records  
`features.py` turns recordings into vectors  
`knn_utils.py` runs KNN on those vectors  
top-level scripts decide which workflow to run