# Professor Feathers Audio Collection

This repository is a speech-command pipeline covering data collection, feature extraction, offline KNN evaluation, and live classification. The project is split between reusable library modules in `src/` and top-level entry scripts.

**Library modules**
- `audio.py` — low-level audio I/O, WAV reading and writing, and the live microphone stream
- `endpointer.py` — energy-based voice activity detection that turns a continuous stream into individual utterances
- `storage.py` — file naming, folder layout, and the sample manifest
- `collect.py` — reusable recording pipeline including hotkeys, batch recording, and sample saving
- `feature_core.py` — core audio feature extraction and feature-vector construction from raw samples.
- `feature_loading.py` — loading WAV samples from the manifest or folder trees and turning them into feature-ready records
- `feature_plotting.py` — PCA computation and feature-space dashboard plotting.
- `knn_utils.py` — shared KNN math used by both offline and live workflows
- `config.py` — app defaults and optional YAML overrides

**Entry scripts**
- `start_recording.py` — launches the data collection pipeline
- `show_feature_space.py` — renders the PCA feature dashboard
- `run_knn_classifier.py` — one-shot train/test split, produces predictions and confusion matrices
- `run_knn_experiments.py` — grid search across feature variants and values of k
- `start_live_classification.py` — continuous real-time listening, classification, unknown rejection, and live label creation

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

