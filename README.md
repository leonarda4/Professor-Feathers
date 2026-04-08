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
