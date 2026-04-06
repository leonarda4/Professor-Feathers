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

This extracts a per-sample feature vector made from MFCC statistics plus pitch (`f0`) statistics, projects all samples into a 2D feature space with PCA, saves the plot to `data/features/feature_space.png`, and opens the plot window. You can also run [show_feature_space.py](/Users/test/Documents/HANDS_ON_AI/Professor%20Feathers/show_feature_space.py) directly from the IDE.

## Notes

- Audio is recorded as mono 16 kHz PCM WAV.
- Samples are saved under `data/raw/<keyword>/<session-id>/sample_XXXX.wav`.
- Metadata is appended to `data/manifests/samples.jsonl`.
