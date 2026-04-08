Manual Feature-Space Selection

Use this folder when you want to plot only a hand-picked subset of the data.

How to use it
1. Copy or move the keyword/session folders you want to visualize into this folder.
2. Keep the keyword/session structure intact when possible:

   data/feature_space_selection/
     sing3/20260407T114911Z/sample_0001.wav
     sing3/20260407T114911Z/sample_0002.wav
     parrot3/20260407T113403Z/sample_0001.wav

3. In show_feature_space.py, set:

   SAMPLES_ROOT = PROJECT_ROOT / "data" / "feature_space_selection"

4. Run show_feature_space.py.

Notes
- When SAMPLES_ROOT is set, the plot ignores the manifest and scans WAV files directly.
- Expected layouts:
  - <samples-root>/<keyword>/<session-id>/sample.wav
  - <samples-root>/<keyword>/sample.wav
- If you want to keep the full raw dataset untouched, copy folders here instead of moving them.
- You can also use the CLI:

  pf-features --samples-root data/feature_space_selection
