from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import build_default_output_path, build_feature_plot


CONFIG_PATH = None
KEYWORD = None
ANNOTATE = False
SAMPLES_ROOT = PROJECT_ROOT / "data" / "feature_space_selection"


def main() -> int:
    saved_path = build_feature_plot(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        keyword=KEYWORD,
        samples_root=SAMPLES_ROOT,
        output_path=build_default_output_path(PROJECT_ROOT),
        show=True,
        annotate=ANNOTATE,
    )
    if saved_path is not None:
        print(f"Saved feature space plot to {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
