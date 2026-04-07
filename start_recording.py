from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collect import build_session_id, run_collection
from config import load_app_config


# Edit these values before running this file from the IDE.
KEYWORD = "sing  3"
CONFIG_PATH = None
ARM_KEY = None
QUIT_KEY = None
BATCH_SIZE = None


def main() -> int:
    config = load_app_config(config_path=CONFIG_PATH)
    config.collection.keyword = KEYWORD
    if ARM_KEY:
        config.collection.arm_key = ARM_KEY
    if QUIT_KEY:
        config.collection.quit_key = QUIT_KEY
    if BATCH_SIZE is not None:
        config.collection.batch_size = max(1, int(BATCH_SIZE))
    return run_collection(
        config=config,
        project_root=PROJECT_ROOT,
        keyword=KEYWORD,
        session_id=build_session_id(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
