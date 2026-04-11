from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collect import apply_cli_overrides, build_session_id, run_collection
from config import load_app_config


# Edit these values before running this file from the IDE.
KEYWORD = "parrot1"
CONFIG_PATH = None
ARM_KEY = None
QUIT_KEY = None
BATCH_SIZE = None


def main() -> int:
    config = apply_cli_overrides(
        load_app_config(config_path=CONFIG_PATH),
        keyword=KEYWORD,
        arm_key=ARM_KEY,
        quit_key=QUIT_KEY,
        batch_size=BATCH_SIZE,
    )
    return run_collection(
        config=config,
        project_root=PROJECT_ROOT,
        keyword=KEYWORD,
        session_id=build_session_id(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
