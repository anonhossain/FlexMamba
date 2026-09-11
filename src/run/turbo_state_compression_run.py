from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.turbo_state_compression_train import (
    run_turbo_state_compression,
)


if __name__ == "__main__":
    run_turbo_state_compression()