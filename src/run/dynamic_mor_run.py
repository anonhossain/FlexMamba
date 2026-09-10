from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dynamic_mor_train import run_dynamic_mor


if __name__ == "__main__":
    run_dynamic_mor()