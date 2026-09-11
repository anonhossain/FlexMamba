from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.grouped_param_reduct_train import run_grouped_param_reduct


if __name__ == "__main__":
    run_grouped_param_reduct()