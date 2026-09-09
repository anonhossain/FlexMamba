# src/run/mamba_16m_run.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.mamba_16m_train import train_mamba
from src.evaluation.mamba_16m_evaluation import evaluate_mamba


CONFIG_PATH = "src/configs/techniques/s00_mamba_16m.yaml"


if __name__ == "__main__":
    summary = train_mamba(
        config_path=CONFIG_PATH,
    )

    evaluation = evaluate_mamba(
        summary
    )

    print("\nMamba pipeline completed.")
    print(f"Run: {summary['run_dir']}")