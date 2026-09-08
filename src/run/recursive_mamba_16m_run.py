#src/run/recursive_mamba_run.py (step-2)

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.recursive_mamba_16m_train import train_recursive_mamba

if __name__ == "__main__":
    train_recursive_mamba(
        base_config="src/configs/base_16m.yaml",
        mamba_config="src/configs/techniques/s01_mamba.yaml",
        recursive_config="src/configs/techniques/s02_recursive_mamba.yaml",
        steps=1,  # smoke test
    )