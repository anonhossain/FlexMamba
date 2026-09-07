import sys
from pathlib import Path


# Project root:
# mor_16m_prototype/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.training.mamba_train import train_mamba


if __name__ == "__main__":
    train_mamba(
        base_config="src/configs/base_16m.yaml",
        mamba_config="src/configs/techniques/1.mamba.yaml",
        #steps=5,  # smoke test
    )