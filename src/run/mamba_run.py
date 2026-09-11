# src/run/mamba_16m_run.py

import sys
from pathlib import Path

PROJECT_ROOT = (Path(__file__).resolve().parents[2])

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT),)

from src.training.mamba_16m_train import train_mamba
from src.evaluation.mamba_16m_evaluation import evaluate_mamba
from src.training.common_16m import save_json


CONFIG_PATH = ("src/configs/techniques/s01_mamba.yaml")

def run_mamba_16m(parent_run_dir=None,):

    print("\n[1/2] Training Mamba 16M...\n")

    summary = train_mamba(config_path=CONFIG_PATH,parent_run_dir=parent_run_dir,)

    # ========================================================
    # EVALUATE
    # ========================================================

    print(
        "\n[2/2] Evaluating Mamba 16M...\n"
    )

    evaluation = evaluate_mamba(summary)

    # ========================================================
    # STATUS
    # ========================================================

    run_dir = Path(
        summary["run_dir"]
    )

    save_json(
        run_dir
        / "status.json",
        {
            "status":
                "completed",

            "stage":
                "s01",

            "technique":
                "mamba",

            "parent_checkpoint":
                summary[
                    "parent_checkpoint"
                ],

            "checkpoint":
                summary[
                    "final_checkpoint"
                ],

            "evaluation":
                str(
                    run_dir
                    / "evaluation.json"
                ),
        },
    )

    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S01 Mamba 16M COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Run:        "
        f"{run_dir}"
    )

    print(
        f"Parent:     "
        f"{summary['parent_checkpoint']}"
    )

    print(
        f"Checkpoint: "
        f"{summary['final_checkpoint']}"
    )

    print(
        f"NLL:        "
        f"{evaluation['quality']['nll']:.4f}"
    )

    print(
        f"PPL:        "
        f"{evaluation['quality']['perplexity']:.2f}"
    )

    return {
        "training":
            summary,

        "evaluation":
            evaluation,
    }


if __name__ == "__main__":

    # If run directly:
    #
    # automatically selects latest
    # completed MoR checkpoint.

    run_mamba_16m()