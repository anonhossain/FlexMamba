# src/run/mor_16m_run.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.mor_16m_train import train_mor
from src.evaluation.mor_16_evaluation import evaluate_mor
from src.training.common_16m import save_json


CONFIG_PATH = "src/configs/techniques/s00_mor_16m.yaml"


def run_mor_16m():
    print("=" * 60)
    print("S00 — MoR 16M")
    print("=" * 60)

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n[1/2] Training MoR 16M...\n")

    summary = train_mor(
        config_path=CONFIG_PATH
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    print("\n[2/2] Evaluating MoR 16M...\n")

    evaluation = evaluate_mor(
        summary
    )

    # ========================================================
    # FINAL PIPELINE STATUS
    # ========================================================

    run_dir = Path(summary["run_dir"])

    save_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "stage": "s00",
            "technique": summary["technique"],
            "scale": summary["scale"],
            "checkpoint": summary["final_checkpoint"],
            "evaluation": str(run_dir / "evaluation.json"),
        },
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 60)
    print("S00 MoR 16M COMPLETED")
    print("=" * 60)

    print(f"Run:        {run_dir}")
    print(f"Checkpoint: {summary['final_checkpoint']}")
    print(f"Evaluation: {run_dir / 'evaluation.json'}")

    print(
        f"NLL:        "
        f"{evaluation['quality']['nll']:.4f}"
    )

    print(
        f"PPL:        "
        f"{evaluation['quality']['perplexity']:.2f}"
    )

    print(
        f"Avg depth:  "
        f"{evaluation['recursion']['average_recursion_depth']:.3f}"
    )

    return {
        "training": summary,
        "evaluation": evaluation,
    }


if __name__ == "__main__":
    run_mor_16m()