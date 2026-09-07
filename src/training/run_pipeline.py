import argparse
import sys
from pathlib import Path

import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from training.run_train import (
    run_training,
)


# ============================================================
# CONFIG
# ============================================================

def load_pipeline(path):
    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    with path.open("r") as file:
        return (
            yaml.safe_load(file)
            or {}
        )["pipeline"]


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    base_config,
    pipeline_config,
    steps=None,
):
    """
    Sequential experiment pipeline.

    Example:

        Mamba
          ↓
        Recursive-Mamba
          ↓
        Dynamic Routing
          ↓
        LTE
          ↓
        ...

    Each stage receives the final checkpoint
    produced by the previous stage.
    """

    pipeline = load_pipeline(
        pipeline_config
    )

    parent_checkpoint = None
    previous_config = None

    for stage in pipeline["stages"]:

        if not stage.get(
            "enabled",
            True,
        ):
            continue

        technique = stage[
            "technique"
        ]

        technique_config = stage[
            "config"
        ]

        print(
            "\n"
            "====================================="
        )

        print(
            f"  {technique.upper()}"
        )

        print(
            "====================================="
        )

        # ====================================================
        # MATCHED CONTINUED-TRAINING CONTROL
        # ====================================================

        if (
            stage.get(
                "matched_control",
                False,
            )
            and parent_checkpoint
            and previous_config
        ):
            print(
                "\nRunning matched control..."
            )

            run_training(
                base_config=
                    base_config,

                technique_config=
                    previous_config,

                steps=
                    steps,

                parent_checkpoint=
                    parent_checkpoint,

                strict_parent=
                    True,

                run_group=
                    f"{technique}_control",
            )

        # ====================================================
        # MAIN TREATMENT
        # ====================================================

        summary = run_training(
            base_config=
                base_config,

            technique_config=
                technique_config,

            steps=
                steps,

            parent_checkpoint=
                parent_checkpoint,
        )

        # ====================================================
        # PASS RESULT TO NEXT STAGE
        # ====================================================

        parent_checkpoint = (
            summary[
                "final_checkpoint"
            ]
        )

        previous_config = (
            technique_config
        )

    print(
        "\n====================================="
    )

    print(
        "Pipeline completed."
    )

    print(
        "Final checkpoint:"
    )

    print(
        parent_checkpoint
    )

    return parent_checkpoint


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        default="configs/base_16m.yaml",
    )

    parser.add_argument(
        "--pipeline",
        default="configs/pipeline_16m.yaml",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    run_pipeline(
        base_config=args.base,
        pipeline_config=args.pipeline,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()