import json
import time
from pathlib import Path

import torch

from src.models.turbo_state_compression_model import (
    TurboStateCompressionModelConfig,
    TurboStateCompressionCausalLM,
)

from src.evaluation.turbo_state_compression_eval import (
    calibrate_state_quantization,
    evaluate_turbo_state_compression,
)

from src.training.common_16m import (
    PROJECT_ROOT,
    load_config,
    choose_device,
    set_seed,
    build_train_loader,
    create_run_dir,
    save_json,
    save_resolved_config,
)


CONFIG_PATH = (
    "src/configs/techniques/"
    "s08_turbo_state_compression.yaml"
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def checkpoint_from_run(run_dir):

    run_dir = resolve_path(
        run_dir
    )

    summary_path = (
        run_dir
        / "run_summary.json"
    )

    if summary_path.exists():

        with summary_path.open(
            "r"
        ) as file:

            summary = json.load(
                file
            )

        checkpoint = resolve_path(
            summary[
                "final_checkpoint"
            ]
        )

        if checkpoint.exists():
            return checkpoint

    checkpoints = sorted(
        (
            run_dir
            / "checkpoints"
        ).glob(
            "step_*.pt"
        )
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}"
        )

    return checkpoints[-1]


# ============================================================
# FIND S07 PARENT
# ============================================================

def find_parent_checkpoint(
    cfg,
    parent_run_dir=None,
):

    if parent_run_dir is not None:

        return checkpoint_from_run(
            parent_run_dir
        )

    if (
        cfg["parent"][
            "technique"
        ]
        != "grouped_lte"
    ):
        raise ValueError(
            "S08 parent must be grouped_lte."
        )

    runs_root = Path(
        cfg[
            "experiment"
        ].get(
            "runs_root",
            "runs",
        )
    )

    if not runs_root.is_absolute():

        runs_root = (
            PROJECT_ROOT
            / runs_root
        )

    technique_dir = (
        runs_root
        / "grouped_lte"
    )

    selected_run = cfg[
        "parent"
    ].get(
        "selected_run",
        "auto",
    )

    if selected_run == "auto":

        runs = sorted(
            technique_dir.glob(
                "run_*"
            ),

            key=lambda path: int(
                path.name.split(
                    "_"
                )[-1]
            ),

            reverse=True,
        )

        for run_dir in runs:

            status_path = (
                run_dir
                / "status.json"
            )

            if not status_path.exists():
                continue

            with status_path.open(
                "r"
            ) as file:

                status = json.load(
                    file
                )

            if (
                status.get(
                    "status"
                )
                != "completed"
            ):
                continue

            try:

                return checkpoint_from_run(
                    run_dir
                )

            except FileNotFoundError:

                continue

        raise FileNotFoundError(
            "No completed S07 Grouped + LTE run found."
        )

    run_name = (
        f"run_{selected_run:03d}"
        if isinstance(
            selected_run,
            int,
        )
        else str(
            selected_run
        )
    )

    if not run_name.startswith(
        "run_"
    ):

        run_name = (
            f"run_{int(run_name):03d}"
        )

    return checkpoint_from_run(
        technique_dir
        / run_name
    )


# ============================================================
# LOAD S07
# ============================================================

def initialize_from_parent(
    model,
    checkpoint_path,
):

    print(
        "\nLoading Grouped + LTE parent:\n"
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "Parent checkpoint has no config."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Parent checkpoint has no model_state_dict."
        )

    parent_cfg = checkpoint[
        "config"
    ]

    parent_type = (
        parent_cfg
        .get(
            "technique",
            {},
        )
        .get(
            "model_type"
        )
    )

    if (
        parent_type
        != "grouped_lte_dynamic_mamba"
    ):
        raise ValueError(
            "Expected grouped_lte_dynamic_mamba parent, "
            f"received {parent_type}"
        )

    parent_model = parent_cfg[
        "model"
    ]

    compatibility = [
        "vocab_size",
        "d_model",
        "input_layers",
        "shared_middle_layers",
        "output_layers",
        "d_state",
        "expand",
        "d_conv",
        "max_seq_len",
    ]

    for key in compatibility:

        parent_value = (
            parent_model[
                key
            ]
        )

        child_value = getattr(
            model.cfg,
            key,
        )

        if (
            parent_value
            != child_value
        ):
            raise ValueError(
                f"Parent/child mismatch for {key}: "
                f"{parent_value} != {child_value}"
            )

    if (
        parent_cfg[
            "grouped_parameterization"
        ][
            "groups"
        ]
        != model.groups
    ):
        raise ValueError(
            "S07/S08 grouping mismatch."
        )

    if (
        parent_cfg[
            "lte"
        ][
            "num_bins"
        ]
        != model.lte_cfg[
            "num_bins"
        ]
    ):
        raise ValueError(
            "S07/S08 LTE bin mismatch."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    return {
        "strategy": (
            "copy_parent_add_state_quantization"
        ),

        "transferred_parameter_count": (
            total_parameters
        ),

        "new_trainable_parameters": 0,

        "total_parameter_count": (
            total_parameters
        ),

        "transferred_parameter_fraction": 1.0,
    }


# ============================================================
# CALIBRATION BATCHES
# ============================================================

def get_calibration_batches(
    cfg,
    model_cfg,
):

    count = cfg[
        "state_quantization"
    ].get(
        "calibration_batches",
        1,
    )

    loader = build_train_loader(
        cfg,
        model_cfg,
    )

    iterator = iter(
        loader
    )

    batches = []

    for _ in range(
        count
    ):

        batches.append(
            next(iterator)
        )

    return batches


# ============================================================
# S08 COMPRESSION
# ============================================================

def compress_turbo_state(
    config_path=CONFIG_PATH,
    parent_run_dir=None,
):

    cfg = load_config(
        config_path
    )

    if (
        cfg["technique"][
            "model_type"
        ]
        != "turboquant_mamba"
    ):
        raise ValueError(
            "S08 requires model_type='turboquant_mamba'."
        )

    if cfg[
        "training"
    ].get(
        "enabled",
        False,
    ):
        raise ValueError(
            "S08 is compression/evaluation only; training.enabled must be false."
        )

    set_seed(
        cfg[
            "experiment"
        ][
            "seed"
        ]
    )

    device = choose_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Technique: "
        f"{cfg['technique']['name']}"
    )

    print(
        f"Scale: "
        f"{cfg['experiment']['scale']}"
    )

    parent_checkpoint = (
        find_parent_checkpoint(
            cfg,
            parent_run_dir,
        )
    )

    model_cfg = (
        TurboStateCompressionModelConfig(
            **cfg[
                "model"
            ]
        )
    )

    model = (
        TurboStateCompressionCausalLM(
            model_cfg,
            cfg[
                "recursion"
            ],
            cfg[
                "state"
            ],
            cfg[
                "routing"
            ],
            cfg[
                "grouped_parameterization"
            ],
            cfg[
                "lte"
            ],
            cfg[
                "state_quantization"
            ],
            seed=cfg[
                "experiment"
            ][
                "seed"
            ],
        )
    )

    init_report = (
        initialize_from_parent(
            model,
            parent_checkpoint,
        )
    )

    print(
        "\nInitialization:"
    )

    print(
        f"Strategy: "
        f"{init_report['strategy']}"
    )

    print(
        "Transferred parameters: "
        f"{init_report['transferred_parameter_count']:,}"
    )

    print(
        "New trainable parameters: 0"
    )

    model = model.to(
        device
    )

    print(
        "\nParameters:",
        model.parameter_report(),
    )

    print(
        "Architecture:",
        model.architecture_report(),
    )

    run_dir = create_run_dir(
        cfg,
        "turboquant_state",
    )

    cfg[
        "runtime"
    ] = {
        "parent_checkpoint": str(
            parent_checkpoint
        ),

        "parent_run_dir": (
            str(
                parent_run_dir
            )
            if parent_run_dir
            else None
        ),

        "initialization_report": (
            init_report
        ),
    }

    save_resolved_config(
        run_dir,
        cfg,
    )

    save_json(
        run_dir
        / "status.json",
        {
            "status": "running",
            "stage": "s08",
            "technique": (
                "turboquant_state"
            ),
            "parent_checkpoint": str(
                parent_checkpoint
            ),
        },
    )

    try:

        start_time = (
            time.perf_counter()
        )

        calibration_batches = (
            get_calibration_batches(
                cfg,
                model_cfg,
            )
        )

        chunk_size = cfg[
            "state_quantization"
        ].get(
            "evaluation_chunk_size",
            64,
        )

        calibration_report = (
            calibrate_state_quantization(
                model,
                calibration_batches,
                device,
                chunk_size,
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        checkpoint_path = (
            run_dir
            / "checkpoints"
            / "step_0000000.pt"
        )

        torch.save(
            {
                "step": 0,

                "model_state_dict": (
                    model.state_dict()
                ),

                "config": cfg,

                "quantization_state": (
                    model.quantization_state_dict()
                ),

                "parent_checkpoint": str(
                    parent_checkpoint
                ),

                "calibration": (
                    calibration_report
                ),
            },

            checkpoint_path,
        )

        summary = {
            "technique": (
                cfg[
                    "technique"
                ][
                    "name"
                ]
            ),

            "scale": (
                cfg[
                    "experiment"
                ][
                    "scale"
                ]
            ),

            "debug": (
                cfg[
                    "experiment"
                ].get(
                    "debug",
                    False,
                )
            ),

            "run": (
                run_dir.name
            ),

            "run_dir": str(
                run_dir
            ),

            "completed_steps": 0,

            "training_tokens": 0,

            "training_seconds": 0.0,

            "compression_seconds": (
                elapsed
            ),

            "parent_checkpoint": str(
                parent_checkpoint
            ),

            "final_checkpoint": str(
                checkpoint_path
            ),

            "parameter_report": (
                model.parameter_report()
            ),

            "architecture_report": (
                model.architecture_report()
            ),

            "initialization_report": (
                init_report
            ),

            "calibration": (
                calibration_report
            ),
        }

        save_json(
            run_dir
            / "run_summary.json",
            summary,
        )

        print(
            f"\nSaved compression checkpoint: "
            f"{checkpoint_path}"
        )

        evaluation = (
            evaluate_turbo_state_compression(
                summary
            )
        )

        save_json(
            run_dir
            / "status.json",
            {
                "status": "completed",

                "stage": "s08",

                "technique": (
                    "turboquant_state"
                ),

                "parent_checkpoint": str(
                    parent_checkpoint
                ),

                "final_checkpoint": str(
                    checkpoint_path
                ),

                "evaluation": str(
                    run_dir
                    / "evaluation.json"
                ),
            },
        )

        return summary, evaluation

    except Exception as error:

        save_json(
            run_dir
            / "status.json",
            {
                "status": "failed",

                "stage": "s08",

                "technique": (
                    "turboquant_state"
                ),

                "parent_checkpoint": str(
                    parent_checkpoint
                ),

                "error": str(
                    error
                ),
            },
        )

        raise


# ============================================================
# RUN S08
# ============================================================

def run_turbo_state_compression(
    parent_run_dir=None,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S08 — TURBOQUANT STATE COMPRESSION"
    )

    print(
        "=" * 60
    )

    summary, evaluation = (
        compress_turbo_state(
            parent_run_dir=parent_run_dir
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S08 TURBOQUANT STATE COMPRESSION COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Run: "
        f"{summary['run_dir']}"
    )

    print(
        f"Parent: "
        f"{summary['parent_checkpoint']}"
    )

    print(
        f"Checkpoint: "
        f"{summary['final_checkpoint']}"
    )

    print(
        "State compression: "
        f"{evaluation['compression']['total_state_compression_ratio']:.2f}x"
    )

    print(
        "State memory reduction: "
        f"{evaluation['compression']['total_state_memory_reduction'] * 100:.2f}%"
    )

    print(
        "PPL change: "
        f"{evaluation['quality_change']['relative_perplexity_change'] * 100:.3f}%"
    )

    return {
        "compression": summary,
        "evaluation": evaluation,
    }