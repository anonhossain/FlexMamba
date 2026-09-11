import copy
from pathlib import Path

import torch

from src.models.grouped_lte_model import (
    GroupedLTEModelConfig,
    GroupedLTECausalLM,
)

from src.evaluation.lte_adaptive_width_eval import (
    run_lte_evaluation,
    model_weight_memory_mb,
)

from src.training.common_16m import (
    PROJECT_ROOT,
    build_validation_batches,
    choose_device,
    save_json,
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# EVALUATE
# ============================================================

def evaluate_grouped_lte(summary):

    checkpoint_path = resolve_path(
        summary["final_checkpoint"]
    )

    run_dir = resolve_path(
        summary["run_dir"]
    )

    device = choose_device()

    print(
        f"\nEvaluation device: {device}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    if (
        cfg["technique"]["model_type"]
        != "grouped_lte_dynamic_mamba"
    ):
        raise ValueError(
            "Checkpoint is not Grouped + LTE Dynamic Mamba."
        )

    model_cfg = GroupedLTEModelConfig(
        **cfg["model"]
    )

    model = GroupedLTECausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["grouped_parameterization"],
        cfg["lte"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    eval_cfg = cfg.get(
        "evaluation",
        {},
    )

    data_cfg = copy.deepcopy(cfg)

    data_cfg["validation"] = {
        "enabled": True,

        "eval_batches": eval_cfg.get(
            "eval_batches",
            cfg.get(
                "validation",
                {},
            ).get(
                "eval_batches",
                20,
            ),
        ),
    }

    data_cfg["training"]["batch_size"] = (
        eval_cfg.get(
            "batch_size",
            cfg["training"]["batch_size"],
        )
    )

    batches = build_validation_batches(
        data_cfg,
        model_cfg,
    )

    metrics = run_lte_evaluation(
        model,
        batches,
        device,
    )

    grouping = model.grouping_report()

    active_parameters = model.active_parameter_report(
        metrics["lte"]["active_fraction"]
    )

    parameter_report = model.parameter_report()

    benchmark_scores = None

    if eval_cfg.get(
        "run_benchmarks",
        False,
    ):

        from src.evaluation.benchmark_eval import (
            run_benchmarks,
        )

        benchmark_scores = run_benchmarks(
            model,
            cfg,
            device,
        )

    state_report = model.state_report(
        batch_size=data_cfg[
            "training"
        ]["batch_size"],

        dtype=next(
            model.parameters()
        ).dtype,
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    # --------------------------------------------------------
    # COMBINED THEORETICAL COMPUTE
    # --------------------------------------------------------

    depth_fraction = (
        metrics[
            "routing"
        ][
            "theoretical_recursive_compute_fraction"
        ]
    )

    width_fraction = (
        metrics[
            "lte"
        ][
            "active_fraction"
        ]
    )

    joint_fraction = (
        depth_fraction
        * width_fraction
    )

    report = {
        "identity": {
            "technique": (
                cfg["technique"]["name"]
            ),

            "model_type": (
                cfg["technique"]["model_type"]
            ),

            "scale": (
                cfg["experiment"]["scale"]
            ),

            "run": summary["run"],

            "checkpoint": str(
                checkpoint_path
            ),

            "parent_checkpoint": (
                summary.get(
                    "parent_checkpoint"
                )
            ),

            "device": str(device),
        },

        "architecture": {
            **model.architecture_report(),

            "vocab_size": model_cfg.vocab_size,
            "d_model": model_cfg.d_model,
            "d_state": model_cfg.d_state,
            "expand": model_cfg.expand,
            "d_conv": model_cfg.d_conv,
            "max_seq_len": model_cfg.max_seq_len,
        },

        "parameters": parameter_report,

        "grouping": grouping,

        "active_parameters": (
            active_parameters
        ),

        "initialization": (
            summary.get(
                "initialization_report"
            )
        ),

        "training": {
            "completed_steps": (
                summary["completed_steps"]
            ),

            "training_tokens": (
                summary["training_tokens"]
            ),

            "final_training_loss": (
                summary["final_training_loss"]
            ),

            "training_seconds": (
                summary["training_seconds"]
            ),

            "training_tokens_per_second": (
                summary["tokens_per_second"]
            ),

            "validation_history": (
                summary["validation_history"]
            ),
        },

        "state": state_report,

        "routing": (
            metrics["routing"]
        ),

        "lte": (
            metrics["lte"]
        ),

        "compute": {
            **metrics["compute"],

            "grouped_projection_parameter_fraction": (
                grouping[
                    "grouped_projection_parameters"
                ]
                / grouping[
                    "dense_projection_parameters"
                ]
            ),

            "grouped_projection_theoretical_reduction": (
                grouping[
                    "projection_parameter_reduction"
                ]
            ),

            "joint_depth_width_fraction": (
                joint_fraction
            ),

            "joint_depth_width_reduction": (
                1.0
                - joint_fraction
            ),

            "grouped_parameter_reduction_realized": True,

            "lte_sparse_compute_realized": False,

            "dynamic_depth_sparse_compute_realized": False,

            "note": (
                "Grouped parameter storage reduction is real. "
                "LTE width and Dynamic MoR depth reductions are "
                "currently theoretical compute reductions because "
                "the prototype still executes dense Mamba operations."
            ),
        },

        "quality": (
            metrics["quality"]
        ),

        "efficiency": (
            metrics["efficiency"]
        ),

        "storage": {
            "model_weight_memory_mb": (
                model_weight_memory_mb(
                    model
                )
            ),

            "checkpoint_size_mb": (
                checkpoint_size_mb
            ),
        },

        "benchmarks": (
            benchmark_scores
        ),

        "evaluation_setup": {
            "evaluation_batches": (
                len(batches)
            ),

            "batch_size": (
                data_cfg[
                    "training"
                ][
                    "batch_size"
                ]
            ),

            "sequence_length": (
                model_cfg.max_seq_len
            ),
        },
    }

    save_json(
        run_dir / "evaluation.json",
        report,
    )

    print(
        "\nGrouped + LTE evaluation completed."
    )

    print(
        f"NLL: "
        f"{report['quality']['nll']:.4f}"
    )

    print(
        f"PPL: "
        f"{report['quality']['perplexity']:.2f}"
    )

    print(
        "Parameters: "
        f"{report['parameters']['total_parameters']:,}"
    )

    print(
        "Stored parameter reduction: "
        f"{report['grouping']['overall_parameter_reduction'] * 100:.2f}%"
    )

    print(
        "Grouped projection reduction: "
        f"{report['grouping']['projection_parameter_reduction'] * 100:.2f}%"
    )

    print(
        "Average recursion: "
        f"{report['routing']['average_recursion_depth']:.3f}"
    )

    print(
        "Average active bins: "
        f"{report['lte']['average_active_bins']:.3f}"
    )

    print(
        "Active width: "
        f"{report['lte']['active_fraction'] * 100:.2f}%"
    )

    print(
        "Joint theoretical depth-width reduction: "
        f"{report['compute']['joint_depth_width_reduction'] * 100:.2f}%"
    )

    print(
        "Tokens/sec: "
        f"{report['efficiency']['tokens_per_second']:.2f}"
    )

    print(
        f"Saved: "
        f"{run_dir / 'evaluation.json'}"
    )

    return report