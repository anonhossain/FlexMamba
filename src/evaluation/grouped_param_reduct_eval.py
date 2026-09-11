import copy
from pathlib import Path

import torch

from src.models.grouped_param_reduct_model import (
    GroupedParamReductionModelConfig,
    GroupedParamReductionCausalLM,
)

from src.evaluation.dynamic_mor_eval import (
    run_evaluation,
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

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_grouped_param_reduction(
    summary,
):

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
        != "grouped_dynamic_mamba"
    ):
        raise ValueError(
            "Checkpoint is not "
            "Grouped Dynamic Mamba."
        )

    model_cfg = (
        GroupedParamReductionModelConfig(
            **cfg["model"]
        )
    )

    model = GroupedParamReductionCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["grouped_parameterization"],
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

    metrics = run_evaluation(
        model,
        batches,
        device,
    )

    benchmark_scores = None

    if eval_cfg.get(
        "run_benchmarks",
        False,
    ):

        from src.evaluation.benchmark_eval import (
            run_benchmarks,
        )

        print(
            "\nRunning downstream benchmarks..."
        )

        benchmark_scores = run_benchmarks(
            model,
            cfg,
            device,
        )

    parameter_report = (
        model.parameter_report()
    )

    grouping_report = (
        model.grouping_report()
    )

    state_report = model.state_report(
        batch_size=data_cfg[
            "training"
        ]["batch_size"],

        dtype=next(
            model.parameters()
        ).dtype,
    )

    parent_parameters = (
        grouping_report[
            "parent_equivalent_parameters"
        ]
    )

    child_parameters = (
        parameter_report[
            "total_parameters"
        ]
    )

    saved_parameters = (
        parent_parameters
        - child_parameters
    )

    overall_reduction = (
        saved_parameters
        / parent_parameters
    )

    embedding_parameters = (
        parameter_report[
            "embedding_parameters"
        ]
    )

    parent_non_embedding = (
        parent_parameters
        - embedding_parameters
    )

    child_non_embedding = (
        child_parameters
        - embedding_parameters
    )

    non_embedding_reduction = (
        1.0
        - (
            child_non_embedding
            / parent_non_embedding
        )
    )

    projection_compute_ratio = (
        grouping_report[
            "grouped_projection_parameters"
        ]
        / grouping_report[
            "dense_projection_parameters"
        ]
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
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

            "vocab_size": (
                model_cfg.vocab_size
            ),

            "d_model": (
                model_cfg.d_model
            ),

            "d_state": (
                model_cfg.d_state
            ),

            "expand": (
                model_cfg.expand
            ),

            "d_conv": (
                model_cfg.d_conv
            ),

            "max_seq_len": (
                model_cfg.max_seq_len
            ),
        },

        "parameters": parameter_report,

        "parameter_reduction": {
            "parent_parameters": (
                parent_parameters
            ),

            "grouped_parameters": (
                child_parameters
            ),

            "saved_parameters": (
                saved_parameters
            ),

            "overall_reduction": (
                overall_reduction
            ),

            "parent_non_embedding_parameters": (
                parent_non_embedding
            ),

            "grouped_non_embedding_parameters": (
                child_non_embedding
            ),

            "non_embedding_reduction": (
                non_embedding_reduction
            ),
        },

        "grouping": grouping_report,

        "compute": {
            "grouped_projection_dense_ratio": (
                projection_compute_ratio
            ),

            "theoretical_grouped_projection_flop_reduction": (
                1.0
                - projection_compute_ratio
            ),

            "routing_theoretical_recursive_compute_fraction": (
                metrics[
                    "routing"
                ][
                    "theoretical_recursive_compute_fraction"
                ]
            ),

            "routing_compute_savings_realized": False,

            "grouped_kernel_speedup_claimed": False,

            "note": (
                "Grouped projection parameter reduction is real. "
                "Projection FLOP reduction is theoretical at this "
                "prototype stage and should not be treated as a "
                "wall-clock kernel speedup claim."
            ),
        },

        "initialization": (
            summary.get(
                "initialization_report"
            )
        ),

        "training": {
            "completed_steps": (
                summary[
                    "completed_steps"
                ]
            ),

            "training_tokens": (
                summary[
                    "training_tokens"
                ]
            ),

            "final_training_loss": (
                summary[
                    "final_training_loss"
                ]
            ),

            "training_seconds": (
                summary[
                    "training_seconds"
                ]
            ),

            "training_tokens_per_second": (
                summary[
                    "tokens_per_second"
                ]
            ),

            "validation_history": (
                summary[
                    "validation_history"
                ]
            ),
        },

        "state": state_report,

        "routing": (
            metrics["routing"]
        ),

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
        "\nGrouped Parameter Reduction "
        "evaluation completed."
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
        f"{child_parameters:,}"
    )

    print(
        "Parameters saved: "
        f"{saved_parameters:,}"
    )

    print(
        "Overall reduction: "
        f"{overall_reduction * 100:.2f}%"
    )

    print(
        "Non-embedding reduction: "
        f"{non_embedding_reduction * 100:.2f}%"
    )

    print(
        "Grouped projection reduction: "
        f"{grouping_report['projection_parameter_reduction'] * 100:.2f}%"
    )

    print(
        "Average recursion: "
        f"{report['routing']['average_recursion_depth']:.3f}"
    )

    print(
        "Depth fractions: "
        f"{report['routing']['depth_fractions']}"
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