import copy
from pathlib import Path

import torch

from src.models.final_joint_model import (
    FinalJointModelConfig,
    FinalJointCausalLM,
)

from src.models.quantization_aware_tuning_model import (
    QuantizationAwareTuningModelConfig,
    QuantizationAwareTuningCausalLM,
)

from src.evaluation.quantization_aware_tuning_eval import (
    evaluate_qat_batches,
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
# BUILD S09 PARENT
# ============================================================

def build_parent_model(
    parent_checkpoint_path,
):

    checkpoint = torch.load(
        parent_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    model_cfg = (
        QuantizationAwareTuningModelConfig(
            **cfg["model"]
        )
    )

    model = QuantizationAwareTuningCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["grouped_parameterization"],
        cfg["lte"],
        cfg["state_quantization"],
        cfg["quantization"],
        seed=cfg["experiment"]["seed"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    return model


# ============================================================
# EVALUATE S10
# ============================================================

def evaluate_final_joint(summary):

    checkpoint_path = resolve_path(
        summary["final_checkpoint"]
    )

    parent_checkpoint_path = resolve_path(
        summary["parent_checkpoint"]
    )

    run_dir = resolve_path(
        summary["run_dir"]
    )

    device = choose_device()

    print(
        f"\nEvaluation device: {device}"
    )

    print(
        f"Final checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    if (
        cfg["technique"]["model_type"]
        != "flexmamba"
    ):
        raise ValueError(
            "Checkpoint is not final FlexMamba."
        )

    model_cfg = FinalJointModelConfig(
        **cfg["model"]
    )

    model = FinalJointCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["grouped_parameterization"],
        cfg["lte"],
        cfg["state_quantization"],
        cfg["quantization"],
        seed=cfg["experiment"]["seed"],
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

    sequence_tokens = eval_cfg.get(
        "sequence_tokens",
        model_cfg.max_seq_len,
    )

    data_cfg = copy.deepcopy(cfg)

    data_cfg["validation"] = {
        "enabled": True,

        "eval_batches": eval_cfg.get(
            "eval_batches",
            1,
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

    # --------------------------------------------------------
    # S10 FLOAT STATE
    # --------------------------------------------------------

    print(
        "\nRunning final FP-state evaluation..."
    )

    final_float = evaluate_qat_batches(
        model,
        batches,
        device,
        state_mode="float",
        sequence_tokens=sequence_tokens,
    )

    # --------------------------------------------------------
    # S10 QUANTIZED STATE
    # --------------------------------------------------------

    print(
        "Running final INT8-state evaluation..."
    )

    final_quantized = evaluate_qat_batches(
        model,
        batches,
        device,
        state_mode="quantized",
        sequence_tokens=sequence_tokens,
    )

    # --------------------------------------------------------
    # S09 PARENT
    # --------------------------------------------------------

    print(
        "Running S09 parent INT8 evaluation..."
    )

    parent_model = build_parent_model(
        parent_checkpoint_path
    )

    parent_model = parent_model.to(
        device
    )

    parent_quantized = evaluate_qat_batches(
        parent_model,
        batches,
        device,
        state_mode="quantized",
        sequence_tokens=sequence_tokens,
    )

    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------

    grouping = model.grouping_report()

    active_parameters = (
        model.active_parameter_report(
            final_quantized[
                "lte"
            ][
                "active_fraction"
            ]
        )
    )

    compression = (
        model.state_compression_report(
            batch_size=data_cfg[
                "training"
            ][
                "batch_size"
            ],

            dtype=next(
                model.parameters()
            ).dtype,
        )
    )

    parent_ppl = (
        parent_quantized[
            "quality"
        ][
            "perplexity"
        ]
    )

    final_ppl = (
        final_quantized[
            "quality"
        ][
            "perplexity"
        ]
    )

    float_ppl = (
        final_float[
            "quality"
        ][
            "perplexity"
        ]
    )

    improvement = (
        parent_ppl
        - final_ppl
    )

    relative_improvement = (
        improvement
        / parent_ppl
    )

    quantization_gap = (
        final_ppl
        - float_ppl
    )

    relative_quantization_gap = (
        quantization_gap
        / float_ppl
    )

    report = {
        "identity": {
            "technique": "final_joint",
            "model_type": "flexmamba",

            "scale": (
                cfg["experiment"][
                    "scale"
                ]
            ),

            "run": summary["run"],

            "checkpoint": str(
                checkpoint_path
            ),

            "parent_checkpoint": str(
                parent_checkpoint_path
            ),

            "device": str(device),
        },

        "architecture": (
            model.architecture_report()
        ),

        "parameters": (
            model.parameter_report()
        ),

        "grouping": (
            grouping
        ),

        "active_parameters": (
            active_parameters
        ),

        "compression": (
            compression
        ),

        "parent_quantized": (
            parent_quantized
        ),

        "final_float": (
            final_float
        ),

        "final_quantized": (
            final_quantized
        ),

        "quality": (
            final_quantized[
                "quality"
            ]
        ),

        "routing": (
            final_quantized[
                "routing"
            ]
        ),

        "lte": (
            final_quantized[
                "lte"
            ]
        ),

        "quantization": (
            final_quantized[
                "quantization"
            ]
        ),

        "efficiency": (
            final_quantized[
                "efficiency"
            ]
        ),

        "quality_change": {
            "parent_ppl": (
                parent_ppl
            ),

            "final_ppl": (
                final_ppl
            ),

            "absolute_improvement": (
                improvement
            ),

            "relative_improvement": (
                relative_improvement
            ),

            "float_vs_quantized_ppl_gap": (
                quantization_gap
            ),

            "relative_quantization_gap": (
                relative_quantization_gap
            ),
        },

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
        },

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

            "sequence_tokens": (
                sequence_tokens
            ),
        },

        "limitations": {
            "dynamic_depth_compute_realized": False,
            "lte_sparse_compute_realized": False,

            "grouped_parameter_storage_realized": True,

            "int8_state_storage_target": True,

            "note": (
                "Dynamic recursion and LTE activity are measured, "
                "but the generic prototype still executes dense "
                "Mamba computation. Grouped parameter reduction "
                "and INT8 recurrent-state storage are the physically "
                "realized efficiency mechanisms."
            ),
        },
    }

    save_json(
        run_dir / "evaluation.json",
        report,
    )

    print(
        "\nFinal Joint FlexMamba evaluation completed."
    )

    print(
        "S09 parent INT8 PPL: "
        f"{parent_ppl:.2f}"
    )

    print(
        "S10 FP PPL: "
        f"{float_ppl:.2f}"
    )

    print(
        "S10 INT8 PPL: "
        f"{final_ppl:.2f}"
    )

    print(
        "S10 improvement vs S09: "
        f"{relative_improvement * 100:.3f}%"
    )

    print(
        "INT8 vs FP PPL gap: "
        f"{relative_quantization_gap * 100:.4f}%"
    )

    print(
        "Stored parameters: "
        f"{report['parameters']['total_parameters']:,}"
    )

    print(
        "Stored parameter reduction: "
        f"{grouping['overall_parameter_reduction'] * 100:.2f}%"
    )

    print(
        "Average recursion: "
        f"{report['routing']['average_recursion_depth']:.3f}"
    )

    print(
        "Active width: "
        f"{report['lte']['active_fraction'] * 100:.2f}%"
    )

    print(
        "State compression: "
        f"{compression['total_state_compression_ratio']:.2f}x"
    )

    print(
        "State memory reduction: "
        f"{compression['total_state_memory_reduction'] * 100:.2f}%"
    )

    print(
        "State quantization MSE: "
        f"{report['quantization']['mse']:.3e}"
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