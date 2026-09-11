import copy
from dataclasses import replace
from pathlib import Path

import torch

from src.models.scaling_efficiency_model import (
    ScalingStudyModelConfig,
    ScalingStudyCausalLM,
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
# ONE SCALE
# ============================================================

def evaluate_scaling_target(
    summary,
):

    checkpoint_path = resolve_path(
        summary[
            "final_checkpoint"
        ]
    )

    run_dir = resolve_path(
        summary[
            "run_dir"
        ]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    model_cfg = (
        ScalingStudyModelConfig(
            **cfg["model"]
        )
    )

    model = (
        ScalingStudyCausalLM(
            model_cfg,
            cfg["recursion"],
            cfg["state"],
            cfg["routing"],
            cfg[
                "grouped_parameterization"
            ],
            cfg["lte"],
            cfg[
                "state_quantization"
            ],
            cfg[
                "quantization"
            ],
            seed=cfg[
                "experiment"
            ][
                "seed"
            ],
        )
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    device = choose_device()

    model = model.to(device)
    model.eval()

    eval_cfg = cfg[
        "evaluation"
    ]

    smoke = cfg.get(
        "smoke",
        {},
    )

    smoke_enabled = smoke.get(
        "enabled",
        False,
    )

    sequence_lengths = (
        smoke.get(
            "sequence_lengths",
            [32],
        )
        if smoke_enabled
        else eval_cfg[
            "sequence_lengths"
        ]
    )

    eval_batches = (
        smoke.get(
            "eval_batches",
            1,
        )
        if smoke_enabled
        else eval_cfg[
            "eval_batches"
        ]
    )

    batch_size = eval_cfg.get(
        "batch_size",
        1,
    )

    sequence_results = {}

    for sequence_length in sequence_lengths:

        data_cfg = copy.deepcopy(
            cfg
        )

        data_cfg["training"][
            "batch_size"
        ] = batch_size

        data_cfg["validation"] = {
            "enabled": True,
            "eval_batches": eval_batches,
        }

        data_model_cfg = replace(
            model_cfg,
            max_seq_len=sequence_length,
        )

        batches = (
            build_validation_batches(
                data_cfg,
                data_model_cfg,
            )
        )

        print(
            f"\nEvaluating "
            f"{cfg['experiment']['scale']} "
            f"@ {sequence_length} tokens..."
        )

        quantized = (
            evaluate_qat_batches(
                model,
                batches,
                device,
                state_mode="quantized",
                sequence_tokens=sequence_length,
            )
        )

        sequence_results[
            str(sequence_length)
        ] = quantized

    primary_length = (
        sequence_lengths[-1]
        if smoke_enabled
        else eval_cfg.get(
            "primary_sequence_length",
            sequence_lengths[-1],
        )
    )

    data_cfg = copy.deepcopy(
        cfg
    )

    data_cfg["training"][
        "batch_size"
    ] = batch_size

    data_cfg["validation"] = {
        "enabled": True,
        "eval_batches": eval_batches,
    }

    primary_model_cfg = replace(
        model_cfg,
        max_seq_len=primary_length,
    )

    primary_batches = (
        build_validation_batches(
            data_cfg,
            primary_model_cfg,
        )
    )

    float_result = (
        evaluate_qat_batches(
            model,
            primary_batches,
            device,
            state_mode="float",
            sequence_tokens=primary_length,
        )
    )

    quantized_result = (
        sequence_results[
            str(primary_length)
        ]
    )

    scale_parameters = (
        model.scale_parameter_report()
    )

    grouping = (
        model.grouping_report()
    )

    active_parameters = (
        model.active_parameter_report(
            quantized_result[
                "lte"
            ][
                "active_fraction"
            ]
        )
    )

    compression = (
        model.state_compression_report(
            batch_size=batch_size,

            dtype=next(
                model.parameters()
            ).dtype,
        )
    )

    average_depth = (
        quantized_result[
            "routing"
        ][
            "average_recursion_depth"
        ]
    )

    depth_fraction = (
        average_depth
        / model.max_recursions
    )

    width_fraction = (
        quantized_result[
            "lte"
        ][
            "active_fraction"
        ]
    )

    base_parameters = (
        scale_parameters[
            "unrolled_dense_equivalent_parameters"
        ]
    )

    dense_flops_proxy = (
        2
        * base_parameters
    )

    active_flops_proxy = (
        dense_flops_proxy
        * depth_fraction
        * width_fraction
    )

    float_ppl = (
        float_result[
            "quality"
        ][
            "perplexity"
        ]
    )

    quantized_ppl = (
        quantized_result[
            "quality"
        ][
            "perplexity"
        ]
    )

    report = {
        "identity": {
            "stage": "s11",
            "technique": "scaling_study",

            "scale": (
                cfg["experiment"][
                    "scale"
                ]
            ),

            "d_model": (
                model_cfg.d_model
            ),

            "checkpoint": str(
                checkpoint_path
            ),

            "device": str(
                device
            ),
        },

        "architecture": (
            model.architecture_report()
        ),

        "parameters": (
            model.parameter_report()
        ),

        "scale_parameters": (
            scale_parameters
        ),

        "grouping": grouping,

        "active_parameters": (
            active_parameters
        ),

        "compression": (
            compression
        ),

        "quality": (
            quantized_result[
                "quality"
            ]
        ),

        "routing": (
            quantized_result[
                "routing"
            ]
        ),

        "lte": (
            quantized_result[
                "lte"
            ]
        ),

        "quantization": (
            quantized_result[
                "quantization"
            ]
        ),

        "efficiency": (
            quantized_result[
                "efficiency"
            ]
        ),

        "sequence_length_results": (
            sequence_results
        ),

        "quantization_gap": {
            "float_perplexity": (
                float_ppl
            ),

            "quantized_perplexity": (
                quantized_ppl
            ),

            "relative_ppl_change": (
                (
                    quantized_ppl
                    - float_ppl
                )
                / float_ppl
            ),
        },

        "flops_proxy": {
            "type": (
                "parameter_based_theoretical_proxy"
            ),

            "dense_forward_flops_per_token": (
                dense_flops_proxy
            ),

            "depth_fraction": (
                depth_fraction
            ),

            "width_fraction": (
                width_fraction
            ),

            "theoretical_active_flops_per_token": (
                active_flops_proxy
            ),

            "warning": (
                "This is not hardware-profiler measured FLOPs. "
                "Dynamic routing and LTE currently use dense "
                "fallback execution."
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

            "training_seconds": (
                summary[
                    "training_seconds"
                ]
            ),
        },
    }

    save_json(
        run_dir
        / "evaluation.json",
        report,
    )

    print(
        "\nScaling evaluation completed."
    )

    print(
        "Base-equivalent parameters: "
        f"{base_parameters:,}"
    )

    print(
        "Stored FlexMamba parameters: "
        f"{scale_parameters['stored_parameters']:,}"
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
        "Average recursion: "
        f"{average_depth:.3f}"
    )

    print(
        "Active width: "
        f"{width_fraction * 100:.2f}%"
    )

    print(
        "Tokens/sec: "
        f"{report['efficiency']['tokens_per_second']:.2f}"
    )

    return report


# ============================================================
# STUDY SUMMARY
# ============================================================

def save_scaling_summary(
    cfg,
    results,
):

    root = Path(
        cfg["experiment"].get(
            "runs_root",
            "runs",
        )
    )

    if not root.is_absolute():
        root = PROJECT_ROOT / root

    study_root = (
        root
        / "scaling_study"
    )

    rows = []

    for result in results:

        evaluation = (
            result["evaluation"]
        )

        scale = (
            evaluation[
                "scale_parameters"
            ]
        )

        rows.append(
            {
                "scale": (
                    evaluation[
                        "identity"
                    ][
                        "scale"
                    ]
                ),

                "d_model": (
                    evaluation[
                        "identity"
                    ][
                        "d_model"
                    ]
                ),

                "base_equivalent_parameters": (
                    scale[
                        "unrolled_dense_equivalent_parameters"
                    ]
                ),

                "stored_parameters": (
                    scale[
                        "stored_parameters"
                    ]
                ),

                "stored_reduction": (
                    scale[
                        "stored_reduction_vs_unrolled"
                    ]
                ),

                "nll": (
                    evaluation[
                        "quality"
                    ][
                        "nll"
                    ]
                ),

                "perplexity": (
                    evaluation[
                        "quality"
                    ][
                        "perplexity"
                    ]
                ),

                "average_recursion": (
                    evaluation[
                        "routing"
                    ][
                        "average_recursion_depth"
                    ]
                ),

                "active_width": (
                    evaluation[
                        "lte"
                    ][
                        "active_fraction"
                    ]
                ),

                "tokens_per_second": (
                    evaluation[
                        "efficiency"
                    ][
                        "tokens_per_second"
                    ]
                ),

                "state_memory_reduction": (
                    evaluation[
                        "compression"
                    ][
                        "total_state_memory_reduction"
                    ]
                ),
            }
        )

    summary = {
        "section": (
            cfg["scaling"].get(
                "section_title",
                "FlexMamba Scale-Efficiency Analysis",
            )
        ),

        "target_basis": (
            cfg["scaling"][
                "target_basis"
            ]
        ),

        "budget": (
            cfg["budget"]
        ),

        "results": rows,
    }

    save_json(
        study_root
        / "scaling_summary.json",
        summary,
    )

    print(
        f"\nScaling summary saved: "
        f"{study_root / 'scaling_summary.json'}"
    )

    return summary