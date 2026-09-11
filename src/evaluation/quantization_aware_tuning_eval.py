import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.quantization_aware_tuning_model import (
    QuantizationAwareTuningModelConfig,
    QuantizationAwareTuningCausalLM,
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


def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


def current_memory_mb(device):

    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)

    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            return None

    return None


# ============================================================
# CORE EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_qat_batches(
    model,
    batches,
    device,
    state_mode,
    sequence_tokens,
):

    model.eval()
    model.set_state_mode(state_mode)

    total_loss = 0.0
    total_tokens = 0

    top1_correct = 0
    top5_correct = 0

    total_time = 0.0

    depth_counts = {
        depth: 0
        for depth in range(
            model.min_recursions,
            model.max_recursions + 1,
        )
    }

    lte_active_bins = 0.0
    lte_token_decisions = 0

    quant_mse = 0.0
    quant_mae = 0.0
    quant_clip = 0.0
    quant_batches = 0

    peak_memory = current_memory_mb(
        device
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for batch in batches:

        batch = batch[
            :,
            :sequence_tokens + 1,
        ].to(device)

        x = batch[:, :-1]
        y = batch[:, 1:]

        synchronize(device)
        start = time.perf_counter()

        output = model(x)

        synchronize(device)

        total_time += (
            time.perf_counter()
            - start
        )

        logits = output["logits"]

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            y.reshape(-1),
            reduction="sum",
        )

        tokens = y.numel()

        total_loss += loss.item()
        total_tokens += tokens

        predictions = logits.argmax(
            dim=-1
        )

        top1_correct += (
            predictions == y
        ).sum().item()

        top5 = logits.topk(
            5,
            dim=-1,
        ).indices

        top5_correct += (
            top5 == y.unsqueeze(-1)
        ).any(
            dim=-1
        ).sum().item()

        depths = output[
            "routing_depths"
        ]

        for depth in depth_counts:
            depth_counts[depth] += (
                depths == depth
            ).sum().item()

        lte_active_bins += (
            output[
                "lte_active_bin_count"
            ].item()
        )

        lte_token_decisions += (
            output[
                "lte_token_decisions"
            ]
        )

        quant_mse += (
            output[
                "state_quantization_mse"
            ]
        )

        quant_mae += (
            output[
                "state_quantization_mae"
            ]
        )

        quant_clip += (
            output[
                "state_quantization_clip_fraction"
            ]
        )

        quant_batches += 1

        memory = current_memory_mb(
            device
        )

        if memory is not None:
            peak_memory = max(
                peak_memory or 0,
                memory,
            )

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    nll = (
        total_loss
        / total_tokens
    )

    routing_tokens = sum(
        depth_counts.values()
    )

    depth_fractions = {
        str(depth): (
            count
            / routing_tokens
        )
        for depth, count
        in depth_counts.items()
    }

    average_depth = (
        sum(
            depth * count
            for depth, count
            in depth_counts.items()
        )
        / routing_tokens
    )

    average_active_bins = (
        lte_active_bins
        / lte_token_decisions
    )

    return {
        "quality": {
            "nll": nll,

            "perplexity": math.exp(
                min(nll, 20.0)
            ),

            "bits_per_token": (
                nll / math.log(2)
            ),

            "token_accuracy": (
                top1_correct
                / total_tokens
            ),

            "top_5_token_accuracy": (
                top5_correct
                / total_tokens
            ),
        },

        "routing": {
            "average_recursion_depth": (
                average_depth
            ),

            "depth_counts": (
                depth_counts
            ),

            "depth_fractions": (
                depth_fractions
            ),
        },

        "lte": {
            "average_active_bins": (
                average_active_bins
            ),

            "active_fraction": (
                average_active_bins
                / model.lte_cfg[
                    "num_bins"
                ]
            ),
        },

        "quantization": {
            "mode": state_mode,

            "mse": (
                quant_mse
                / quant_batches
            ),

            "mae": (
                quant_mae
                / quant_batches
            ),

            "clip_fraction": (
                quant_clip
                / quant_batches
            ),
        },

        "efficiency": {
            "evaluation_tokens": (
                total_tokens
            ),

            "forward_seconds": (
                total_time
            ),

            "tokens_per_second": (
                total_tokens
                / total_time
            ),

            "milliseconds_per_token": (
                total_time
                * 1000
                / total_tokens
            ),

            "peak_memory_mb": (
                peak_memory
            ),
        },
    }


# ============================================================
# LOAD S08 PARENT INTO QAT SIMULATOR
# ============================================================

def build_parent_model(
    cfg,
    parent_checkpoint_path,
):

    parent_checkpoint = torch.load(
        parent_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_cfg = (
        QuantizationAwareTuningModelConfig(
            **cfg["model"]
        )
    )

    model = (
        QuantizationAwareTuningCausalLM(
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

    model.load_parent_state_dict(
        parent_checkpoint[
            "model_state_dict"
        ],
        parent_checkpoint[
            "quantization_state"
        ],
    )

    return model


# ============================================================
# FULL S09 EVALUATION
# ============================================================

def evaluate_quantization_aware_tuning(
    summary,
):

    checkpoint_path = resolve_path(
        summary[
            "final_checkpoint"
        ]
    )

    parent_checkpoint_path = resolve_path(
        summary[
            "parent_checkpoint"
        ]
    )

    run_dir = resolve_path(
        summary["run_dir"]
    )

    device = choose_device()

    print(
        f"\nEvaluation device: {device}"
    )

    print(
        f"QAT checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    if (
        cfg["technique"]["model_type"]
        != "qat_joint_mamba"
    ):
        raise ValueError(
            "Checkpoint is not QAT Joint Mamba."
        )

    model_cfg = (
        QuantizationAwareTuningModelConfig(
            **cfg["model"]
        )
    )

    model = (
        QuantizationAwareTuningCausalLM(
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

    data_cfg = copy.deepcopy(
        cfg
    )

    data_cfg["validation"] = {
        "enabled": True,

        "eval_batches": eval_cfg.get(
            "eval_batches",
            1,
        ),
    }

    data_cfg["training"][
        "batch_size"
    ] = eval_cfg.get(
        "batch_size",
        cfg["training"][
            "batch_size"
        ],
    )

    batches = build_validation_batches(
        data_cfg,
        model_cfg,
    )

    # --------------------------------------------------------
    # TUNED MODEL: FP RECURRENT STATE
    # --------------------------------------------------------

    print(
        "\nRunning S09 FP-state evaluation..."
    )

    tuned_float = evaluate_qat_batches(
        model,
        batches,
        device,
        state_mode="float",
        sequence_tokens=sequence_tokens,
    )

    # --------------------------------------------------------
    # TUNED MODEL: QUANTIZED RECURRENT STATE
    # --------------------------------------------------------

    print(
        "Running S09 INT8-state evaluation..."
    )

    tuned_quantized = (
        evaluate_qat_batches(
            model,
            batches,
            device,
            state_mode="quantized",
            sequence_tokens=sequence_tokens,
        )
    )

    # --------------------------------------------------------
    # S08 PARENT USING THE SAME CORRECT RECURRENT SIMULATOR
    # --------------------------------------------------------

    print(
        "Running S08 parent through corrected "
        "INT8 recurrent-state evaluation..."
    )

    parent_model = build_parent_model(
        cfg,
        parent_checkpoint_path,
    )

    parent_model = (
        parent_model.to(device)
    )

    parent_quantized = (
        evaluate_qat_batches(
            parent_model,
            batches,
            device,
            state_mode="quantized",
            sequence_tokens=sequence_tokens,
        )
    )

    # --------------------------------------------------------
    # DIFFERENCES
    # --------------------------------------------------------

    tuned_ppl_delta = (
        tuned_quantized[
            "quality"
        ][
            "perplexity"
        ]
        - tuned_float[
            "quality"
        ][
            "perplexity"
        ]
    )

    tuned_relative_ppl_delta = (
        tuned_ppl_delta
        / tuned_float[
            "quality"
        ][
            "perplexity"
        ]
    )

    qat_ppl_improvement = (
        parent_quantized[
            "quality"
        ][
            "perplexity"
        ]
        - tuned_quantized[
            "quality"
        ][
            "perplexity"
        ]
    )

    qat_relative_improvement = (
        qat_ppl_improvement
        / parent_quantized[
            "quality"
        ][
            "perplexity"
        ]
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

    report = {
        "identity": {
            "technique": (
                cfg["technique"][
                    "name"
                ]
            ),

            "model_type": (
                cfg["technique"][
                    "model_type"
                ]
            ),

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

        "compression": (
            compression
        ),

        "parent_quantized": (
            parent_quantized
        ),

        "tuned_float": (
            tuned_float
        ),

        "tuned_quantized": (
            tuned_quantized
        ),

        "quality": (
            tuned_quantized[
                "quality"
            ]
        ),

        "quantization": (
            tuned_quantized[
                "quantization"
            ]
        ),

        "quality_change": {
            "quantized_vs_float_ppl_change": (
                tuned_ppl_delta
            ),

            "quantized_vs_float_relative_change": (
                tuned_relative_ppl_delta
            ),

            "qat_ppl_improvement_vs_parent": (
                qat_ppl_improvement
            ),

            "qat_relative_improvement_vs_parent": (
                qat_relative_improvement
            ),
        },

        "routing": (
            tuned_quantized[
                "routing"
            ]
        ),

        "lte": (
            tuned_quantized[
                "lte"
            ]
        ),

        "efficiency": (
            tuned_quantized[
                "efficiency"
            ]
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

            "sequence_tokens": (
                sequence_tokens
            ),
        },

        "research_note": {
            "turboquant_original_qat": False,

            "description": (
                "S09 is a FlexMamba extension that performs "
                "task-level quantization-aware tuning against "
                "fake-quantized recurrent Mamba SSM states."
            ),
        },
    }

    save_json(
        run_dir
        / "evaluation.json",
        report,
    )

    print(
        "\nQuantization-Aware Tuning evaluation completed."
    )

    print(
        f"S08 corrected INT8 PPL: "
        f"{parent_quantized['quality']['perplexity']:.2f}"
    )

    print(
        f"S09 FP PPL: "
        f"{tuned_float['quality']['perplexity']:.2f}"
    )

    print(
        f"S09 INT8 PPL: "
        f"{tuned_quantized['quality']['perplexity']:.2f}"
    )

    print(
        "S09 INT8 vs FP PPL change: "
        f"{tuned_relative_ppl_delta * 100:.3f}%"
    )

    print(
        "QAT improvement vs S08 parent: "
        f"{qat_relative_improvement * 100:.3f}%"
    )

    print(
        "State quantization MSE: "
        f"{tuned_quantized['quantization']['mse']:.8f}"
    )

    print(
        "State clip fraction: "
        f"{tuned_quantized['quantization']['clip_fraction'] * 100:.4f}%"
    )

    print(
        "Total state compression: "
        f"{compression['total_state_compression_ratio']:.2f}x"
    )

    print(
        "State memory reduction: "
        f"{compression['total_state_memory_reduction'] * 100:.2f}%"
    )

    print(
        f"Saved: "
        f"{run_dir / 'evaluation.json'}"
    )

    return report