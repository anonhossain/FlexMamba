import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.turbo_state_compression_model import (
    TurboStateCompressionModelConfig,
    TurboStateCompressionCausalLM,
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
# CALIBRATION
# ============================================================

@torch.inference_mode()
def calibrate_state_quantization(
    model,
    batches,
    device,
    chunk_size,
):

    print(
        "\nCalibrating recurrent SSM state quantizers..."
    )

    model.eval()
    model.reset_state_calibration()

    total_chunks = 0

    for batch in batches:

        batch = batch.to(device)

        x = batch[:, :-1]

        state_bank = None

        for start in range(
            0,
            x.size(1),
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                x.size(1),
            )

            output = model.forward_chunk(
                x[:, start:end],
                state_bank=state_bank,
            )

            state_bank = output[
                "state_bank"
            ]

            total_chunks += 1

    model.finalize_state_calibration()

    scale_values = []

    for quantizer in (
        model._state_quantizers.values()
    ):

        scale_values.append(
            quantizer.scale.float().mean().item()
        )

    report = {
        "calibration_batches": len(batches),
        "calibration_chunks": total_chunks,
        "chunk_size": chunk_size,

        "quantizers": len(
            model._state_quantizers
        ),

        "method": model.state_quant_cfg[
            "calibration_method"
        ],

        "percentile": model.state_quant_cfg[
            "calibration_percentile"
        ],

        "average_scale": (
            sum(scale_values)
            / len(scale_values)
        ),
    }

    print(
        f"Calibration completed: "
        f"{report['quantizers']} state quantizers."
    )

    return report


# ============================================================
# CHUNKED EVALUATION
# ============================================================

@torch.inference_mode()
def run_chunked_evaluation(
    model,
    batches,
    device,
    chunk_size,
    quantized,
):

    model.eval()

    model.set_state_quantization_mode(
        "quantized"
        if quantized
        else "float"
    )

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

    routing_entropy_sum = 0.0
    routing_batches = 0

    lte_active_bins = 0.0
    lte_token_decisions = 0

    sampled_peak_memory = (
        current_memory_mb(device)
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for batch in batches:

        batch = batch.to(device)

        x = batch[:, :-1]
        y = batch[:, 1:]

        state_bank = None

        for start in range(
            0,
            x.size(1),
            chunk_size,
        ):

            end = min(
                start + chunk_size,
                x.size(1),
            )

            x_chunk = x[:, start:end]
            y_chunk = y[:, start:end]

            synchronize(device)

            start_time = time.perf_counter()

            output = model.forward_chunk(
                x_chunk,
                state_bank=state_bank,
            )

            synchronize(device)

            total_time += (
                time.perf_counter()
                - start_time
            )

            state_bank = output[
                "state_bank"
            ]

            logits = output[
                "logits"
            ]

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                y_chunk.reshape(-1),
                reduction="sum",
            )

            tokens = y_chunk.numel()

            total_loss += loss.item()
            total_tokens += tokens

            predictions = logits.argmax(
                dim=-1
            )

            top1_correct += (
                predictions
                == y_chunk
            ).sum().item()

            top5 = logits.topk(
                5,
                dim=-1,
            ).indices

            top5_correct += (
                top5
                == y_chunk.unsqueeze(-1)
            ).any(
                dim=-1
            ).sum().item()

            depths = output[
                "routing_depths"
            ]

            for depth in depth_counts:

                depth_counts[depth] += (
                    depths
                    == depth
                ).sum().item()

            routing_entropy_sum += (
                output[
                    "routing_entropy"
                ].item()
            )

            routing_batches += 1

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

            memory = current_memory_mb(
                device
            )

            if memory is not None:

                sampled_peak_memory = max(
                    sampled_peak_memory or 0,
                    memory,
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

    active_width_fraction = (
        average_active_bins
        / model.lte_cfg[
            "num_bins"
        ]
    )

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    else:

        peak_memory = (
            sampled_peak_memory
        )

    return {
        "quality": {
            "nll": nll,

            "perplexity": math.exp(
                min(nll, 20.0)
            ),

            "bits_per_token": (
                nll
                / math.log(2)
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

            "average_router_entropy": (
                routing_entropy_sum
                / routing_batches
            ),
        },

        "lte": {
            "average_active_bins": (
                average_active_bins
            ),

            "active_fraction": (
                active_width_fraction
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
# FULL S08 EVALUATION
# ============================================================

def evaluate_turbo_state_compression(
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

    cfg = checkpoint[
        "config"
    ]

    if (
        cfg["technique"][
            "model_type"
        ]
        != "turboquant_mamba"
    ):
        raise ValueError(
            "Checkpoint is not TurboQuant State Mamba."
        )

    model_cfg = (
        TurboStateCompressionModelConfig(
            **cfg["model"]
        )
    )

    model = TurboStateCompressionCausalLM(
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
        seed=cfg[
            "experiment"
        ]["seed"],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.load_quantization_state_dict(
        checkpoint[
            "quantization_state"
        ]
    )

    model = model.to(device)
    model.eval()

    eval_cfg = cfg.get(
        "evaluation",
        {},
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

    data_cfg[
        "training"
    ][
        "batch_size"
    ] = eval_cfg.get(
        "batch_size",
        cfg[
            "training"
        ][
            "batch_size"
        ],
    )

    batches = build_validation_batches(
        data_cfg,
        model_cfg,
    )

    chunk_size = eval_cfg.get(
        "chunk_size",
        cfg[
            "state_quantization"
        ].get(
            "evaluation_chunk_size",
            64,
        ),
    )

    print(
        "\nRunning FP recurrent-state baseline..."
    )

    baseline = run_chunked_evaluation(
        model,
        batches,
        device,
        chunk_size,
        quantized=False,
    )

    print(
        "Running INT8 recurrent-state evaluation..."
    )

    quantized = run_chunked_evaluation(
        model,
        batches,
        device,
        chunk_size,
        quantized=True,
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

    ppl_change = (
        quantized[
            "quality"
        ][
            "perplexity"
        ]
        - baseline[
            "quality"
        ][
            "perplexity"
        ]
    )

    relative_ppl_change = (
        ppl_change
        / baseline[
            "quality"
        ][
            "perplexity"
        ]
    )

    nll_change = (
        quantized[
            "quality"
        ][
            "nll"
        ]
        - baseline[
            "quality"
        ][
            "nll"
        ]
    )

    throughput_change = (
        quantized[
            "efficiency"
        ][
            "tokens_per_second"
        ]
        / baseline[
            "efficiency"
        ][
            "tokens_per_second"
        ]
        - 1.0
    )

    report = {
        "identity": {
            "technique": (
                cfg[
                    "technique"
                ][
                    "name"
                ]
            ),

            "model_type": (
                cfg[
                    "technique"
                ][
                    "model_type"
                ]
            ),

            "scale": (
                cfg[
                    "experiment"
                ][
                    "scale"
                ]
            ),

            "run": (
                summary["run"]
            ),

            "checkpoint": str(
                checkpoint_path
            ),

            "parent_checkpoint": (
                summary.get(
                    "parent_checkpoint"
                )
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

        "calibration": (
            summary[
                "calibration"
            ]
        ),

        "compression": (
            compression
        ),

        "baseline": (
            baseline
        ),

        "quantized": (
            quantized
        ),

        "quality": (
            quantized[
                "quality"
            ]
        ),

        "quality_change": {
            "nll_change": (
                nll_change
            ),

            "perplexity_change": (
                ppl_change
            ),

            "relative_perplexity_change": (
                relative_ppl_change
            ),
        },

        "routing": (
            quantized[
                "routing"
            ]
        ),

        "lte": (
            quantized[
                "lte"
            ]
        ),

        "efficiency": (
            quantized[
                "efficiency"
            ]
        ),

        "throughput_change": (
            throughput_change
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

            "chunk_size": (
                chunk_size
            ),
        },

        "limitations": {
            "exact_turboquant_reproduction": False,

            "note": (
                "This is a TurboQuant-inspired adaptation: "
                "randomized Hadamard rotation plus calibrated "
                "symmetric scalar quantization is applied to "
                "Mamba recurrent SSM state rather than Transformer KV cache."
            ),
        },
    }

    save_json(
        run_dir
        / "evaluation.json",
        report,
    )

    print(
        "\nTurboQuant State Compression evaluation completed."
    )

    print(
        f"Baseline NLL: "
        f"{baseline['quality']['nll']:.4f}"
    )

    print(
        f"INT8 NLL: "
        f"{quantized['quality']['nll']:.4f}"
    )

    print(
        f"Baseline PPL: "
        f"{baseline['quality']['perplexity']:.2f}"
    )

    print(
        f"INT8 PPL: "
        f"{quantized['quality']['perplexity']:.2f}"
    )

    print(
        "Relative PPL change: "
        f"{relative_ppl_change * 100:.3f}%"
    )

    print(
        "Recurrent state compression: "
        f"{compression['recurrent_payload_compression_ratio']:.2f}x"
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
        "Quantized tokens/sec: "
        f"{quantized['efficiency']['tokens_per_second']:.2f}"
    )

    print(
        f"Saved: "
        f"{run_dir / 'evaluation.json'}"
    )

    return report