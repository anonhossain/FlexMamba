import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.lte_adaptive_width_model import (
    LTEAdaptiveWidthModelConfig,
    LTEAdaptiveWidthCausalLM,
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
        return (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        )

    if device.type == "mps":
        try:
            return (
                torch.mps.current_allocated_memory()
                / (1024 ** 2)
            )
        except Exception:
            return None

    return None


def model_weight_memory_mb(model):

    total = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    return total / (1024 ** 2)


# ============================================================
# CORE EVALUATION
# ============================================================

@torch.inference_mode()
def run_lte_evaluation(
    model,
    batches,
    device,
):

    model.eval()

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

    lte_hard_active_bins = 0.0
    lte_soft_active_bins = 0.0
    lte_token_decisions = 0

    lte_efficiency_loss_sum = 0.0
    lte_batches = 0

    layer_hard_fraction_sum = {}
    layer_batch_count = {}

    sampled_peak_memory = (
        current_memory_mb(device)
    )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    warmup = batches[0].to(device)

    model(
        warmup[:, :-1]
    )

    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    for batch in batches:

        batch = batch.to(device)

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
        depths = output["routing_depths"]

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
            top5
            == y.unsqueeze(-1)
        ).any(dim=-1).sum().item()

        # ----------------------------------------------------
        # DEPTH ROUTING
        # ----------------------------------------------------

        for depth in depth_counts:

            depth_counts[depth] += (
                depths == depth
            ).sum().item()

        routing_entropy_sum += (
            output[
                "routing_entropy"
            ].item()
        )

        routing_batches += 1

        # ----------------------------------------------------
        # LTE
        # ----------------------------------------------------

        lte_hard_active_bins += (
            output[
                "lte_active_bin_count"
            ].item()
        )

        lte_token_decisions += (
            output[
                "lte_token_decisions"
            ]
        )

        soft_fraction = (
            output[
                "lte_soft_active_fraction"
            ].item()
        )

        lte_soft_active_bins += (
            soft_fraction
            * output[
                "lte_token_decisions"
            ]
            * model.lte_cfg[
                "num_bins"
            ]
        )

        lte_efficiency_loss_sum += (
            output[
                "lte_efficiency_loss"
            ].item()
        )

        lte_batches += 1

        for layer_index, stats in (
            output[
                "lte_layer_stats"
            ].items()
        ):

            layer_hard_fraction_sum[
                layer_index
            ] = (
                layer_hard_fraction_sum.get(
                    layer_index,
                    0.0,
                )
                + stats[
                    "active_fraction"
                ].item()
            )

            layer_batch_count[
                layer_index
            ] = (
                layer_batch_count.get(
                    layer_index,
                    0,
                )
                + 1
            )

        memory = current_memory_mb(
            device
        )

        if memory is not None:

            sampled_peak_memory = max(
                sampled_peak_memory or 0,
                memory,
            )

    # ========================================================
    # QUALITY
    # ========================================================

    nll = (
        total_loss
        / total_tokens
    )

    # ========================================================
    # DEPTH ROUTING
    # ========================================================

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

    active_fraction_by_recursion = {}

    for recursion in range(
        1,
        model.max_recursions + 1,
    ):

        active = sum(
            count
            for depth, count
            in depth_counts.items()
            if depth >= recursion
        )

        active_fraction_by_recursion[
            str(recursion)
        ] = (
            active
            / routing_tokens
        )

    average_effective_layers = (
        model.cfg.input_layers
        + model.cfg.shared_middle_layers
        * average_depth
        + model.cfg.output_layers
    )

    recursive_compute_fraction = (
        average_depth
        / model.max_recursions
    )

    # ========================================================
    # LTE WIDTH
    # ========================================================

    num_bins = (
        model.lte_cfg[
            "num_bins"
        ]
    )

    average_active_bins = (
        lte_hard_active_bins
        / lte_token_decisions
    )

    lte_active_fraction = (
        average_active_bins
        / num_bins
    )

    soft_active_fraction = (
        lte_soft_active_bins
        / (
            lte_token_decisions
            * num_bins
        )
    )

    per_layer_active_fraction = {
        layer: (
            layer_hard_fraction_sum[
                layer
            ]
            / layer_batch_count[
                layer
            ]
        )
        for layer
        in layer_hard_fraction_sum
    }

    # Approximate joint recursive-core fraction.
    theoretical_joint_fraction = (
        recursive_compute_fraction
        * lte_active_fraction
    )

    # ========================================================
    # MEMORY
    # ========================================================

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
            "validation_loss": nll,
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
            "router_type": (
                "token_choice"
            ),

            "average_recursion_depth": (
                average_depth
            ),

            "depth_counts": (
                depth_counts
            ),

            "depth_fractions": (
                depth_fractions
            ),

            "active_fraction_by_recursion": (
                active_fraction_by_recursion
            ),

            "average_effective_layers": (
                average_effective_layers
            ),

            "average_router_entropy": (
                routing_entropy_sum
                / routing_batches
            ),

            "theoretical_recursive_compute_fraction": (
                recursive_compute_fraction
            ),

            "routing_compute_savings_realized": False,
        },

        "lte": {
            "num_bins": num_bins,

            "minimum_active_bins": (
                model.lte_cfg[
                    "minimum_active_bins"
                ]
            ),

            "maximum_active_bins": (
                model.lte_cfg[
                    "maximum_active_bins"
                ]
            ),

            "target_active_fraction": (
                model.lte_cfg[
                    "target_active_fraction"
                ]
            ),

            "average_active_bins": (
                average_active_bins
            ),

            "active_fraction": (
                lte_active_fraction
            ),

            "soft_active_fraction": (
                soft_active_fraction
            ),

            "theoretical_width_reduction": (
                1.0
                - lte_active_fraction
            ),

            "average_efficiency_loss": (
                lte_efficiency_loss_sum
                / lte_batches
            ),

            "per_layer_active_fraction": (
                per_layer_active_fraction
            ),

            "dense_compute_fallback": True,

            "sparse_compute_savings_realized": False,
        },

        "compute": {
            "recursive_core_depth_fraction": (
                recursive_compute_fraction
            ),

            "recursive_core_width_fraction": (
                lte_active_fraction
            ),

            "theoretical_joint_recursive_width_fraction": (
                theoretical_joint_fraction
            ),

            "theoretical_joint_recursive_width_reduction": (
                1.0
                - theoretical_joint_fraction
            ),

            "realized_sparse_kernel": False,

            "note": (
                "LTE masks channel contribution but the "
                "current generic implementation still performs "
                "dense Mamba operations. Width FLOP savings are "
                "therefore theoretical."
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
# FULL EVALUATION
# ============================================================

def evaluate_lte_adaptive_width(
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

    cfg = checkpoint["config"]

    if (
        cfg["technique"]["model_type"]
        != "lte_dynamic_mamba"
    ):
        raise ValueError(
            "Checkpoint is not "
            "LTE Dynamic Mamba."
        )

    model_cfg = (
        LTEAdaptiveWidthModelConfig(
            **cfg["model"]
        )
    )

    model = LTEAdaptiveWidthCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["lte"],
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

    data_cfg = copy.deepcopy(
        cfg
    )

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
            cfg["training"][
                "batch_size"
            ],
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

    active_parameters = (
        model.active_parameter_report(
            metrics[
                "lte"
            ][
                "active_fraction"
            ]
        )
    )

    benchmark_scores = None

    if eval_cfg.get(
        "run_benchmarks",
        False,
    ):

        from src.evaluation.benchmark_eval import (
            run_benchmarks,
        )

        benchmark_scores = (
            run_benchmarks(
                model,
                cfg,
                device,
            )
        )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    state_report = model.state_report(
        batch_size=data_cfg[
            "training"
        ][
            "batch_size"
        ],

        dtype=next(
            model.parameters()
        ).dtype,
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

        "parameters": (
            model.parameter_report()
        ),

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

        "state": (
            state_report
        ),

        "routing": (
            metrics[
                "routing"
            ]
        ),

        "lte": (
            metrics[
                "lte"
            ]
        ),

        "compute": (
            metrics[
                "compute"
            ]
        ),

        "quality": (
            metrics[
                "quality"
            ]
        ),

        "efficiency": (
            metrics[
                "efficiency"
            ]
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
        run_dir
        / "evaluation.json",
        report,
    )

    print(
        "\nLTE Adaptive Width "
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
        "Theoretical width reduction: "
        f"{report['lte']['theoretical_width_reduction'] * 100:.2f}%"
    )

    print(
        "Theoretical active parameters: "
        f"{report['active_parameters']['theoretical_active_parameters']:,.0f}"
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