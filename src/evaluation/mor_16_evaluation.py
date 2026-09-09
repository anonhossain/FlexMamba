# src/evaluation/mor_16_evaluation.py
from src.evaluation.benchmark_eval import (
    run_benchmarks,
)
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.mor_16m import (
    MoRModelConfig,
    MoRCausalLM,
)

from src.training.common_16m import (
    PROJECT_ROOT,
    build_validation_batches,
    choose_device,
    save_json,
)


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


def model_weight_memory_mb(model):
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    return total_bytes / (1024 ** 2)


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


def get_state_dict(checkpoint):
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "model_state" in checkpoint:
        return checkpoint["model_state"]

    raise KeyError(
        "Checkpoint does not contain model state."
    )


@torch.inference_mode()
def run_evaluation(
    model,
    batches,
    device,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    top1_correct = 0
    top5_correct = 0

    total_forward_time = 0.0

    depth_sum = 0.0
    depth_count = 0

    depth_histogram = {
        str(i): 0
        for i in range(
            1,
            model.cfg.num_recursions + 1,
        )
    }

    balance_loss_sum = 0.0
    z_loss_sum = 0.0
    routing_batches = 0

    sampled_peak_memory = (
        current_memory_mb(device)
    )

    # ========================================================
    # WARMUP
    # ========================================================

    warmup = batches[0].to(device)

    _ = model(
        warmup[:, :-1]
    )

    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # EVALUATION
    # ========================================================

    for batch in batches:
        batch = batch.to(device)

        x = batch[:, :-1]
        y = batch[:, 1:]

        synchronize(device)

        start = time.perf_counter()

        output = model(x)

        synchronize(device)

        elapsed = (
            time.perf_counter()
            - start
        )

        logits = output["logits"]
        depths = output["depths"]

        total_forward_time += elapsed

        # ----------------------------------------------------
        # QUALITY
        # ----------------------------------------------------

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
        ).any(dim=-1).sum().item()

        # ----------------------------------------------------
        # RECURSION
        # ----------------------------------------------------

        depth_sum += depths.float().sum().item()
        depth_count += depths.numel()

        for depth in depth_histogram:
            depth_histogram[depth] += (
                depths == int(depth)
            ).sum().item()

        # ----------------------------------------------------
        # ROUTER
        # ----------------------------------------------------

        balance_loss, z_loss = (
            model.router_losses(
                output["router_logits"],
                output["router_probs"],
                depths,
            )
        )

        balance_loss_sum += (
            balance_loss.item()
        )

        z_loss_sum += (
            z_loss.item()
        )

        routing_batches += 1

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        memory = current_memory_mb(
            device
        )

        if memory is not None:
            sampled_peak_memory = max(
                sampled_peak_memory or 0,
                memory,
            )

    # ========================================================
    # RESULTS
    # ========================================================

    nll = (
        total_loss
        / total_tokens
    )

    avg_depth = (
        depth_sum
        / depth_count
    )

    depth_fraction = {
        depth: (
            count / depth_count
        )
        for depth, count
        in depth_histogram.items()
    }

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
    else:
        peak_memory = sampled_peak_memory

    return {
        "quality": {
            "validation_loss": nll,
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

        "efficiency": {
            "evaluation_tokens": total_tokens,
            "forward_seconds": (
                total_forward_time
            ),
            "tokens_per_second": (
                total_tokens
                / total_forward_time
            ),
            "milliseconds_per_token": (
                total_forward_time
                * 1000
                / total_tokens
            ),
            "peak_memory_mb": (
                peak_memory
            ),
        },

        "recursion": {
            "configured_max_recursions": (
                model.cfg.num_recursions
            ),
            "average_recursion_depth": (
                avg_depth
            ),
            "depth_histogram": (
                depth_histogram
            ),
            "depth_fraction": (
                depth_fraction
            ),
        },

        "routing": {
            "router_type": (
                model.cfg.router_type
            ),
            "average_balance_loss": (
                balance_loss_sum
                / routing_batches
            ),
            "average_z_loss": (
                z_loss_sum
                / routing_batches
            ),
        },
    }


def evaluate_mor(summary):
    checkpoint_path = resolve_path(
        summary["final_checkpoint"]
    )

    run_dir = resolve_path(
        summary["run_dir"]
    )

    device = choose_device()

    print(
        f"\nEvaluation device: "
        f"{device}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    if (
        cfg["technique"]["model_type"]
        != "mor"
    ):
        raise ValueError(
            "Checkpoint is not a MoR model."
        )

    model_cfg = MoRModelConfig(
        **cfg["model"]
    )

    model = MoRCausalLM(
        model_cfg
    )

    model.load_state_dict(
        get_state_dict(checkpoint),
        strict=True,
    )

    model = model.to(device)
    model.eval()

    # ========================================================
    # EVALUATION DATA
    # ========================================================

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

    # ========================================================
    # EVALUATE
    # ========================================================

    metrics = run_evaluation(
        model,
        batches,
        device,
    )
    benchmark_scores = None

    if cfg.get(
        "evaluation",
        {}
    ).get(
        "run_benchmarks",
        False,
    ):
        print(
            "\nRunning downstream "
            "benchmark evaluation..."
        )

        benchmark_scores = run_benchmarks(
            model=model,
            cfg=cfg,
            device=device,
        )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "identity": {
            "technique": (
                cfg["technique"]["name"]
            ),
            "model_type": "mor",
            "scale": (
                cfg["experiment"].get(
                    "scale"
                )
            ),
            "run": summary["run"],
            "checkpoint": (
                str(checkpoint_path)
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

            "n_heads": (
                model_cfg.n_heads
            ),

            "n_kv_heads": (
                model_cfg.n_kv_heads
            ),

            "d_ff": (
                model_cfg.d_ff
            ),

            "max_seq_len": (
                model_cfg.max_seq_len
            ),
        },

        "parameters": (
            model.parameter_report()
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
                summary[
                    "validation_history"
                ]
            ),
        },

        "quality": (
            metrics["quality"]
        ),

        "efficiency": (
            metrics["efficiency"]
        ),

        "recursion": (
            metrics["recursion"]
        ),

        "routing": {
            **metrics["routing"],

            "router_hidden_mult": (
                model_cfg.router_hidden_mult
            ),

            "router_balance_coeff": (
                model_cfg.router_balance_coeff
            ),

            "router_z_loss_coeff": (
                model_cfg.router_z_loss_coeff
            ),
        },

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
        
        "benchmarks": benchmark_scores,

        "evaluation_setup": {
            "evaluation_batches": (
                len(batches)
            ),

            "batch_size": (
                data_cfg[
                    "training"
                ]["batch_size"]
            ),

            "sequence_length": (
                model_cfg.max_seq_len
            ),

            "same_validation_split": True,
        },
    }

    save_json(
        run_dir / "evaluation.json",
        report,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\nMoR evaluation completed.")

    print(
        f"NLL: "
        f"{report['quality']['nll']:.4f}"
    )

    print(
        f"PPL: "
        f"{report['quality']['perplexity']:.2f}"
    )

    print(
        f"Average depth: "
        f"{report['recursion']['average_recursion_depth']:.3f}"
    )

    print(
        f"Tokens/sec: "
        f"{report['efficiency']['tokens_per_second']:.2f}"
    )

    print(
        f"Saved: "
        f"{run_dir / 'evaluation.json'}"
    )

    return report