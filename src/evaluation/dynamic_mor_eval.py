import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.dynamic_mor_models import (
    DynamicMoRModelConfig,
    DynamicMoRCausalLM,
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


def model_weight_memory_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 ** 2)


def current_memory_mb(device):

    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)

    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            return None

    return None


@torch.inference_mode()
def run_evaluation(model, batches, device):

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    total_time = 0.0

    depth_counts = {
        depth: 0
        for depth in range(model.min_recursions, model.max_recursions + 1)
    }

    entropy_sum = 0.0
    entropy_batches = 0
    sampled_peak_memory = current_memory_mb(device)

    warmup = batches[0].to(device)
    model(warmup[:, :-1])
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for batch in batches:

        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        synchronize(device)
        start = time.perf_counter()

        output = model(x)

        synchronize(device)
        total_time += time.perf_counter() - start

        logits = output["logits"]
        depths = output["routing_depths"]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        tokens = y.numel()

        total_loss += loss.item()
        total_tokens += tokens

        predictions = logits.argmax(dim=-1)
        top1_correct += (predictions == y).sum().item()

        top5 = logits.topk(5, dim=-1).indices
        top5_correct += (top5 == y.unsqueeze(-1)).any(dim=-1).sum().item()

        for depth in depth_counts:
            depth_counts[depth] += (depths == depth).sum().item()

        entropy_sum += output["routing_entropy"].item()
        entropy_batches += 1

        memory = current_memory_mb(device)

        if memory is not None:
            sampled_peak_memory = max(sampled_peak_memory or 0, memory)

    nll = total_loss / total_tokens

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = sampled_peak_memory

    routing_tokens = sum(depth_counts.values())

    depth_fractions = {
        str(depth): count / routing_tokens
        for depth, count in depth_counts.items()
    }

    avg_depth = sum(
        depth * count
        for depth, count in depth_counts.items()
    ) / routing_tokens

    active_fraction_by_recursion = {}

    for recursion in range(1, model.max_recursions + 1):

        active = sum(
            count
            for depth, count in depth_counts.items()
            if depth >= recursion
        )

        active_fraction_by_recursion[str(recursion)] = active / routing_tokens

    avg_effective_layers = (
        model.cfg.input_layers
        + model.cfg.shared_middle_layers * avg_depth
        + model.cfg.output_layers
    )

    theoretical_recursive_fraction = avg_depth / model.max_recursions

    return {
        "quality": {
            "validation_loss": nll,
            "nll": nll,
            "perplexity": math.exp(min(nll, 20.0)),
            "bits_per_token": nll / math.log(2),
            "token_accuracy": top1_correct / total_tokens,
            "top_5_token_accuracy": top5_correct / total_tokens,
        },

        "routing": {
            "router_type": "token_choice",
            "average_recursion_depth": avg_depth,
            "depth_counts": depth_counts,
            "depth_fractions": depth_fractions,
            "active_fraction_by_recursion": active_fraction_by_recursion,
            "average_effective_layers": avg_effective_layers,
            "average_router_entropy": entropy_sum / entropy_batches,

            "theoretical_recursive_compute_fraction": theoretical_recursive_fraction,
            "theoretical_recursive_compute_reduction": 1 - theoretical_recursive_fraction,

            "routing_compute_savings_realized": False,
            "note": "Current dense fallback computes each recursive Mamba cycle before masking token updates.",
        },

        "efficiency": {
            "evaluation_tokens": total_tokens,
            "forward_seconds": total_time,
            "tokens_per_second": total_tokens / total_time,
            "milliseconds_per_token": total_time * 1000 / total_tokens,
            "peak_memory_mb": peak_memory,
        },
    }


def evaluate_dynamic_mor(summary):

    checkpoint_path = resolve_path(summary["final_checkpoint"])
    run_dir = resolve_path(summary["run_dir"])
    device = choose_device()

    print(f"\nEvaluation device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    if cfg["technique"]["model_type"] != "dynamic_recursive_mamba":
        raise ValueError("Checkpoint is not Dynamic MoR Mamba.")

    model_cfg = DynamicMoRModelConfig(**cfg["model"])

    model = DynamicMoRCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    eval_cfg = cfg.get("evaluation", {})
    data_cfg = copy.deepcopy(cfg)

    data_cfg["validation"] = {
        "enabled": True,
        "eval_batches": eval_cfg.get(
            "eval_batches",
            cfg.get("validation", {}).get("eval_batches", 20),
        ),
    }

    data_cfg["training"]["batch_size"] = eval_cfg.get(
        "batch_size",
        cfg["training"]["batch_size"],
    )

    batches = build_validation_batches(data_cfg, model_cfg)
    metrics = run_evaluation(model, batches, device)

    benchmark_scores = None

    if eval_cfg.get("run_benchmarks", False):

        from src.evaluation.benchmark_eval import run_benchmarks

        print("\nRunning downstream benchmarks...")
        benchmark_scores = run_benchmarks(model, cfg, device)

    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)

    state_report = model.state_report(
        batch_size=data_cfg["training"]["batch_size"],
        dtype=next(model.parameters()).dtype,
    )

    report = {
        "identity": {
            "technique": cfg["technique"]["name"],
            "model_type": cfg["technique"]["model_type"],
            "scale": cfg["experiment"]["scale"],
            "run": summary["run"],
            "checkpoint": str(checkpoint_path),
            "parent_checkpoint": summary.get("parent_checkpoint"),
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

        "parameters": model.parameter_report(),
        "initialization": summary.get("initialization_report"),

        "training": {
            "completed_steps": summary["completed_steps"],
            "training_tokens": summary["training_tokens"],
            "final_training_loss": summary["final_training_loss"],
            "training_seconds": summary["training_seconds"],
            "training_tokens_per_second": summary["tokens_per_second"],
            "validation_history": summary["validation_history"],
        },

        "state": state_report,
        "routing": metrics["routing"],
        "quality": metrics["quality"],
        "efficiency": metrics["efficiency"],

        "storage": {
            "model_weight_memory_mb": model_weight_memory_mb(model),
            "checkpoint_size_mb": checkpoint_size_mb,
        },

        "benchmarks": benchmark_scores,

        "evaluation_setup": {
            "evaluation_batches": len(batches),
            "batch_size": data_cfg["training"]["batch_size"],
            "sequence_length": model_cfg.max_seq_len,
        },
    }

    save_json(run_dir / "evaluation.json", report)

    print("\nDynamic MoR evaluation completed.")
    print(f"NLL: {report['quality']['nll']:.4f}")
    print(f"PPL: {report['quality']['perplexity']:.2f}")
    print(f"Average recursion: {report['routing']['average_recursion_depth']:.3f}")
    print(f"Depth fractions: {report['routing']['depth_fractions']}")
    print(f"Average effective layers: {report['routing']['average_effective_layers']:.3f}")
    print(f"Tokens/sec: {report['efficiency']['tokens_per_second']:.2f}")
    print(f"Saved: {run_dir / 'evaluation.json'}")

    return report