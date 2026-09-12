import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models.static_recursive_mamba import (
    RecursiveMambaModelConfig,
    RecursiveMambaCausalLM,
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

        memory = current_memory_mb(device)

        if memory is not None:
            sampled_peak_memory = max(sampled_peak_memory or 0, memory)

    nll = total_loss / total_tokens

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = sampled_peak_memory

    return {
        "quality": {
            "validation_loss": nll,
            "nll": nll,
            "perplexity": math.exp(min(nll, 20.0)),
            "bits_per_token": nll / math.log(2),
            "token_accuracy": top1_correct / total_tokens,
            "top_5_token_accuracy": top5_correct / total_tokens,
        },

        "efficiency": {
            "evaluation_tokens": total_tokens,
            "forward_seconds": total_time,
            "tokens_per_second": total_tokens / total_time,
            "milliseconds_per_token": total_time * 1000 / total_tokens,
            "peak_memory_mb": peak_memory,
        },
    }


def evaluate_static_recursive_mamba(summary):

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

    if cfg["technique"]["model_type"] != "recursive_mamba":
        raise ValueError("Checkpoint is not Static Recursive Mamba.")

    model_cfg = RecursiveMambaModelConfig(**cfg["model"])
    model = RecursiveMambaCausalLM(model_cfg)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

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

    report = {
        "identity": {
            "technique": cfg["technique"]["name"],
            "model_type": "recursive_mamba",
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

        "recursion": {
            "mode": "static",
            "num_recursions": model_cfg.num_recursions,
            "physical_layers": model_cfg.physical_layers,
            "effective_layers": model_cfg.effective_layers,
            "average_recursion_depth": float(model_cfg.num_recursions),
        },

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

    print("\nStatic Recursive Mamba evaluation completed.")
    print(f"NLL: {report['quality']['nll']:.4f}")
    print(f"PPL: {report['quality']['perplexity']:.2f}")
    print(f"Physical layers: {model_cfg.physical_layers}")
    print(f"Effective layers: {model_cfg.effective_layers}")
    print(f"Tokens/sec: {report['efficiency']['tokens_per_second']:.2f}")
    print(f"Saved: {run_dir / 'evaluation.json'}")

    return report