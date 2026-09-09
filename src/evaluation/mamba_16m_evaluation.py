# src/evaluation/mamba_16m_evaluation.py

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.mamba import MambaModelConfig, MambaCausalLM
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
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def current_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)

    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 ** 2)

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

    # Warmup
    batch = batches[0].to(device)
    x = batch[:, :-1]

    model(x)
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
        elapsed = time.perf_counter() - start

        logits = output["logits"]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        predictions = logits.argmax(dim=-1)
        top5 = logits.topk(5, dim=-1).indices

        tokens = y.numel()

        total_loss += loss.item()
        total_tokens += tokens
        total_time += elapsed

        top1_correct += (predictions == y).sum().item()

        top5_correct += (
            top5 == y.unsqueeze(-1)
        ).any(dim=-1).sum().item()

        memory = current_memory_mb(device)

        if memory is not None:
            sampled_peak_memory = max(
                sampled_peak_memory or 0,
                memory,
            )

    nll = total_loss / total_tokens

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


def evaluate_mamba(summary):
    checkpoint_path = resolve_path(summary["final_checkpoint"])
    run_dir = resolve_path(summary["run_dir"])

    device = choose_device()

    print(f"\nEvaluation device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # --------------------------------------------------------
    # CHECKPOINT / CONFIG
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    if cfg["technique"]["model_type"] != "mamba":
        raise ValueError("Checkpoint is not a Mamba model.")

    model_cfg = MambaModelConfig(**cfg["model"])

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MambaCausalLM(model_cfg)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # EVALUATION DATA
    # --------------------------------------------------------

    eval_cfg = cfg.get("evaluation", {})

    data_cfg = copy.deepcopy(cfg)

    data_cfg["validation"] = {
        "enabled": True,
        "eval_batches": eval_cfg.get(
            "eval_batches",
            cfg.get("validation", {}).get("eval_batches", 20),
        ),
    }

    batches = build_validation_batches(
        data_cfg,
        model_cfg,
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    metrics = run_evaluation(
        model,
        batches,
        device,
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / (1024 ** 2)
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "identity": {
            "technique": cfg["technique"]["name"],
            "scale": cfg["experiment"].get("scale"),
            "run": summary["run"],
            "checkpoint": str(checkpoint_path),
            "device": str(device),
        },

        "architecture": {
            **model.architecture_report(),
            "vocab_size": model_cfg.vocab_size,
            "d_model": model_cfg.d_model,
            "n_layers": model_cfg.n_layers,
            "d_state": model_cfg.d_state,
            "expand": model_cfg.expand,
            "d_conv": model_cfg.d_conv,
            "max_seq_len": model_cfg.max_seq_len,
        },

        "parameters": model.parameter_report(),

        "training": {
            "completed_steps": summary["completed_steps"],
            "training_tokens": summary["training_tokens"],
            "final_training_loss": summary["final_training_loss"],
            "training_seconds": summary["training_seconds"],
            "training_tokens_per_second": summary["tokens_per_second"],
            "validation_history": summary["validation_history"],
        },

        "quality": metrics["quality"],
        "efficiency": metrics["efficiency"],

        "storage": {
            "model_weight_memory_mb": model_weight_memory_mb(model),
            "checkpoint_size_mb": checkpoint_size_mb,
        },

        "evaluation_setup": {
            "evaluation_batches": len(batches),
            "batch_size": cfg["training"]["batch_size"],
            "sequence_length": model_cfg.max_seq_len,
            "same_validation_split": True,
        },
    }

    save_json(
        run_dir / "evaluation.json",
        report,
    )

    print("\nEvaluation completed.")
    print(f"NLL: {report['quality']['nll']:.4f}")
    print(f"PPL: {report['quality']['perplexity']:.2f}")
    print(
        f"Tokens/sec: "
        f"{report['efficiency']['tokens_per_second']:.2f}"
    )
    print(f"Saved: {run_dir / 'evaluation.json'}")

    return report