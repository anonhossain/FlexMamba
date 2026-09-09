# src/training/common_16m.py

import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.fineweb import PackedFineWebDataset
from src.utils.checkpoint import save_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# CONFIG
# ============================================================

def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r") as file:
        return yaml.safe_load(file) or {}


def load_config(path):
    return load_yaml(path)


# ============================================================
# DEVICE / SEED
# ============================================================

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def build_train_loader(cfg, model_cfg):
    data_cfg = cfg["data"]
    tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer_name"])

    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({len(tokenizer)}) != "
            f"model vocab ({model_cfg.vocab_size})"
        )

    dataset = PackedFineWebDataset(
        tokenizer=tokenizer,
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg["dataset_config"],
        seq_len=model_cfg.max_seq_len,
        mode="train",
        validation_docs=data_cfg["validation_docs"],
        shuffle_buffer=data_cfg["shuffle_buffer"],
        seed=cfg["experiment"]["seed"],
    )

    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=0,
    )


def build_validation_batches(cfg, model_cfg):
    val_cfg = cfg.get("validation", {})

    if not val_cfg.get("enabled", False):
        return []

    data_cfg = cfg["data"]
    tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer_name"])

    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({len(tokenizer)}) != "
            f"model vocab ({model_cfg.vocab_size})"
        )

    dataset = PackedFineWebDataset(
        tokenizer=tokenizer,
        dataset_name=data_cfg["dataset_name"],
        dataset_config=data_cfg["dataset_config"],
        seq_len=model_cfg.max_seq_len,
        mode="validation",
        validation_docs=data_cfg["validation_docs"],
        shuffle_buffer=0,
        seed=cfg["experiment"]["seed"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=0,
    )

    batches = []

    for i, batch in enumerate(loader):
        if i >= val_cfg.get("eval_batches", 20):
            break

        batches.append(batch.cpu())

    if not batches:
        raise RuntimeError("No validation batches were created.")

    return batches


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def validate_model(model, val_batches, device):
    if not val_batches:
        return None

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in val_batches:
        batch = batch.to(device)

        x = batch[:, :-1]
        y = batch[:, 1:]

        output = model(x, labels=y)
        loss = output["lm_loss"]

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    model.train()

    nll = total_loss / total_tokens

    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "tokens": total_tokens,
    }


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(model, cfg):
    return AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )


# ============================================================
# SCHEDULER
# ============================================================

def build_scheduler(optimizer, cfg, max_steps):
    warmup_steps = max(1, int(max_steps * cfg["warmup_ratio"]))
    min_lr_ratio = cfg["min_learning_rate"] / cfg["learning_rate"]

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# RUN DIRECTORY
# ============================================================

def create_run_dir(cfg, technique_name):
    runs_root = Path(cfg["experiment"].get("runs_root", "runs"))

    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    technique_dir = runs_root / technique_name
    technique_dir.mkdir(parents=True, exist_ok=True)

    run_numbers = []

    for path in technique_dir.glob("run_*"):
        try:
            run_numbers.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue

    next_run = max(run_numbers) + 1 if run_numbers else 1

    run_dir = technique_dir / f"run_{next_run:03d}"
    run_dir.mkdir()

    (run_dir / "checkpoints").mkdir()

    return run_dir


# ============================================================
# FILE HELPERS
# ============================================================

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(data, file, indent=2)


def save_resolved_config(run_dir, cfg):
    with (Path(run_dir) / "resolved_config.yaml").open("w") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_batches,
    optimizer,
    scheduler,
    device,
    cfg,
    run_dir,
    max_steps,
):
    train_cfg = cfg["training"]
    val_cfg = cfg.get("validation", {})

    grad_accum = train_cfg["grad_accum_steps"]

    save_steps = set(
        cfg.get("checkpoint", {}).get("save_steps", [])
    )

    save_steps.add(max_steps)

    checkpoint_dir = Path(run_dir) / "checkpoints"
    train_iter = iter(train_loader)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    start_time = time.perf_counter()

    final_loss = None
    validation_history = []

    progress = tqdm(
        range(1, max_steps + 1),
        desc="training",
    )

    for step in progress:
        step_loss = 0.0

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(device)

            x = batch[:, :-1]
            y = batch[:, 1:]

            output = model(x, labels=y)
            loss = output["loss"]

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at step {step}"
                )

            (loss / grad_accum).backward()

            step_loss += (
                loss.detach().float().item()
                / grad_accum
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_cfg["grad_clip"],
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        final_loss = step_loss

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        if step == 1 or step % train_cfg["log_every"] == 0:
            progress.set_postfix(
                loss=f"{step_loss:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        should_validate = (
            val_cfg.get("enabled", False)
            and val_batches
            and (
                step % val_cfg.get("eval_every", max_steps) == 0
                or step == max_steps
            )
        )

        if should_validate:
            result = validate_model(
                model,
                val_batches,
                device,
            )

            result["step"] = step
            validation_history.append(result)

            print(
                f"\nValidation step {step}: "
                f"NLL={result['nll']:.4f} | "
                f"PPL={result['perplexity']:.2f}"
            )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if step in save_steps:
            checkpoint_path = (
                checkpoint_dir
                / f"step_{step:07d}.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                step,
                cfg,
            )

            print(
                f"\nSaved checkpoint: "
                f"{checkpoint_path}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = time.perf_counter() - start_time

    final_checkpoint = (
        checkpoint_dir
        / f"step_{max_steps:07d}.pt"
    )

    tokens_per_step = (
        train_cfg["batch_size"]
        * grad_accum
        * model.cfg.max_seq_len
    )

    training_tokens = tokens_per_step * max_steps

    return {
        "technique": cfg["technique"]["name"],
        "scale": cfg["experiment"].get("scale"),
        "debug": cfg["experiment"].get("debug", False),
        "run": Path(run_dir).name,
        "run_dir": str(run_dir),
        "completed_steps": max_steps,
        "final_training_loss": final_loss,
        "tokens_per_step": tokens_per_step,
        "training_tokens": training_tokens,
        "training_seconds": elapsed,
        "seconds_per_step": elapsed / max_steps,
        "tokens_per_second": training_tokens / elapsed,

        "validation_history": validation_history,
        "final_validation": (
            validation_history[-1]
            if validation_history
            else None
        ),
        "final_checkpoint": str(final_checkpoint),
        "parameter_report": model.parameter_report(),
        "architecture_report": model.architecture_report(),
    }