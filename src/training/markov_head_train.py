import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.markov_head_model import (
    FlexMambaMarkov,
    load_flexmamba_target,
)

from src.evaluation.markov_head_eval import (
    evaluate_markov_head,
)

from src.training.common_16m import (
    PROJECT_ROOT,
    load_config,
    choose_device,
    set_seed,
    build_train_loader,
    build_validation_batches,
    build_scheduler,
    create_run_dir,
    save_json,
    save_resolved_config,
)


CONFIG_PATH = "src/configs/techniques/s12_markov_head.yaml"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# PARENT CHECKPOINT
# ============================================================

def checkpoint_from_run(run_dir):

    run_dir = resolve_path(run_dir)
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists():

        with summary_path.open("r") as file:
            summary = json.load(file)

        checkpoint = resolve_path(summary["final_checkpoint"])

        if checkpoint.exists():
            return checkpoint

    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    return checkpoints[-1]


def find_parent_checkpoint(cfg, parent_run_dir=None):

    if parent_run_dir is not None:
        return checkpoint_from_run(parent_run_dir)

    if cfg["parent"]["technique"] != "final_joint":
        raise ValueError("S12 parent must be final_joint.")

    root = Path(cfg["experiment"].get("runs_root", "runs"))

    if not root.is_absolute():
        root = PROJECT_ROOT / root

    technique_root = root / "final_joint"
    selected = cfg["parent"].get("selected_run", "auto")

    if selected == "auto":

        runs = sorted(
            technique_root.glob("run_*"),
            key=lambda path: int(path.name.split("_")[-1]),
            reverse=True,
        )

        for run_dir in runs:

            status_path = run_dir / "status.json"

            if not status_path.exists():
                continue

            with status_path.open("r") as file:
                status = json.load(file)

            if status.get("status") != "completed":
                continue

            try:
                return checkpoint_from_run(run_dir)
            except FileNotFoundError:
                continue

        raise FileNotFoundError("No completed final_joint run found.")

    run_name = f"run_{selected:03d}" if isinstance(selected, int) else str(selected)

    if not run_name.startswith("run_"):
        run_name = f"run_{int(run_name):03d}"

    return checkpoint_from_run(technique_root / run_name)


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def validate_markov(model, batches, device, sequence_tokens, anchors):

    model.eval()

    totals = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "tv_loss": 0.0,
        "confidence_loss": 0.0,
        "confidence_mae": 0.0,
        "tau_probabilistic": 0.0,
    }

    for batch in batches:

        x = batch[:, :sequence_tokens].to(device)

        result = model.training_forward(
            x,
            anchors_per_sequence=anchors,
        )

        for key in totals:
            totals[key] += float(result[key].detach().cpu())

    count = max(1, len(batches))

    return {
        key: value / count
        for key, value in totals.items()
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_markov_checkpoint(path, model, optimizer, scheduler, step, cfg, parent_checkpoint):

    torch.save(
        {
            "step": step,
            "adapter_state_dict": model.adapter_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
            "parent_checkpoint": str(parent_checkpoint),
        },
        path,
    )


# ============================================================
# TRAINING
# ============================================================

def train_markov_head(
    config_path=CONFIG_PATH,
    parent_run_dir=None,
    steps=None,
):

    cfg = load_config(config_path)

    if cfg["technique"]["model_type"] != "flexmamba_markov":
        raise ValueError("S12 requires model_type='flexmamba_markov'.")

    set_seed(cfg["experiment"]["seed"])

    device = choose_device()

    print(f"Device: {device}")
    print(f"Technique: {cfg['technique']['name']}")
    print(f"Scale: {cfg['experiment']['scale']}")

    parent_checkpoint = find_parent_checkpoint(cfg, parent_run_dir)

    print(f"\nLoading frozen FlexMamba target:\n{parent_checkpoint}")

    target, parent_cfg = load_flexmamba_target(
        parent_checkpoint,
        device,
    )

    model = FlexMambaMarkov(
        target,
        cfg,
    ).to(device)

    print("\nParameters:", model.parameter_report())
    print("Architecture:", model.architecture_report())

    smoke = cfg.get("smoke", {})
    smoke_enabled = smoke.get("enabled", False)

    sequence_tokens = (
        smoke.get("sequence_tokens", 64)
        if smoke_enabled
        else cfg["training"]["sequence_tokens"]
    )

    anchors = (
        smoke.get("anchors_per_sequence", 2)
        if smoke_enabled
        else cfg["markov_head"]["anchors_per_sequence"]
    )

    max_steps = (
        steps
        if steps is not None
        else (
            smoke.get("max_steps", 1)
            if smoke_enabled
            else cfg["training"]["max_steps"]
        )
    )

    grad_accum = (
        smoke.get("grad_accum_steps", 1)
        if smoke_enabled
        else cfg["training"]["grad_accum_steps"]
    )

    # Common FineWeb loader uses the parent target's model configuration.
    target_cfg = target.cfg

    train_loader = build_train_loader(
        cfg,
        target_cfg,
    )

    val_cfg = dict(cfg)

    if smoke_enabled:
        val_cfg["validation"] = dict(cfg["validation"])
        val_cfg["validation"]["eval_batches"] = 1

    val_batches = build_validation_batches(
        val_cfg,
        target_cfg,
    )

    trainable = model.adapter_parameters()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg["training"]["learning_rate"]),
        betas=(
            float(cfg["training"]["beta1"]),
            float(cfg["training"]["beta2"]),
        ),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler_cfg = dict(cfg["training"])
    scheduler_cfg["grad_accum_steps"] = grad_accum

    scheduler = build_scheduler(
        optimizer,
        scheduler_cfg,
        max_steps,
    )

    run_dir = create_run_dir(
        cfg,
        "markov_head",
    )

    cfg["runtime"] = {
        "parent_checkpoint": str(parent_checkpoint),
        "parent_model_type": parent_cfg["technique"]["model_type"],
        "target_frozen": True,
        "sequence_tokens": sequence_tokens,
        "anchors_per_sequence": anchors,
    }

    save_resolved_config(run_dir, cfg)

    save_json(
        run_dir / "status.json",
        {
            "status": "running",
            "stage": "s12",
            "technique": "markov_head",
            "parent_checkpoint": str(parent_checkpoint),
        },
    )

    checkpoint_steps = set(
        cfg.get("checkpoint", {}).get("save_steps", [])
    )

    checkpoint_steps.add(max_steps)

    train_iterator = iter(train_loader)

    optimizer.zero_grad(set_to_none=True)

    start_time = time.perf_counter()

    final_loss = None
    validation_history = []

    progress = tqdm(
        range(1, max_steps + 1),
        desc="markov_training",
    )

    try:

        for step in progress:

            model.train()

            accumulated = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "tv_loss": 0.0,
                "confidence_loss": 0.0,
                "tau_probabilistic": 0.0,
            }

            for _ in range(grad_accum):

                try:
                    batch = next(train_iterator)

                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                x = batch[:, :sequence_tokens].to(device)

                output = model.training_forward(
                    x,
                    anchors_per_sequence=anchors,
                )

                loss = output["loss"]

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Non-finite Markov loss at step {step}."
                    )

                (loss / grad_accum).backward()

                for key in accumulated:
                    accumulated[key] += (
                        float(output[key].detach().cpu())
                        / grad_accum
                    )

            torch.nn.utils.clip_grad_norm_(
                trainable,
                float(cfg["training"]["grad_clip"]),
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            final_loss = accumulated["loss"]

            if step == 1 or step % cfg["training"]["log_every"] == 0:

                progress.set_postfix(
                    loss=f"{final_loss:.4f}",
                    ce=f"{accumulated['ce_loss']:.3f}",
                    tv=f"{accumulated['tv_loss']:.3f}",
                    tau=f"{accumulated['tau_probabilistic']:.2f}",
                )

            should_validate = (
                cfg["validation"].get("enabled", True)
                and (
                    step % cfg["validation"]["eval_every"] == 0
                    or step == max_steps
                )
            )

            if should_validate:

                validation = validate_markov(
                    model,
                    val_batches,
                    device,
                    sequence_tokens,
                    anchors,
                )

                validation["step"] = step
                validation_history.append(validation)

                print(
                    f"\nValidation step {step}: "
                    f"loss={validation['loss']:.4f} | "
                    f"CE={validation['ce_loss']:.4f} | "
                    f"TV={validation['tv_loss']:.4f} | "
                    f"conf={validation['confidence_loss']:.4f} | "
                    f"tau={validation['tau_probabilistic']:.3f}"
                )

            if step in checkpoint_steps:

                checkpoint_path = (
                    run_dir
                    / "checkpoints"
                    / f"step_{step:07d}.pt"
                )

                save_markov_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    cfg,
                    parent_checkpoint,
                )

                print(f"\nSaved Markov checkpoint: {checkpoint_path}")

        elapsed = time.perf_counter() - start_time

        final_checkpoint = (
            run_dir
            / "checkpoints"
            / f"step_{max_steps:07d}.pt"
        )

        tokens_per_step = (
            cfg["training"]["batch_size"]
            * grad_accum
            * sequence_tokens
        )

        summary = {
            "technique": "markov_head",
            "scale": cfg["experiment"]["scale"],

            "run": run_dir.name,
            "run_dir": str(run_dir),

            "completed_steps": max_steps,
            "final_training_loss": final_loss,

            "tokens_per_step": tokens_per_step,
            "training_tokens": tokens_per_step * max_steps,

            "training_seconds": elapsed,
            "seconds_per_step": elapsed / max_steps,

            "parent_checkpoint": str(parent_checkpoint),
            "final_checkpoint": str(final_checkpoint),

            "validation_history": validation_history,
            "final_validation": (
                validation_history[-1]
                if validation_history
                else None
            ),

            "parameter_report": model.parameter_report(),
            "architecture_report": model.architecture_report(),
        }

        save_json(
            run_dir / "run_summary.json",
            summary,
        )

        evaluation = evaluate_markov_head(
            summary
        )

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "stage": "s12",
                "technique": "markov_head",
                "final_checkpoint": str(final_checkpoint),
                "evaluation": str(run_dir / "evaluation.json"),
            },
        )

        return summary, evaluation

    except Exception as error:

        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "stage": "s12",
                "technique": "markov_head",
                "error": str(error),
            },
        )

        raise


def run_markov_head(parent_run_dir=None, steps=None):

    print("\n" + "=" * 60)
    print("S12 — MARKOV SPECULATIVE GENERATION")
    print("=" * 60)

    summary, evaluation = train_markov_head(
        parent_run_dir=parent_run_dir,
        steps=steps,
    )

    print("\n" + "=" * 60)
    print("S12 MARKOV HEAD COMPLETED")
    print("=" * 60)

    print(f"Run: {summary['run_dir']}")
    print(f"Target: {summary['parent_checkpoint']}")
    print(f"Checkpoint: {summary['final_checkpoint']}")

    print(
        "Acceptance rate: "
        f"{evaluation['speculative']['draft_acceptance_rate'] * 100:.2f}%"
    )

    print(
        "Accepted length τ: "
        f"{evaluation['speculative']['accepted_length_tau']:.3f}"
    )

    print(
        "Measured speedup: "
        f"{evaluation['speedup_ratio']:.3f}x"
    )

    print(
        "Greedy equivalence: "
        f"{evaluation['quality_equivalence']['greedy_sequence_match']}"
    )

    return {
        "training": summary,
        "evaluation": evaluation,
    }