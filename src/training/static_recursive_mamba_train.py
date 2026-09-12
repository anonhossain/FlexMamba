import json
from pathlib import Path

import torch

from models.static_recursive_mamba import (
    RecursiveMambaModelConfig,
    RecursiveMambaCausalLM,
)

from evaluation.static_recursive_mamba_eval import evaluate_recursive_mamba

from src.training.common_16m import (
    PROJECT_ROOT,
    load_config,
    choose_device,
    set_seed,
    build_train_loader,
    build_validation_batches,
    build_optimizer,
    build_scheduler,
    create_run_dir,
    save_json,
    save_resolved_config,
    train_model,
)


CONFIG_PATH = "src/configs/techniques/s02_static_recursive_mamba.yaml"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_from_run(run_dir):

    run_dir = resolve_path(run_dir)
    summary_path = run_dir / "run_summary.json"

    if summary_path.exists():

        with summary_path.open("r") as file:
            summary = json.load(file)

        checkpoint = resolve_path(summary["final_checkpoint"])

        if checkpoint.exists():
            return checkpoint

    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    return checkpoints[-1]


def find_parent_checkpoint(cfg, parent_run_dir=None):

    if parent_run_dir is not None:
        return checkpoint_from_run(parent_run_dir)

    if cfg["parent"]["technique"] != "mamba":
        raise ValueError("S02 parent must be Mamba.")

    runs_root = Path(cfg["experiment"].get("runs_root", "runs"))

    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    technique_dir = runs_root / "mamba"
    selected_run = cfg["parent"].get("selected_run", "auto")

    if selected_run == "auto":

        run_dirs = sorted(
            technique_dir.glob("run_*"),
            key=lambda path: int(path.name.split("_")[-1]),
            reverse=True,
        )

        for run_dir in run_dirs:

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

        raise FileNotFoundError("No completed Mamba run found.")

    if isinstance(selected_run, int):
        run_name = f"run_{selected_run:03d}"
    else:
        run_name = str(selected_run)

        if not run_name.startswith("run_"):
            run_name = f"run_{int(run_name):03d}"

    return checkpoint_from_run(technique_dir / run_name)


def initialize_from_mamba(model, checkpoint_path, strategy):

    print(f"\nLoading Mamba parent:\n{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError("Parent checkpoint contains no config.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Parent checkpoint contains no model_state_dict.")

    parent_cfg = checkpoint["config"]
    parent_type = parent_cfg.get("technique", {}).get("model_type")

    if parent_type != "mamba":
        raise ValueError(f"Expected Mamba parent, received {parent_type}")

    parent_model = parent_cfg["model"]

    compatibility = [
        "vocab_size",
        "d_model",
        "d_state",
        "expand",
        "d_conv",
        "max_seq_len",
    ]

    for key in compatibility:

        parent_value = parent_model[key]
        child_value = getattr(model.cfg, key)

        if parent_value != child_value:
            raise ValueError(
                f"Parent/child mismatch for {key}: "
                f"{parent_value} != {child_value}"
            )

    report = model.load_mamba_parent_state_dict(
        parent_state=checkpoint["model_state_dict"],
        parent_n_layers=parent_model["n_layers"],
        strategy=strategy,
    )

    parent_params = checkpoint.get("parameter_report", {}).get("total_parameters")
    child_params = model.parameter_report()["total_parameters"]

    report["parent_parameter_count"] = parent_params
    report["child_parameter_count"] = child_params

    if parent_params:
        report["parameter_reduction"] = 1 - (child_params / parent_params)

    return report


def train_recursive_mamba(
    config_path=CONFIG_PATH,
    parent_run_dir=None,
    steps=None,
):

    cfg = load_config(config_path)

    if cfg["technique"]["model_type"] != "recursive_mamba":
        raise ValueError("S02 requires model_type='recursive_mamba'.")

    if cfg["initialization"]["mode"] != "warm_start":
        raise ValueError("S02 requires warm_start.")

    set_seed(cfg["experiment"]["seed"])
    device = choose_device()

    print(f"Device: {device}")
    print(f"Technique: {cfg['technique']['name']}")
    print(f"Scale: {cfg['experiment']['scale']}")

    parent_checkpoint = find_parent_checkpoint(cfg, parent_run_dir)

    model_cfg = RecursiveMambaModelConfig(**cfg["model"])
    model = RecursiveMambaCausalLM(model_cfg)

    init_report = initialize_from_mamba(
        model,
        parent_checkpoint,
        cfg["initialization"]["strategy"],
    )

    print("\nInitialization:")
    print(f"Strategy: {init_report['strategy']}")
    print(f"Parent layers: {init_report['parent_layers']}")
    print(f"Physical layers: {init_report['physical_layers']}")
    print(f"Effective layers: {init_report['effective_layers']}")

    if "parameter_reduction" in init_report:
        print(f"Parameter reduction: {init_report['parameter_reduction'] * 100:.2f}%")

    print("\nLayer mapping:")

    for target, sources in init_report["layer_mapping"].items():
        print(f"Child layer {target} <- Parent {sources}")

    model = model.to(device)

    print("\nParameters:", model.parameter_report())
    print("Architecture:", model.architecture_report())

    train_loader = build_train_loader(cfg, model_cfg)
    val_batches = build_validation_batches(cfg, model_cfg)

    max_steps = steps if steps is not None else cfg["training"]["max_steps"]

    optimizer = build_optimizer(model, cfg["training"])
    scheduler = build_scheduler(optimizer, cfg["training"], max_steps)

    run_dir = create_run_dir(cfg, "static_recursive_mamba")

    cfg["runtime"] = {
        "parent_checkpoint": str(parent_checkpoint),
        "parent_run_dir": str(parent_run_dir) if parent_run_dir else None,
        "initialization_report": init_report,
    }

    save_resolved_config(run_dir, cfg)

    save_json(
        run_dir / "status.json",
        {
            "status": "running",
            "stage": "s02",
            "technique": "static_recursive_mamba",
            "parent_checkpoint": str(parent_checkpoint),
        },
    )

    try:

        summary = train_model(
            model=model,
            train_loader=train_loader,
            val_batches=val_batches,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            run_dir=run_dir,
            max_steps=max_steps,
        )

        summary["parent_checkpoint"] = str(parent_checkpoint)
        summary["initialization_report"] = init_report

        save_json(run_dir / "run_summary.json", summary)

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "stage": "s02",
                "technique": "static_recursive_mamba",
                "parent_checkpoint": str(parent_checkpoint),
                "final_checkpoint": summary["final_checkpoint"],
            },
        )

        print("\nStatic Recursive Mamba training completed.")
        print(f"Run: {run_dir}")
        print(f"Checkpoint: {summary['final_checkpoint']}")

        return summary

    except Exception as error:

        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "stage": "s02",
                "technique": "static_recursive_mamba",
                "parent_checkpoint": str(parent_checkpoint),
                "error": str(error),
            },
        )

        raise


def run_static_recursive_mamba(parent_run_dir=None, steps=None):

    print("\n" + "=" * 60)
    print("S02 — STATIC RECURSIVE MAMBA 16M")
    print("=" * 60)

    summary = train_recursive_mamba(
        parent_run_dir=parent_run_dir,
        steps=steps,
    )

    evaluation = evaluate_recursive_mamba(summary)

    run_dir = Path(summary["run_dir"])

    save_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "stage": "s02",
            "technique": "static_recursive_mamba",
            "parent_checkpoint": summary["parent_checkpoint"],
            "checkpoint": summary["final_checkpoint"],
            "evaluation": str(run_dir / "evaluation.json"),
        },
    )

    print("\n" + "=" * 60)
    print("S02 STATIC RECURSIVE MAMBA COMPLETED")
    print("=" * 60)

    print(f"Run: {run_dir}")
    print(f"Parent: {summary['parent_checkpoint']}")
    print(f"Checkpoint: {summary['final_checkpoint']}")
    print(f"NLL: {evaluation['quality']['nll']:.4f}")
    print(f"PPL: {evaluation['quality']['perplexity']:.2f}")

    return {
        "training": summary,
        "evaluation": evaluation,
    }