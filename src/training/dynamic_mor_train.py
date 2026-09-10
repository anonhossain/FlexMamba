import json
from pathlib import Path

import torch

from src.models.dynamic_mor_models import (
    DynamicMoRModelConfig,
    DynamicMoRCausalLM,
)

from src.evaluation.dynamic_mor_eval import evaluate_dynamic_mor

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


CONFIG_PATH = "src/configs/techniques/s04_dynamic_mor.yaml"


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

    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    return checkpoints[-1]


def find_parent_checkpoint(cfg, parent_run_dir=None):

    if parent_run_dir is not None:
        return checkpoint_from_run(parent_run_dir)

    if cfg["parent"]["technique"] != "recursion_wise_state":
        raise ValueError("S04 parent must be recursion_wise_state.")

    runs_root = Path(cfg["experiment"].get("runs_root", "runs"))

    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    technique_dir = runs_root / "recursion_wise_state"
    selected_run = cfg["parent"].get("selected_run", "auto")

    if selected_run == "auto":

        runs = sorted(
            technique_dir.glob("run_*"),
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

        raise FileNotFoundError("No completed S03 run found.")

    run_name = (
        f"run_{selected_run:03d}"
        if isinstance(selected_run, int)
        else str(selected_run)
    )

    if not run_name.startswith("run_"):
        run_name = f"run_{int(run_name):03d}"

    return checkpoint_from_run(technique_dir / run_name)


def initialize_from_parent(model, checkpoint_path):

    print(f"\nLoading Recursion-Wise State parent:\n{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError("Parent checkpoint has no config.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Parent checkpoint has no model_state_dict.")

    parent_cfg = checkpoint["config"]
    parent_type = parent_cfg.get("technique", {}).get("model_type")

    if parent_type != "recursion_wise_state_mamba":
        raise ValueError(
            f"Expected recursion_wise_state_mamba parent, received {parent_type}"
        )

    parent_model = parent_cfg["model"]

    compatibility = [
        "vocab_size",
        "d_model",
        "input_layers",
        "shared_middle_layers",
        "output_layers",
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

    if parent_model["num_recursions"] != model.max_recursions:
        raise ValueError(
            f"S03 recursions ({parent_model['num_recursions']}) != "
            f"S04 max recursions ({model.max_recursions})"
        )

    return model.load_parent_state_dict(
        checkpoint["model_state_dict"]
    )


def train_dynamic_mor(
    config_path=CONFIG_PATH,
    parent_run_dir=None,
    steps=None,
):

    cfg = load_config(config_path)

    if cfg["technique"]["model_type"] != "dynamic_recursive_mamba":
        raise ValueError("S04 requires model_type='dynamic_recursive_mamba'.")

    if cfg["initialization"]["mode"] != "warm_start":
        raise ValueError("S04 requires warm_start.")

    set_seed(cfg["experiment"]["seed"])
    device = choose_device()

    print(f"Device: {device}")
    print(f"Technique: {cfg['technique']['name']}")
    print(f"Scale: {cfg['experiment']['scale']}")

    parent_checkpoint = find_parent_checkpoint(cfg, parent_run_dir)

    model_cfg = DynamicMoRModelConfig(**cfg["model"])

    model = DynamicMoRCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
    )

    init_report = initialize_from_parent(
        model,
        parent_checkpoint,
    )

    print("\nInitialization:")
    print(f"Strategy: {init_report['strategy']}")
    print(f"Transferred parameters: {init_report['transferred_parameter_count']:,}")
    print(f"New router parameters: {init_report['new_router_parameter_count']:,}")
    print(f"Transfer fraction: {init_report['transferred_parameter_fraction']:.4f}")

    model = model.to(device)

    print("\nParameters:", model.parameter_report())
    print("Architecture:", model.architecture_report())

    train_loader = build_train_loader(cfg, model_cfg)
    val_batches = build_validation_batches(cfg, model_cfg)

    max_steps = steps if steps is not None else cfg["training"]["max_steps"]

    # Fresh optimizer because the router is new.
    optimizer = build_optimizer(model, cfg["training"])
    scheduler = build_scheduler(optimizer, cfg["training"], max_steps)

    run_dir = create_run_dir(cfg, "dynamic_mor")

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
            "stage": "s04",
            "technique": "dynamic_mor",
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

        evaluation = evaluate_dynamic_mor(summary)

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "stage": "s04",
                "technique": "dynamic_mor",
                "parent_checkpoint": str(parent_checkpoint),
                "final_checkpoint": summary["final_checkpoint"],
                "evaluation": str(run_dir / "evaluation.json"),
            },
        )

        return summary, evaluation

    except Exception as error:

        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "stage": "s04",
                "technique": "dynamic_mor",
                "parent_checkpoint": str(parent_checkpoint),
                "error": str(error),
            },
        )

        raise


def run_dynamic_mor(parent_run_dir=None, steps=None):

    print("\n" + "=" * 60)
    print("S04 — DYNAMIC MoR ROUTING")
    print("=" * 60)

    summary, evaluation = train_dynamic_mor(
        parent_run_dir=parent_run_dir,
        steps=steps,
    )

    print("\n" + "=" * 60)
    print("S04 DYNAMIC MoR COMPLETED")
    print("=" * 60)

    print(f"Run: {summary['run_dir']}")
    print(f"Parent: {summary['parent_checkpoint']}")
    print(f"Checkpoint: {summary['final_checkpoint']}")
    print(f"NLL: {evaluation['quality']['nll']:.4f}")
    print(f"PPL: {evaluation['quality']['perplexity']:.2f}")
    print(f"Average recursion: {evaluation['routing']['average_recursion_depth']:.3f}")
    print(f"Depth fractions: {evaluation['routing']['depth_fractions']}")

    return {
        "training": summary,
        "evaluation": evaluation,
    }