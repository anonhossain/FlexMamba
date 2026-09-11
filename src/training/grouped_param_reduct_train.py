import json
from pathlib import Path

import torch

from src.models.grouped_param_reduct_model import (
    GroupedParamReductionModelConfig,
    GroupedParamReductionCausalLM,
)

from src.evaluation.grouped_param_reduct_eval import (
    evaluate_grouped_param_reduction,
)

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


CONFIG_PATH = "src/configs/techniques/s05_grouped_parameter_reduct.yaml"


def resolve_path(path):

    path = Path(path)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


# ============================================================
# CHECKPOINT
# ============================================================

def checkpoint_from_run(run_dir):

    run_dir = resolve_path(run_dir)

    summary_path = (
        run_dir
        / "run_summary.json"
    )

    if summary_path.exists():

        with summary_path.open("r") as file:
            summary = json.load(file)

        checkpoint = resolve_path(
            summary["final_checkpoint"]
        )

        if checkpoint.exists():
            return checkpoint

    checkpoints = sorted(
        (
            run_dir
            / "checkpoints"
        ).glob("step_*.pt")
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}"
        )

    return checkpoints[-1]


# ============================================================
# FIND S04 PARENT
# ============================================================

def find_parent_checkpoint(
    cfg,
    parent_run_dir=None,
):

    if parent_run_dir is not None:
        return checkpoint_from_run(
            parent_run_dir
        )

    if cfg["parent"]["technique"] != "dynamic_mor":
        raise ValueError(
            "S05 parent must be dynamic_mor."
        )

    runs_root = Path(
        cfg["experiment"].get(
            "runs_root",
            "runs",
        )
    )

    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    technique_dir = (
        runs_root
        / "dynamic_mor"
    )

    selected_run = cfg["parent"].get(
        "selected_run",
        "auto",
    )

    if selected_run == "auto":

        runs = sorted(
            technique_dir.glob("run_*"),
            key=lambda path: int(
                path.name.split("_")[-1]
            ),
            reverse=True,
        )

        for run_dir in runs:

            status_path = (
                run_dir
                / "status.json"
            )

            if not status_path.exists():
                continue

            with status_path.open("r") as file:
                status = json.load(file)

            if status.get("status") != "completed":
                continue

            try:
                return checkpoint_from_run(
                    run_dir
                )

            except FileNotFoundError:
                continue

        raise FileNotFoundError(
            "No completed S04 Dynamic MoR run found."
        )

    run_name = (
        f"run_{selected_run:03d}"
        if isinstance(selected_run, int)
        else str(selected_run)
    )

    if not run_name.startswith("run_"):
        run_name = (
            f"run_{int(run_name):03d}"
        )

    return checkpoint_from_run(
        technique_dir / run_name
    )


# ============================================================
# INITIALIZE FROM S04
# ============================================================

def initialize_from_parent(
    model,
    checkpoint_path,
):

    print(
        "\nLoading Dynamic MoR parent:\n"
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "Parent checkpoint has no config."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Parent checkpoint has no model_state_dict."
        )

    parent_cfg = checkpoint["config"]

    parent_type = (
        parent_cfg
        .get("technique", {})
        .get("model_type")
    )

    if parent_type != "dynamic_recursive_mamba":
        raise ValueError(
            "Expected dynamic_recursive_mamba parent, "
            f"received {parent_type}"
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
        child_value = getattr(
            model.cfg,
            key,
        )

        if parent_value != child_value:
            raise ValueError(
                f"Parent/child mismatch for {key}: "
                f"{parent_value} != {child_value}"
            )

    parent_recursion = parent_cfg["recursion"]

    if (
        parent_recursion["min_recursions"]
        != model.min_recursions
    ):
        raise ValueError(
            "min_recursions mismatch."
        )

    if (
        parent_recursion["max_recursions"]
        != model.max_recursions
    ):
        raise ValueError(
            "max_recursions mismatch."
        )

    return model.load_parent_state_dict(
        checkpoint["model_state_dict"]
    )


# ============================================================
# TRAIN
# ============================================================

def train_grouped_param_reduction(
    config_path=CONFIG_PATH,
    parent_run_dir=None,
    steps=None,
):

    cfg = load_config(config_path)

    if (
        cfg["technique"]["model_type"]
        != "grouped_dynamic_mamba"
    ):
        raise ValueError(
            "S05 requires "
            "model_type='grouped_dynamic_mamba'."
        )

    if cfg["initialization"]["mode"] != "warm_start":
        raise ValueError(
            "S05 requires warm_start."
        )

    set_seed(
        cfg["experiment"]["seed"]
    )

    device = choose_device()

    print(f"Device: {device}")
    print(
        f"Technique: "
        f"{cfg['technique']['name']}"
    )
    print(
        f"Scale: "
        f"{cfg['experiment']['scale']}"
    )

    parent_checkpoint = (
        find_parent_checkpoint(
            cfg,
            parent_run_dir,
        )
    )

    model_cfg = (
        GroupedParamReductionModelConfig(
            **cfg["model"]
        )
    )

    model = GroupedParamReductionCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["grouped_parameterization"],
    )

    init_report = initialize_from_parent(
        model,
        parent_checkpoint,
    )

    print("\nInitialization:")
    print(
        f"Strategy: "
        f"{init_report['strategy']}"
    )
    print(
        f"Groups: "
        f"{init_report['groups']}"
    )
    print(
        "Parent equivalent parameters: "
        f"{init_report['parent_equivalent_parameter_count']:,}"
    )
    print(
        "Grouped child parameters: "
        f"{init_report['child_parameter_count']:,}"
    )
    print(
        "Parameter savings: "
        f"{init_report['parameter_savings']:,}"
    )
    print(
        "Parameter reduction: "
        f"{init_report['parameter_reduction_fraction'] * 100:.2f}%"
    )

    model = model.to(device)

    print(
        "\nParameters:",
        model.parameter_report(),
    )

    print(
        "Architecture:",
        model.architecture_report(),
    )

    print(
        "Grouping:",
        model.grouping_report(),
    )

    train_loader = build_train_loader(
        cfg,
        model_cfg,
    )

    val_batches = (
        build_validation_batches(
            cfg,
            model_cfg,
        )
    )

    max_steps = (
        steps
        if steps is not None
        else cfg["training"]["max_steps"]
    )

    # Architecture changed, therefore fresh optimizer/scheduler.
    optimizer = build_optimizer(
        model,
        cfg["training"],
    )

    scheduler = build_scheduler(
        optimizer,
        cfg["training"],
        max_steps,
    )

    run_dir = create_run_dir(
        cfg,
        "grouped_parameter",
    )

    cfg["runtime"] = {
        "parent_checkpoint": str(
            parent_checkpoint
        ),

        "parent_run_dir": (
            str(parent_run_dir)
            if parent_run_dir
            else None
        ),

        "initialization_report": (
            init_report
        ),
    }

    save_resolved_config(
        run_dir,
        cfg,
    )

    save_json(
        run_dir / "status.json",
        {
            "status": "running",
            "stage": "s05",
            "technique": "grouped_parameter",
            "parent_checkpoint": str(
                parent_checkpoint
            ),
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

        summary["parent_checkpoint"] = str(
            parent_checkpoint
        )

        summary[
            "initialization_report"
        ] = init_report

        summary[
            "grouping_report"
        ] = model.grouping_report()

        save_json(
            run_dir / "run_summary.json",
            summary,
        )

        evaluation = (
            evaluate_grouped_param_reduction(
                summary
            )
        )

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "stage": "s05",
                "technique": "grouped_parameter",
                "parent_checkpoint": str(
                    parent_checkpoint
                ),
                "final_checkpoint": (
                    summary[
                        "final_checkpoint"
                    ]
                ),
                "evaluation": str(
                    run_dir
                    / "evaluation.json"
                ),
            },
        )

        return summary, evaluation

    except Exception as error:

        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "stage": "s05",
                "technique": "grouped_parameter",
                "parent_checkpoint": str(
                    parent_checkpoint
                ),
                "error": str(error),
            },
        )

        raise


# ============================================================
# FULL S05
# ============================================================

def run_grouped_param_reduct(
    parent_run_dir=None,
    steps=None,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S05 — GROUPED PARAMETER REDUCTION"
    )

    print(
        "=" * 60
    )

    summary, evaluation = (
        train_grouped_param_reduction(
            parent_run_dir=parent_run_dir,
            steps=steps,
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S05 GROUPED PARAMETER REDUCTION COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"Run: "
        f"{summary['run_dir']}"
    )

    print(
        f"Parent: "
        f"{summary['parent_checkpoint']}"
    )

    print(
        f"Checkpoint: "
        f"{summary['final_checkpoint']}"
    )

    print(
        "Parameters: "
        f"{evaluation['parameters']['total_parameters']:,}"
    )

    print(
        "Parameter reduction: "
        f"{evaluation['parameter_reduction']['overall_reduction'] * 100:.2f}%"
    )

    print(
        f"NLL: "
        f"{evaluation['quality']['nll']:.4f}"
    )

    print(
        f"PPL: "
        f"{evaluation['quality']['perplexity']:.2f}"
    )

    print(
        "Average recursion: "
        f"{evaluation['routing']['average_recursion_depth']:.3f}"
    )

    return {
        "training": summary,
        "evaluation": evaluation,
    }