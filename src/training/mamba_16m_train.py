# src/training/mamba_16m_train.py

import json
from pathlib import Path

import torch

from src.models.mamba import (
    MambaModelConfig,
    MambaCausalLM,
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


CONFIG_PATH = (
    "src/configs/techniques/"
    "s01_mamba_16m.yaml"
)


# ============================================================
# PATH
# ============================================================

def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# PARENT RUN
# ============================================================

def checkpoint_from_run(run_dir):

    run_dir = resolve_path(run_dir)

    summary_path = (
        run_dir / "run_summary.json"
    )

    if summary_path.exists():

        with summary_path.open("r") as f:
            summary = json.load(f)

        checkpoint = Path(
            summary["final_checkpoint"]
        )

        if not checkpoint.is_absolute():
            checkpoint = (
                PROJECT_ROOT
                / checkpoint
            )

        if checkpoint.exists():
            return checkpoint

    # fallback:
    # use latest checkpoint in directory

    checkpoint_dir = (
        run_dir / "checkpoints"
    )

    checkpoints = sorted(
        checkpoint_dir.glob(
            "step_*.pt"
        )
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in "
            f"{checkpoint_dir}"
        )

    return checkpoints[-1]


def find_parent_checkpoint(
    cfg,
    parent_run_dir=None,
):

    # Pipeline supplied exact parent.
    if parent_run_dir is not None:

        return checkpoint_from_run(
            parent_run_dir
        )

    parent_cfg = cfg["parent"]

    technique = parent_cfg[
        "technique"
    ]

    if technique != "mor":
        raise ValueError(
            "Mamba parent must be MoR."
        )

    runs_root = Path(
        cfg["experiment"].get(
            "runs_root",
            "runs",
        )
    )

    if not runs_root.is_absolute():
        runs_root = (
            PROJECT_ROOT
            / runs_root
        )

    technique_dir = (
        runs_root
        / technique
    )

    if not technique_dir.exists():
        raise FileNotFoundError(
            f"No parent runs found: "
            f"{technique_dir}"
        )

    selected_run = parent_cfg.get(
        "selected_run",
        "auto",
    )

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    if selected_run == "auto":

        run_dirs = sorted(
            technique_dir.glob(
                "run_*"
            ),
            reverse=True,
        )

        for run_dir in run_dirs:

            status_path = (
                run_dir
                / "status.json"
            )

            if not status_path.exists():
                continue

            with status_path.open(
                "r"
            ) as f:
                status = json.load(f)

            if (
                status.get("status")
                != "completed"
            ):
                continue

            try:
                return checkpoint_from_run(
                    run_dir
                )

            except FileNotFoundError:
                continue

        raise FileNotFoundError(
            "No completed MoR run "
            "with checkpoint found."
        )

    # --------------------------------------------------------
    # SPECIFIC RUN
    # --------------------------------------------------------

    if isinstance(
        selected_run,
        int,
    ):
        run_name = (
            f"run_{selected_run:03d}"
        )

    else:
        run_name = str(
            selected_run
        )

        if not run_name.startswith(
            "run_"
        ):
            run_name = (
                f"run_"
                f"{int(run_name):03d}"
            )

    return checkpoint_from_run(
        technique_dir
        / run_name
    )


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_from_mor(
    model,
    checkpoint_path,
):

    print(
        "\nLoading MoR parent:"
    )

    print(
        checkpoint_path
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
            "Parent checkpoint has no "
            "model_state_dict."
        )

    parent_cfg = checkpoint[
        "config"
    ]

    parent_type = (
        parent_cfg
        .get(
            "technique",
            {},
        )
        .get(
            "model_type"
        )
    )

    if parent_type != "mor":
        raise ValueError(
            f"Expected MoR parent, "
            f"received {parent_type}"
        )

    # --------------------------------------------------------
    # ARCHITECTURAL COMPATIBILITY
    # --------------------------------------------------------

    parent_model_cfg = (
        parent_cfg["model"]
    )

    checks = [
        "vocab_size",
        "d_model",
        "max_seq_len",
    ]

    for key in checks:

        parent_value = (
            parent_model_cfg[key]
        )

        child_value = getattr(
            model.cfg,
            key,
        )

        if parent_value != child_value:
            raise ValueError(
                f"Parent/child mismatch "
                f"for {key}: "
                f"{parent_value} != "
                f"{child_value}"
            )

    return (
        model.load_mor_parent_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )
    )


# ============================================================
# TRAIN
# ============================================================

def train_mamba(
    config_path=CONFIG_PATH,
    parent_run_dir=None,
    steps=None,
):

    cfg = load_config(
        config_path
    )

    if (
        cfg["technique"][
            "model_type"
        ]
        != "mamba"
    ):
        raise ValueError(
            "Mamba config requires "
            "model_type='mamba'."
        )

    if (
        cfg["initialization"][
            "mode"
        ]
        != "warm_start"
    ):
        raise ValueError(
            "Mamba stage must use "
            "warm_start."
        )

    # ========================================================
    # SETUP
    # ========================================================

    set_seed(
        cfg["experiment"][
            "seed"
        ]
    )

    device = choose_device()

    print(
        f"Device: {device}"
    )

    print(
        f"Technique: "
        f"{cfg['technique']['name']}"
    )

    print(
        f"Scale: "
        f"{cfg['experiment']['scale']}"
    )

    # ========================================================
    # PARENT
    # ========================================================

    parent_checkpoint = (
        find_parent_checkpoint(
            cfg,
            parent_run_dir,
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model_cfg = MambaModelConfig(
        **cfg["model"]
    )

    model = MambaCausalLM(
        model_cfg
    )

    initialization_report = (
        initialize_from_mor(
            model,
            parent_checkpoint,
        )
    )

    print(
        "\nInitialization:"
    )

    print(
        f"Transferred parameters: "
        f"{initialization_report['transferred_parameter_count']:,}"
    )

    print(
        f"New Mamba parameters: "
        f"{initialization_report['new_parameter_count']:,}"
    )

    print(
        f"Transfer fraction: "
        f"{initialization_report['transferred_parameter_fraction']:.4f}"
    )

    model = model.to(
        device
    )

    print(
        "\nParameters:",
        model.parameter_report(),
    )

    print(
        "Architecture:",
        model.architecture_report(),
    )

    # ========================================================
    # DATA
    # ========================================================

    train_loader = (
        build_train_loader(
            cfg,
            model_cfg,
        )
    )

    val_batches = (
        build_validation_batches(
            cfg,
            model_cfg,
        )
    )

    # ========================================================
    # TRAINING
    # ========================================================

    max_steps = (
        steps
        if steps is not None
        else cfg["training"][
            "max_steps"
        ]
    )

    optimizer = build_optimizer(
        model,
        cfg["training"],
    )

    scheduler = build_scheduler(
        optimizer,
        cfg["training"],
        max_steps,
    )

    # ========================================================
    # RUN
    # ========================================================

    run_dir = create_run_dir(
        cfg,
        "mamba",
    )

    cfg["runtime"] = {
        "parent_checkpoint":
            str(parent_checkpoint),

        "parent_run_dir":
            (
                str(parent_run_dir)
                if parent_run_dir
                else None
            ),

        "initialization_report":
            initialization_report,
    }

    save_resolved_config(
        run_dir,
        cfg,
    )

    save_json(
        run_dir
        / "status.json",
        {
            "status":
                "running",

            "stage":
                "s01",

            "technique":
                "mamba",

            "parent_checkpoint":
                str(
                    parent_checkpoint
                ),
        },
    )

    # ========================================================
    # EXECUTE
    # ========================================================

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

        summary[
            "parent_checkpoint"
        ] = str(
            parent_checkpoint
        )

        summary[
            "initialization_report"
        ] = (
            initialization_report
        )

        save_json(
            run_dir
            / "run_summary.json",
            summary,
        )

        save_json(
            run_dir
            / "status.json",
            {
                "status":
                    "completed",

                "stage":
                    "s01",

                "technique":
                    "mamba",

                "parent_checkpoint":
                    str(
                        parent_checkpoint
                    ),

                "final_checkpoint":
                    summary[
                        "final_checkpoint"
                    ],
            },
        )

        print(
            "\nMamba training completed."
        )

        print(
            f"Run: {run_dir}"
        )

        print(
            f"Checkpoint: "
            f"{summary['final_checkpoint']}"
        )

        return summary

    except Exception as error:

        save_json(
            run_dir
            / "status.json",
            {
                "status":
                    "failed",

                "stage":
                    "s01",

                "technique":
                    "mamba",

                "parent_checkpoint":
                    str(
                        parent_checkpoint
                    ),

                "error":
                    str(error),
            },
        )

        raise