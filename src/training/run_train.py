import argparse
import copy
import gc
import importlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from data.fineweb import PackedFineWebDataset
from training.setup import (
    choose_device,
    load_parent,
    make_optimizer,
    make_scheduler,
    set_seed,
)
from training.trainer import train


# ============================================================
# SUPPORTED MODELS
# ============================================================

MODELS = {
    "mamba": (
        "models.mamba",
        "MambaModelConfig",
        "MambaCausalLM",
    ),
    "recursive_mamba": (
        "models.recursive_mamba",
        "RecursiveMambaConfig",
        "RecursiveMambaCausalLM",
    ),
}


# ============================================================
# CONFIG
# ============================================================

def resolve_path(path):
    """Resolve project-relative paths."""

    path = Path(path)

    if path.is_absolute():
        return path

    # Preferred structure:
    # FlexMamba/configs/...
    project_path = ROOT / path

    if project_path.exists():
        return project_path

    # Compatibility with older:
    # FlexMamba/src/configs/...
    src_path = SRC / path

    if src_path.exists():
        return src_path

    return project_path


def load_yaml(path):
    path = resolve_path(path)

    with path.open("r") as file:
        return yaml.safe_load(file) or {}


def merge(base, override):
    """Recursively merge two dictionaries."""

    result = copy.deepcopy(base)

    for key, value in override.items():

        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = merge(
                result[key],
                value,
            )
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_technique(path):
    """
    Load technique YAML and recursively apply inheritance.

    Example:
        recursive_mamba.yaml
            inherits: mamba
    """

    path = resolve_path(path)

    cfg = load_yaml(path)

    parent = cfg.get("inherits")

    if not parent:
        return cfg

    parent_path = (
        path.parent
        / f"{parent}.yaml"
    )

    return merge(
        load_technique(parent_path),
        cfg,
    )


def load_config(
    base_config,
    technique_config,
):
    return merge(
        load_yaml(base_config),
        load_technique(technique_config),
    )


# ============================================================
# MODEL
# ============================================================

def build_model(cfg):
    """Create model from YAML configuration."""

    model_type = cfg["technique"]["model_type"]

    if model_type not in MODELS:
        raise ValueError(
            f"Unsupported model_type: {model_type}"
        )

    module_name, config_name, model_name = (
        MODELS[model_type]
    )

    module = importlib.import_module(
        module_name
    )

    config_class = getattr(
        module,
        config_name,
    )

    model_class = getattr(
        module,
        model_name,
    )

    # Ignore inherited fields explicitly disabled
    # with YAML `null`.
    model_values = {
        key: value
        for key, value in cfg["model"].items()
        if value is not None
    }

    model_cfg = config_class(
        **model_values
    )

    model = model_class(
        model_cfg
    )

    return model, model_cfg


# ============================================================
# DATA
# ============================================================

def build_loaders(
    cfg,
    model_cfg,
):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(
        data_cfg["tokenizer_name"]
    )

    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({len(tokenizer)}) != "
            f"model vocab ({model_cfg.vocab_size})"
        )

    common = {
        "tokenizer":
            tokenizer,

        "dataset_name":
            data_cfg["dataset_name"],

        "dataset_config":
            data_cfg["dataset_config"],

        "seq_len":
            model_cfg.max_seq_len,

        "validation_docs":
            data_cfg["validation_docs"],

        "seed":
            cfg["experiment"]["seed"],
    }

    train_dataset = PackedFineWebDataset(
        **common,
        mode="train",
        shuffle_buffer=
            data_cfg["shuffle_buffer"],
    )

    val_dataset = PackedFineWebDataset(
        **common,
        mode="validation",
        shuffle_buffer=0,
    )

    batch_size = train_cfg[
        "batch_size"
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    return train_loader, val_loader


# ============================================================
# RUN MANAGEMENT
# ============================================================

def create_run_dir(
    cfg,
    group=None,
):
    runs_root = Path(
        cfg["experiment"].get(
            "runs_root",
            "runs",
        )
    )

    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root

    name = (
        group
        or cfg["technique"]["name"]
    )

    technique_dir = (
        runs_root / name
    )

    technique_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    numbers = []

    for path in technique_dir.glob(
        "run_*"
    ):
        try:
            numbers.append(
                int(
                    path.name.split("_")[-1]
                )
            )
        except ValueError:
            pass

    number = (
        max(numbers) + 1
        if numbers
        else 1
    )

    run_dir = (
        technique_dir
        / f"run_{number:03d}"
    )

    run_dir.mkdir()

    return run_dir


def save_json(
    path,
    data,
):
    with Path(path).open("w") as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_resolved_config(
    run_dir,
    cfg,
):
    with (
        Path(run_dir)
        / "resolved_config.yaml"
    ).open("w") as file:

        yaml.safe_dump(
            cfg,
            file,
            sort_keys=False,
        )


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elif (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.empty_cache()


# ============================================================
# SINGLE TRAINING RUN
# ============================================================

def run_training(
    base_config,
    technique_config,
    steps=None,
    parent_checkpoint=None,
    strict_parent=False,
    run_group=None,
):
    """
    Run one complete training experiment.

    Used by:
        - command line
        - run_pipeline.py

    Returns:
        run summary including final checkpoint.
    """

    cfg = load_config(
        base_config,
        technique_config,
    )

    # Hyperparameter sweeps are intentionally
    # handled separately later.
    if cfg.get(
        "search",
        {},
    ).get(
        "enabled",
        False,
    ):
        raise ValueError(
            "Hyperparameter search is enabled. "
            "Disable it for normal training."
        )

    # --------------------------------------------------------
    # Parent requirement check
    # --------------------------------------------------------

    init_mode = (
        cfg.get(
            "initialization",
            {},
        ).get(
            "mode",
            "scratch",
        )
    )

    if (
        init_mode != "scratch"
        and parent_checkpoint is None
    ):
        raise ValueError(
            f"{cfg['technique']['name']} requires "
            "a parent checkpoint."
        )

    # --------------------------------------------------------
    # Run directory
    # --------------------------------------------------------

    run_dir = create_run_dir(
        cfg,
        run_group,
    )

    save_json(
        run_dir / "status.json",
        {"status": "starting"},
    )

    model = None
    optimizer = None
    scheduler = None
    train_loader = None
    val_loader = None

    try:
        # ----------------------------------------------------
        # Reproducibility + device
        # ----------------------------------------------------

        set_seed(
            cfg["experiment"]["seed"]
        )

        device = choose_device()

        print(
            f"\nDevice: {device}"
        )

        print(
            "Technique: "
            f"{cfg['technique']['name']}"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model, model_cfg = build_model(
            cfg
        )

        parent_path = None

        if parent_checkpoint:
            parent_path = resolve_path(
                parent_checkpoint
            )

            if not parent_path.exists():
                raise FileNotFoundError(
                    f"Parent checkpoint not found: "
                    f"{parent_path}"
                )

        init_report = load_parent(
            model,
            str(parent_path)
            if parent_path
            else None,
            cfg,
            strict=strict_parent,
        )

        model = model.to(device)

        print("\nParameter report:")

        for name, count in (
            model.parameter_report().items()
        ):
            print(
                f"  {name}: "
                f"{count:,} "
                f"({count / 1e6:.3f}M)"
            )

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        train_loader, val_loader = (
            build_loaders(
                cfg,
                model_cfg,
            )
        )

        # ----------------------------------------------------
        # Training settings
        # ----------------------------------------------------

        train_cfg = cfg["training"]

        max_steps = (
            steps
            if steps is not None
            else train_cfg["max_steps"]
        )

        optimizer = make_optimizer(
            model,
            train_cfg,
        )

        scheduler = make_scheduler(
            optimizer,
            train_cfg,
            max_steps,
        )

        # ----------------------------------------------------
        # Record exact run configuration
        # ----------------------------------------------------

        cfg["runtime"] = {
            "run_dir":
                str(run_dir),

            "parent_checkpoint":
                str(parent_path)
                if parent_path
                else None,

            "initialization":
                init_report,

            "max_steps":
                max_steps,
        }

        save_resolved_config(
            run_dir,
            cfg,
        )

        save_json(
            run_dir / "status.json",
            {"status": "running"},
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        summary = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            run_dir=run_dir,
            max_steps=max_steps,
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        save_json(
            run_dir / "run_summary.json",
            summary,
        )

        save_json(
            run_dir / "status.json",
            {
                "status":
                    "completed",

                "final_checkpoint":
                    summary["final_checkpoint"],

                "final_evaluation":
                    summary["final_evaluation"],
            },
        )

        print(
            "\nTraining completed."
        )

        print(
            "Checkpoint: "
            f"{summary['final_checkpoint']}"
        )

        return summary

    except Exception as error:

        save_json(
            run_dir / "status.json",
            {
                "status":
                    "failed",

                "error":
                    str(error),
            },
        )

        raise

    finally:
        del model
        del optimizer
        del scheduler
        del train_loader
        del val_loader

        cleanup_memory()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        default="configs/base_16m.yaml",
    )

    parser.add_argument(
        "--technique",
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--parent",
        default=None,
    )

    parser.add_argument(
        "--strict-parent",
        action="store_true",
    )

    parser.add_argument(
        "--group",
        default=None,
    )

    args = parser.parse_args()

    run_training(
        base_config=args.base,
        technique_config=args.technique,
        steps=args.steps,
        parent_checkpoint=args.parent,
        strict_parent=args.strict_parent,
        run_group=args.group,
    )


if __name__ == "__main__":
    main()