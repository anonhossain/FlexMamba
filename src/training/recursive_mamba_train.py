from pathlib import Path
import torch

from src.models.recursive_mamba import RecursiveMambaConfig, RecursiveMambaCausalLM
from src.training.common import (
    PROJECT_ROOT,
    build_optimizer,
    build_scheduler,
    build_train_loader,
    choose_device,
    create_run_dir,
    load_yaml,
    merge_config,
    save_json,
    save_resolved_config,
    set_seed,
    train_model,
)


BASE_CONFIG = "src/configs/base_16m.yaml"
MAMBA_CONFIG = "src/configs/techniques/1.mamba.yaml"
RECURSIVE_CONFIG = "src/configs/techniques/2.recursive_mamba.yaml"

# CONFIG

def load_recursive_config(base_config=BASE_CONFIG, mamba_config=MAMBA_CONFIG, recursive_config=RECURSIVE_CONFIG):
    base_cfg = load_yaml(base_config)
    mamba_cfg = load_yaml(mamba_config)
    recursive_cfg = load_yaml(recursive_config)

    if recursive_cfg.get("inherits") != "mamba":
        raise ValueError("Recursive-Mamba must inherit from Mamba.")

    merged_cfg = merge_config(base_cfg, mamba_cfg)
    merged_cfg = merge_config(merged_cfg, recursive_cfg)
    return merged_cfg

# PARENT CHECKPOINT

def find_parent_checkpoint(cfg):
    parent_cfg = cfg["parent"]

    technique = parent_cfg["technique"]
    selected_run = parent_cfg["selected_run"]
    checkpoint_step = parent_cfg["checkpoint_step"]

    runs_root = Path(cfg["experiment"].get("runs_root", "runs"))
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root

    technique_dir = runs_root / technique

    if not technique_dir.exists():
        raise FileNotFoundError(f"No runs found for parent technique: {technique}")

    checkpoint_name = f"step_{checkpoint_step:07d}.pt"

    if selected_run == "auto":
        for run_dir in sorted(technique_dir.glob("run_*"), reverse=True):
            checkpoint = run_dir / "checkpoints" / checkpoint_name
            status_file = run_dir / "status.json"

            if checkpoint.exists() and status_file.exists():
                return checkpoint

        raise FileNotFoundError(
            f"No completed {technique} run contains {checkpoint_name}"
        )

    if isinstance(selected_run, int):
        run_name = f"run_{selected_run:03d}"
    else:
        run_name = str(selected_run)

        if not run_name.startswith("run_"):
            run_name = f"run_{int(run_name):03d}"

    checkpoint = technique_dir / run_name / "checkpoints" / checkpoint_name

    if not checkpoint.exists():
        raise FileNotFoundError(f"Parent checkpoint not found: {checkpoint}")

    return checkpoint


# PARENT INITIALIZATION

def initialize_from_parent(model, checkpoint_path, cfg):
    print(f"\nLoading parent checkpoint:\n{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Parent checkpoint does not contain 'model_state_dict'.")

    if "config" not in checkpoint:
        raise KeyError("Parent checkpoint does not contain its training config.")

    parent_config = checkpoint["config"]
    parent_technique = parent_config.get("technique", {}).get("name")
    expected_parent = cfg["parent"]["technique"]

    if parent_technique and parent_technique != expected_parent:
        raise ValueError(
            f"Expected parent technique '{expected_parent}', got '{parent_technique}'."
        )

    strategy = cfg["initialization"].get("strategy", "cycle_average")

    return model.load_parent_state_dict(
        parent_state=checkpoint["model_state_dict"],
        parent_config=parent_config,
        strategy=strategy,
    )

# ============================================================
# TRAIN
# ============================================================

def train_recursive_mamba(
    base_config=BASE_CONFIG,
    mamba_config=MAMBA_CONFIG,
    recursive_config=RECURSIVE_CONFIG,
    steps=None,
):
    cfg = load_recursive_config(base_config, mamba_config, recursive_config)

    if cfg["technique"]["model_type"] != "recursive_mamba":
        raise ValueError("recursive_mamba_train.py requires model_type='recursive_mamba'.")

    if cfg["initialization"]["mode"] != "warm_start":
        raise ValueError("Recursive-Mamba must use initialization.mode='warm_start'.")

    set_seed(cfg["experiment"]["seed"])
    device = choose_device()
    print(f"Device: {device}")

    parent_checkpoint = find_parent_checkpoint(cfg)

    model_values = {k: v for k, v in cfg["model"].items() if v is not None}
    model_cfg = RecursiveMambaConfig(**model_values)

    print("\nArchitecture:")
    print(f"  Physical layers: {model_cfg.physical_layers}")
    print(f"  Effective layers: {model_cfg.effective_layers}")
    print(f"  Recursions: {model_cfg.num_recursions}")

    model = RecursiveMambaCausalLM(model_cfg)
    init_report = initialize_from_parent(model, parent_checkpoint, cfg)

    print("\nInitialization report:")
    print(f"  Strategy: {init_report['strategy']}")
    print(f"  Loaded fraction: {init_report['loaded_parameter_fraction']:.4f}")

    if init_report["missing_keys"]:
        print("  Missing keys:")
        for key in init_report["missing_keys"]:
            print(f"    {key}")

    model = model.to(device)

    print("\nParameter report:")
    for name, count in model.parameter_report().items():
        print(f"  {name}: {count:,} ({count / 1e6:.3f}M)")

    train_loader = build_train_loader(cfg, model_cfg)
    train_cfg = cfg["training"]
    max_steps = steps if steps is not None else train_cfg["max_steps"]

    # Fresh optimizer/scheduler after architecture conversion.
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, max_steps)

    run_dir = create_run_dir(cfg, technique_name="recursive_mamba")

    cfg["runtime"] = {
        "run_dir": str(run_dir),
        "max_steps": max_steps,
        "parent_checkpoint": str(parent_checkpoint),
        "initialization_report": init_report,
    }

    save_resolved_config(run_dir, cfg)

    save_json(
        run_dir / "status.json",
        {
            "status": "running",
            "parent_checkpoint": str(parent_checkpoint),
        },
    )

    try:
        summary = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg,
            run_dir=run_dir,
            max_steps=max_steps,
        )

        summary["parent_checkpoint"] = str(parent_checkpoint)
        summary["initialization_report"] = init_report
        summary["architecture_report"] = model.architecture_report()

        save_json(run_dir / "run_summary.json", summary)

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "parent_checkpoint": str(parent_checkpoint),
                "final_checkpoint": summary["final_checkpoint"],
            },
        )

        print("\nRecursive-Mamba training completed.")
        print(f"Final checkpoint:\n{summary['final_checkpoint']}")

        return summary

    except Exception as error:
        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "parent_checkpoint": str(parent_checkpoint),
                "error": str(error),
            },
        )
        raise