# src/training/mor_16m_train.py

from src.models.mor_16m import (
    MoRModelConfig,
    MoRCausalLM,
)

from src.training.common_16m import (
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


CONFIG_PATH = "src/configs/techniques/s00.yaml"


def train_mor(
    config_path=CONFIG_PATH,
    steps=None,
):
    cfg = load_config(config_path)

    if cfg["technique"]["model_type"] != "mor":
        raise ValueError(
            "s00 config model_type must be 'mor'."
        )

    if cfg["initialization"]["mode"] != "scratch":
        raise ValueError(
            "s00 MoR must initialize from scratch."
        )

    # ========================================================
    # SETUP
    # ========================================================

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

    # ========================================================
    # MODEL
    # ========================================================

    model_cfg = MoRModelConfig(
        **cfg["model"]
    )

    model = MoRCausalLM(
        model_cfg
    ).to(device)

    print(
        "Parameters:",
        model.parameter_report(),
    )

    print(
        "Architecture:",
        model.architecture_report(),
    )

    # ========================================================
    # DATA
    # ========================================================

    train_loader = build_train_loader(
        cfg,
        model_cfg,
    )

    val_batches = build_validation_batches(
        cfg,
        model_cfg,
    )

    # ========================================================
    # TRAINING SETUP
    # ========================================================

    max_steps = (
        steps
        if steps is not None
        else cfg["training"]["max_steps"]
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
        cfg["technique"]["name"],
    )

    save_resolved_config(
        run_dir,
        cfg,
    )

    save_json(
        run_dir / "status.json",
        {
            "status": "running",
            "technique": cfg["technique"]["name"],
            "scale": cfg["experiment"]["scale"],
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

        save_json(
            run_dir / "run_summary.json",
            summary,
        )

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "technique": cfg["technique"]["name"],
                "scale": cfg["experiment"]["scale"],
                "final_checkpoint": (
                    summary["final_checkpoint"]
                ),
            },
        )

        print("\nMoR training completed.")
        print(f"Run: {run_dir}")
        print(
            f"Checkpoint: "
            f"{summary['final_checkpoint']}"
        )

        return summary

    except Exception as error:
        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "technique": cfg["technique"]["name"],
                "error": str(error),
            },
        )

        raise