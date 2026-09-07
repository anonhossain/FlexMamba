#src/training/mamba_train.py
import argparse

from src.models.mamba import (
    MambaModelConfig,
    MambaCausalLM,
)

from src.training.common import (
    build_optimizer,
    build_scheduler,
    build_train_loader,
    choose_device,
    create_run_dir,
    load_config,
    save_json,
    save_resolved_config,
    set_seed,
    train_model,
)


BASE_CONFIG = "src/configs/base_16m.yaml"
MAMBA_CONFIG = "src/configs/techniques/1.mamba.yaml"


def train_mamba(
    base_config=BASE_CONFIG,
    mamba_config=MAMBA_CONFIG,
    steps=None,
):
    """
    Train one standard Mamba baseline run.

    Flow:
        YAML
          ↓
        MambaModelConfig
          ↓
        MambaCausalLM
          ↓
        FineWeb
          ↓
        training
          ↓
        runs/mamba/run_XXX/
    """

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    cfg = load_config(
        base_config,
        mamba_config,
    )

    if cfg["technique"]["model_type"] != "mamba":
        raise ValueError(
            "mamba_train.py requires model_type='mamba'"
        )

    if cfg["initialization"]["mode"] != "scratch":
        raise ValueError(
            "Base Mamba must use initialization.mode='scratch'"
        )

    # --------------------------------------------------------
    # REPRODUCIBILITY + DEVICE
    # --------------------------------------------------------

    set_seed(
        cfg["experiment"]["seed"]
    )

    device = choose_device()

    print(f"Device: {device}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_cfg = MambaModelConfig(
        **cfg["model"]
    )

    model = MambaCausalLM(
        model_cfg
    ).to(device)

    print("\nParameter report:")

    for name, count in (
        model.parameter_report().items()
    ):
        print(
            f"  {name}: "
            f"{count:,} "
            f"({count / 1e6:.3f}M)"
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_loader = build_train_loader(
        cfg,
        model_cfg,
    )

    # --------------------------------------------------------
    # TRAINING SETUP
    # --------------------------------------------------------

    train_cfg = cfg["training"]

    max_steps = (
        steps
        if steps is not None
        else train_cfg["max_steps"]
    )

    optimizer = build_optimizer(
        model,
        train_cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        train_cfg,
        max_steps,
    )

    # --------------------------------------------------------
    # RUN DIRECTORY
    # --------------------------------------------------------

    run_dir = create_run_dir(
        cfg,
        technique_name="mamba",
    )

    cfg["runtime"] = {
        "run_dir": str(run_dir),
        "max_steps": max_steps,
        "parent_checkpoint": None,
    }

    save_resolved_config(
        run_dir,
        cfg,
    )

    save_json(
        run_dir / "status.json",
        {
            "status": "running",
        },
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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

        save_json(
            run_dir / "run_summary.json",
            summary,
        )

        save_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "final_checkpoint":
                    summary["final_checkpoint"],
            },
        )

        print("\nMamba training completed.")

        print(
            "Final checkpoint:"
        )

        print(
            summary["final_checkpoint"]
        )

        return summary

    except Exception as error:
        save_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "error": str(error),
            },
        )

        raise


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        default=BASE_CONFIG,
    )

    parser.add_argument(
        "--config",
        default=MAMBA_CONFIG,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps for smoke testing.",
    )

    args = parser.parse_args()

    train_mamba(
        base_config=args.base,
        mamba_config=args.config,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()