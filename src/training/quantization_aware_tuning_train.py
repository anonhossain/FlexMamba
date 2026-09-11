import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.models.quantization_aware_tuning_model import (
    QuantizationAwareTuningModelConfig,
    QuantizationAwareTuningCausalLM,
)

from src.evaluation.quantization_aware_tuning_eval import (
    evaluate_qat_batches,
    evaluate_quantization_aware_tuning,
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
)


CONFIG_PATH = (
    "src/configs/techniques/"
    "s09_quantization_aware_tuning.yaml"
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# CHECKPOINT
# ============================================================

def checkpoint_from_run(run_dir):

    run_dir = resolve_path(
        run_dir
    )

    summary_path = (
        run_dir
        / "run_summary.json"
    )

    if summary_path.exists():

        with summary_path.open(
            "r"
        ) as file:

            summary = json.load(
                file
            )

        checkpoint = resolve_path(
            summary[
                "final_checkpoint"
            ]
        )

        if checkpoint.exists():
            return checkpoint

    checkpoints = sorted(
        (
            run_dir
            / "checkpoints"
        ).glob(
            "step_*.pt"
        )
    )

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {run_dir}"
        )

    return checkpoints[-1]


# ============================================================
# FIND S08
# ============================================================

def find_parent_checkpoint(
    cfg,
    parent_run_dir=None,
):

    if parent_run_dir is not None:
        return checkpoint_from_run(
            parent_run_dir
        )

    if (
        cfg["parent"][
            "technique"
        ]
        != "turboquant_state"
    ):
        raise ValueError(
            "S09 parent must be turboquant_state."
        )

    runs_root = Path(
        cfg[
            "experiment"
        ].get(
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
        / "turboquant_state"
    )

    selected_run = cfg[
        "parent"
    ].get(
        "selected_run",
        "auto",
    )

    if selected_run == "auto":

        runs = sorted(
            technique_dir.glob(
                "run_*"
            ),

            key=lambda path: int(
                path.name.split(
                    "_"
                )[-1]
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

            with status_path.open(
                "r"
            ) as file:

                status = json.load(
                    file
                )

            if (
                status.get(
                    "status"
                )
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
            "No completed S08 run found."
        )

    run_name = (
        f"run_{selected_run:03d}"
        if isinstance(
            selected_run,
            int,
        )
        else str(
            selected_run
        )
    )

    if not run_name.startswith(
        "run_"
    ):
        run_name = (
            f"run_{int(run_name):03d}"
        )

    return checkpoint_from_run(
        technique_dir
        / run_name
    )


# ============================================================
# INITIALIZE FROM S08
# ============================================================

def initialize_from_parent(
    model,
    checkpoint_path,
):

    print(
        "\nLoading TurboQuant State parent:\n"
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError(
            "S08 checkpoint has no config."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "S08 checkpoint has no model_state_dict."
        )

    if (
        "quantization_state"
        not in checkpoint
    ):
        raise KeyError(
            "S08 checkpoint has no quantization_state."
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

    if parent_type != "turboquant_mamba":
        raise ValueError(
            "Expected turboquant_mamba parent, "
            f"received {parent_type}"
        )

    parent_model = parent_cfg[
        "model"
    ]

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

        parent_value = (
            parent_model[key]
        )

        child_value = getattr(
            model.cfg,
            key,
        )

        if (
            parent_value
            != child_value
        ):
            raise ValueError(
                f"Parent/child mismatch for {key}: "
                f"{parent_value} != {child_value}"
            )

    report = (
        model.load_parent_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            checkpoint[
                "quantization_state"
            ],
        )
    )

    model.set_state_mode(
        "fake_quant"
    )

    return report


# ============================================================
# SAVE QAT CHECKPOINT
# ============================================================

def save_qat_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    cfg,
):

    torch.save(
        {
            "step": step,

            "model_state_dict": (
                model.state_dict()
            ),

            "optimizer_state_dict": (
                optimizer.state_dict()
            ),

            "scheduler_state_dict": (
                scheduler.state_dict()
            ),

            "config": cfg,

            "quantization_state": (
                model.quantization_state_dict()
            ),
        },
        path,
    )


# ============================================================
# CUSTOM QAT TRAIN LOOP
# ============================================================

def train_qat_model(
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

    train_cfg = cfg[
        "training"
    ]

    val_cfg = cfg.get(
        "validation",
        {},
    )

    grad_accum = train_cfg[
        "grad_accum_steps"
    ]

    sequence_tokens = train_cfg.get(
        "sequence_tokens",
        model.cfg.max_seq_len,
    )

    save_steps = set(
        cfg.get(
            "checkpoint",
            {},
        ).get(
            "save_steps",
            [],
        )
    )

    save_steps.add(
        max_steps
    )

    checkpoint_dir = (
        Path(run_dir)
        / "checkpoints"
    )

    train_iter = iter(
        train_loader
    )

    model.train()
    model.set_state_mode(
        "fake_quant"
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    start_time = (
        time.perf_counter()
    )

    final_loss = None
    validation_history = []

    progress = tqdm(
        range(
            1,
            max_steps + 1,
        ),
        desc="qat_training",
    )

    for step in progress:

        step_loss = 0.0

        for _ in range(
            grad_accum
        ):

            try:
                batch = next(
                    train_iter
                )

            except StopIteration:
                train_iter = iter(
                    train_loader
                )
                batch = next(
                    train_iter
                )

            batch = batch[
                :,
                :sequence_tokens + 1,
            ].to(device)

            x = batch[:, :-1]
            y = batch[:, 1:]

            output = model(
                x,
                labels=y,
            )

            loss = output["loss"]

            if not torch.isfinite(
                loss
            ).item():

                raise FloatingPointError(
                    f"Non-finite QAT loss at step {step}"
                )

            (
                loss
                / grad_accum
            ).backward()

            step_loss += (
                loss.detach()
                .float()
                .item()
                / grad_accum
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_cfg[
                "grad_clip"
            ],
        )

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        final_loss = (
            step_loss
        )

        if (
            step == 1
            or step
            % train_cfg[
                "log_every"
            ]
            == 0
        ):

            progress.set_postfix(
                loss=f"{step_loss:.3f}",
                lr=(
                    f"{scheduler.get_last_lr()[0]:.2e}"
                ),
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        should_validate = (
            val_cfg.get(
                "enabled",
                False,
            )
            and val_batches
            and (
                step
                % val_cfg.get(
                    "eval_every",
                    max_steps,
                )
                == 0
                or step
                == max_steps
            )
        )

        if should_validate:

            val_sequence_tokens = (
                val_cfg.get(
                    "sequence_tokens",
                    sequence_tokens,
                )
            )

            result = (
                evaluate_qat_batches(
                    model,
                    val_batches,
                    device,
                    state_mode="fake_quant",
                    sequence_tokens=(
                        val_sequence_tokens
                    ),
                )
            )

            validation_entry = {
                "step": step,

                "nll": (
                    result[
                        "quality"
                    ][
                        "nll"
                    ]
                ),

                "perplexity": (
                    result[
                        "quality"
                    ][
                        "perplexity"
                    ]
                ),

                "state_quantization_mse": (
                    result[
                        "quantization"
                    ][
                        "mse"
                    ]
                ),
            }

            validation_history.append(
                validation_entry
            )

            model.train()
            model.set_state_mode(
                "fake_quant"
            )

            print(
                f"\nValidation step {step}: "
                f"NLL={validation_entry['nll']:.4f} | "
                f"PPL={validation_entry['perplexity']:.2f} | "
                f"State-MSE="
                f"{validation_entry['state_quantization_mse']:.8f}"
            )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        if step in save_steps:

            checkpoint_path = (
                checkpoint_dir
                / f"step_{step:07d}.pt"
            )

            save_qat_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                step,
                cfg,
            )

            print(
                f"\nSaved QAT checkpoint: "
                f"{checkpoint_path}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    final_checkpoint = (
        checkpoint_dir
        / f"step_{max_steps:07d}.pt"
    )

    tokens_per_step = (
        train_cfg[
            "batch_size"
        ]
        * grad_accum
        * sequence_tokens
    )

    training_tokens = (
        tokens_per_step
        * max_steps
    )

    return {
        "technique": (
            cfg["technique"][
                "name"
            ]
        ),

        "scale": (
            cfg["experiment"][
                "scale"
            ]
        ),

        "debug": (
            cfg["experiment"].get(
                "debug",
                False,
            )
        ),

        "run": (
            Path(run_dir).name
        ),

        "run_dir": str(
            run_dir
        ),

        "completed_steps": (
            max_steps
        ),

        "sequence_tokens": (
            sequence_tokens
        ),

        "tokens_per_step": (
            tokens_per_step
        ),

        "training_tokens": (
            training_tokens
        ),

        "final_training_loss": (
            final_loss
        ),

        "training_seconds": (
            elapsed
        ),

        "seconds_per_step": (
            elapsed
            / max_steps
        ),

        "tokens_per_second": (
            training_tokens
            / elapsed
        ),

        "validation_history": (
            validation_history
        ),

        "final_validation": (
            validation_history[-1]
            if validation_history
            else None
        ),

        "final_checkpoint": str(
            final_checkpoint
        ),

        "parameter_report": (
            model.parameter_report()
        ),

        "architecture_report": (
            model.architecture_report()
        ),
    }


# ============================================================
# S09 TRAIN
# ============================================================

def train_quantization_aware_tuning(
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
        != "qat_joint_mamba"
    ):
        raise ValueError(
            "S09 requires model_type='qat_joint_mamba'."
        )

    if not cfg[
        "training"
    ].get(
        "enabled",
        True,
    ):
        raise ValueError(
            "S09 requires training.enabled=true."
        )

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

    parent_checkpoint = (
        find_parent_checkpoint(
            cfg,
            parent_run_dir,
        )
    )

    model_cfg = (
        QuantizationAwareTuningModelConfig(
            **cfg["model"]
        )
    )

    model = (
        QuantizationAwareTuningCausalLM(
            model_cfg,
            cfg["recursion"],
            cfg["state"],
            cfg["routing"],
            cfg[
                "grouped_parameterization"
            ],
            cfg["lte"],
            cfg[
                "state_quantization"
            ],
            cfg[
                "quantization"
            ],
            seed=cfg[
                "experiment"
            ][
                "seed"
            ],
        )
    )

    init_report = (
        initialize_from_parent(
            model,
            parent_checkpoint,
        )
    )

    print(
        "\nInitialization:"
    )

    print(
        f"Strategy: "
        f"{init_report['strategy']}"
    )

    print(
        "Transferred parameters: "
        f"{init_report['transferred_parameter_count']:,}"
    )

    print(
        "New trainable parameters: "
        f"{init_report['new_trainable_parameters']:,}"
    )

    print(
        "Quantizers: "
        f"{init_report['quantizer_count']}"
    )

    print(
        "Quantizer scales reused: "
        f"{init_report['quantizer_scales_reused']}"
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

    max_steps = (
        steps
        if steps is not None
        else cfg[
            "training"
        ][
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

    run_dir = create_run_dir(
        cfg,
        "quantization_aware_tuning",
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
        run_dir
        / "status.json",
        {
            "status": "running",
            "stage": "s09",
            "technique": (
                "quantization_aware_tuning"
            ),
            "parent_checkpoint": str(
                parent_checkpoint
            ),
        },
    )

    try:

        summary = train_qat_model(
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
        ] = init_report

        save_json(
            run_dir
            / "run_summary.json",
            summary,
        )

        evaluation = (
            evaluate_quantization_aware_tuning(
                summary
            )
        )

        save_json(
            run_dir
            / "status.json",
            {
                "status": "completed",

                "stage": "s09",

                "technique": (
                    "quantization_aware_tuning"
                ),

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
            run_dir
            / "status.json",
            {
                "status": "failed",

                "stage": "s09",

                "technique": (
                    "quantization_aware_tuning"
                ),

                "parent_checkpoint": str(
                    parent_checkpoint
                ),

                "error": str(
                    error
                ),
            },
        )

        raise


# ============================================================
# RUN
# ============================================================

def run_quantization_aware_tuning(
    parent_run_dir=None,
    steps=None,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S09 — QUANTIZATION-AWARE TUNING"
    )

    print(
        "=" * 60
    )

    summary, evaluation = (
        train_quantization_aware_tuning(
            parent_run_dir=parent_run_dir,
            steps=steps,
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "S09 QUANTIZATION-AWARE TUNING COMPLETED"
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
        "S08 corrected INT8 PPL: "
        f"{evaluation['parent_quantized']['quality']['perplexity']:.2f}"
    )

    print(
        "S09 INT8 PPL: "
        f"{evaluation['tuned_quantized']['quality']['perplexity']:.2f}"
    )

    print(
        "QAT improvement: "
        f"{evaluation['quality_change']['qat_relative_improvement_vs_parent'] * 100:.3f}%"
    )

    print(
        "State compression: "
        f"{evaluation['compression']['total_state_compression_ratio']:.2f}x"
    )

    return {
        "training": summary,
        "evaluation": evaluation,
    }