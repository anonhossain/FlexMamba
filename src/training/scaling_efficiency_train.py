import copy
import math
from dataclasses import replace
from pathlib import Path

import torch

from src.models.scaling_efficiency_model import (
    ScalingStudyModelConfig,
    ScalingStudyCausalLM,
)

from src.training.quantization_aware_tuning_train import (
    train_qat_model,
)

from src.evaluation.scaling_efficiency_eval import (
    evaluate_scaling_target,
    save_scaling_summary,
)

from src.training.common_16m import (
    PROJECT_ROOT,
    load_config,
    choose_device,
    set_seed,
    build_train_loader,
    build_validation_batches,
    build_optimizer,
    save_json,
    save_resolved_config,
)


CONFIG_PATH = (
    "src/configs/techniques/"
    "s11_scaling_study.yaml"
)


# ============================================================
# MODEL CONFIG
# ============================================================

def build_model_cfg(cfg, d_model):

    template = copy.deepcopy(
        cfg["model_template"]
    )

    template["d_model"] = d_model

    return ScalingStudyModelConfig(
        **template
    )


# ============================================================
# META PARAMETER COUNT
# ============================================================

def count_scale_parameters(
    cfg,
    d_model,
):

    model_cfg = build_model_cfg(
        cfg,
        d_model,
    )

    try:

        with torch.device("meta"):

            model = ScalingStudyCausalLM(
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
                cfg["quantization"],
                seed=cfg[
                    "experiment"
                ][
                    "seed"
                ],
            )

        report = (
            model.scale_parameter_report()
        )

        del model

        return report

    except Exception as error:

        raise RuntimeError(
            "Meta-device parameter planning failed. "
            "Do not fall back to allocating the 350M model "
            "just to count parameters."
        ) from error


# ============================================================
# TARGET PLANNER
# ============================================================

def plan_target(
    cfg,
    target,
):

    scaling = cfg["scaling"]

    basis = scaling.get(
        "target_basis",
        "unrolled_dense_equivalent_parameters",
    )

    target_parameters = int(
        target["target_parameters"]
    )

    multiple = scaling.get(
        "width_multiple",
        64,
    )

    low = math.ceil(
        scaling.get(
            "minimum_d_model",
            128,
        )
        / multiple
    )

    high = math.floor(
        scaling.get(
            "maximum_d_model",
            8192,
        )
        / multiple
    )

    best = None

    while low <= high:

        middle = (
            low + high
        ) // 2

        d_model = (
            middle
            * multiple
        )

        report = count_scale_parameters(
            cfg,
            d_model,
        )

        measured = report[basis]

        error = abs(
            measured
            - target_parameters
        )

        if (
            best is None
            or error < best["absolute_error"]
        ):

            best = {
                "name": target["name"],
                "target_parameters": target_parameters,
                "d_model": d_model,

                "absolute_error": error,

                "parameter_report": report,
            }

        if measured < target_parameters:
            low = middle + 1
        else:
            high = middle - 1

    if best is None:
        raise RuntimeError(
            f"Could not plan target {target['name']}."
        )

    best["error_percent"] = (
        100.0
        * best["absolute_error"]
        / target_parameters
    )

    tolerance = scaling.get("parameter_tolerance_percent", 5)

    best["within_tolerance"] = best["error_percent"] <= tolerance
    best["requested_scale_name"] = target["name"]
    best["planning_warning"] = None

    if not best["within_tolerance"]:
        best["planning_warning"] = (
            f"Nearest valid FlexMamba configuration differs from "
            f"{target_parameters:,} by {best['error_percent']:.2f}%. "
            f"Using d_model={best['d_model']} with "
            f"{best['parameter_report'][basis]:,} "
            f"{basis} parameters."
        )

    return best

def build_scaling_plan(cfg):

    plans = [
        plan_target(cfg, target)
        for target in cfg["scaling"]["targets"]
    ]

    print("\n" + "=" * 70)
    print("FLEXMAMBA SCALE PLAN")
    print("=" * 70)

    for plan in plans:
        actual = plan["parameter_report"][
            cfg["scaling"]["target_basis"]
        ]

        print(
            f"{plan['name']:>5} | "
            f"d_model={plan['d_model']:>4} | "
            f"target={plan['target_parameters']:,} | "
            f"actual={actual:,} | "
            f"error={plan['error_percent']:.2f}%"
        )

        if plan["planning_warning"]:
            print(f"      WARNING: {plan['planning_warning']}")

    return plans


# ============================================================
# TRAPEZOID SCHEDULER
# ============================================================

def build_trapezoid_scheduler(
    optimizer,
    max_steps,
    warmup_ratio,
    cooldown_ratio,
    minimum_ratio=0.0,
):

    warmup_steps = max(
        1,
        int(
            max_steps
            * warmup_ratio
        ),
    )

    cooldown_steps = max(
        1,
        int(
            max_steps
            * cooldown_ratio
        ),
    )

    cooldown_start = max(
        warmup_steps,
        max_steps
        - cooldown_steps,
    )

    def lr_lambda(step):

        if step < warmup_steps:

            return max(
                minimum_ratio,
                step
                / warmup_steps,
            )

        if step < cooldown_start:
            return 1.0

        progress = (
            step
            - cooldown_start
        ) / max(
            1,
            max_steps
            - cooldown_start,
        )

        return max(
            minimum_ratio,
            1.0 - progress,
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ============================================================
# RUN DIRECTORY
# ============================================================

def create_scale_run_dir(
    cfg,
    target_name,
):

    root = Path(
        cfg["experiment"].get(
            "runs_root",
            "runs",
        )
    )

    if not root.is_absolute():
        root = PROJECT_ROOT / root

    target_root = (
        root
        / "scaling_study"
        / target_name
    )

    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = [
        int(
            path.name.split("_")[-1]
        )
        for path in target_root.glob(
            "run_*"
        )
        if path.name.split("_")[-1].isdigit()
    ]

    run_number = (
        max(existing) + 1
        if existing
        else 1
    )

    run_dir = (
        target_root
        / f"run_{run_number:03d}"
    )

    (
        run_dir
        / "checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    return run_dir


# ============================================================
# CALIBRATION
# ============================================================

@torch.inference_mode()
def calibrate_model(
    model,
    loader,
    device,
    calibration_batches,
    sequence_tokens,
):

    print(
        f"Calibrating {len(model.state_quantizers)} "
        "recurrent-state quantizers..."
    )

    model.eval()
    model.start_state_calibration()

    iterator = iter(loader)

    for _ in range(
        calibration_batches
    ):

        batch = next(iterator)

        batch = batch[
            :,
            :sequence_tokens + 1,
        ].to(device)

        model(
            batch[:, :-1]
        )

    model.finish_state_calibration()

    model.train()

    print(
        "State calibration completed."
    )


# ============================================================
# BUDGET
# ============================================================

def resolve_training_budget(
    cfg,
    target_plan,
    sequence_tokens,
    batch_size,
    grad_accum_steps,
):

    smoke = cfg.get(
        "smoke",
        {},
    )

    if smoke.get(
        "enabled",
        False,
    ):

        return smoke.get(
            "max_steps",
            1,
        )

    budget = cfg["budget"]

    tokens_per_step = (
        batch_size
        * grad_accum_steps
        * sequence_tokens
    )

    if (
        budget["protocol"]
        == "fixed_tokens"
    ):

        training_tokens = int(
            budget[
                "training_tokens_per_scale"
            ]
        )

        return math.ceil(
            training_tokens
            / tokens_per_step
        )

    if (
        budget["protocol"]
        == "iso_flop"
    ):

        flops_budget = float(
            budget[
                "flops_budget"
            ]
        )

        base_parameters = (
            target_plan[
                "parameter_report"
            ][
                "unrolled_dense_equivalent_parameters"
            ]
        )

        flops_per_token = (
            budget.get(
                "flops_per_token_factor",
                6.0,
            )
            * base_parameters
        )

        training_tokens = (
            flops_budget
            / flops_per_token
        )

        return math.ceil(
            training_tokens
            / tokens_per_step
        )

    raise ValueError(
        "Unknown scaling budget protocol."
    )


# ============================================================
# ONE SCALE
# ============================================================

def train_one_scale(
    base_cfg,
    plan,
):

    cfg = copy.deepcopy(
        base_cfg
    )

    smoke = cfg.get(
        "smoke",
        {},
    )

    smoke_enabled = smoke.get(
        "enabled",
        False,
    )

    target_name = plan["name"]

    cfg["experiment"]["scale"] = (
        target_name
    )

    model_cfg = build_model_cfg(
        cfg,
        plan["d_model"],
    )

    cfg["model"] = {
        key: value
        for key, value
        in vars(model_cfg).items()
    }

    sequence_tokens = (
        smoke.get(
            "sequence_tokens",
            32,
        )
        if smoke_enabled
        else cfg[
            "training"
        ][
            "sequence_tokens"
        ]
    )

    micro_batch = (
        cfg["training"].get(
            "micro_batch_size",
            1,
        )
    )

    global_tokens = (
        cfg["training"].get(
            "global_tokens_per_update",
            8192,
        )
    )

    grad_accum = max(
        1,
        math.ceil(
            global_tokens
            / (
                micro_batch
                * sequence_tokens
            )
        ),
    )

    if smoke_enabled:
        grad_accum = 1

    cfg["training"][
        "batch_size"
    ] = micro_batch

    cfg["training"][
        "grad_accum_steps"
    ] = grad_accum

    cfg["training"][
        "sequence_tokens"
    ] = sequence_tokens

    max_steps = resolve_training_budget(
        cfg,
        plan,
        sequence_tokens,
        micro_batch,
        grad_accum,
    )

    cfg["training"][
        "max_steps"
    ] = max_steps

    cfg["checkpoint"] = {
        "save_steps": [
            max_steps
        ]
    }

    if smoke_enabled:

        cfg["validation"][
            "eval_every"
        ] = 1

        cfg["validation"][
            "eval_batches"
        ] = 1

        cfg["validation"][
            "sequence_tokens"
        ] = sequence_tokens

    else:

        cfg["validation"][
            "sequence_tokens"
        ] = sequence_tokens

    device = choose_device()

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"S11 SCALE — {target_name.upper()}"
    )

    print(
        "=" * 60
    )

    print(
        f"Target base-equivalent parameters: "
        f"{plan['target_parameters']:,}"
    )

    print(
        f"d_model: {plan['d_model']}"
    )

    print(
        "Planned unrolled dense equivalent: "
        f"{plan['parameter_report']['unrolled_dense_equivalent_parameters']:,}"
    )

    print(
        "Planned stored parameters: "
        f"{plan['parameter_report']['stored_parameters']:,}"
    )

    model = ScalingStudyCausalLM(
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
        cfg["quantization"],
        seed=cfg[
            "experiment"
        ][
            "seed"
        ],
    ).to(device)

    actual_report = (
        model.scale_parameter_report()
    )

    train_data_cfg = replace(
        model_cfg,
        max_seq_len=sequence_tokens,
    )

    train_loader = build_train_loader(
        cfg,
        train_data_cfg,
    )

    calibration_batches = (
        smoke.get(
            "calibration_batches",
            1,
        )
        if smoke_enabled
        else cfg[
            "calibration"
        ][
            "batches"
        ]
    )

    calibrate_model(
        model,
        train_loader,
        device,
        calibration_batches,
        sequence_tokens,
    )

    val_batches = (
        build_validation_batches(
            cfg,
            train_data_cfg,
        )
    )

    optimizer = build_optimizer(
        model,
        cfg["training"],
    )

    minimum_ratio = (
        cfg["training"].get(
            "min_learning_rate",
            0.0,
        )
        / cfg["training"][
            "learning_rate"
        ]
    )

    scheduler = (
        build_trapezoid_scheduler(
            optimizer,
            max_steps,
            cfg["training"][
                "warmup_ratio"
            ],
            cfg["training"][
                "cooldown_ratio"
            ],
            minimum_ratio,
        )
    )

    run_dir = create_scale_run_dir(
        cfg,
        target_name,
    )

    cfg["runtime"] = {
        "scaling_plan": plan,

        "initialization": (
            "from_scratch"
        ),

        "parent_weights_loaded": False,
    }

    save_resolved_config(
        run_dir,
        cfg,
    )

    save_json(
        run_dir / "status.json",
        {
            "status": "running",

            "stage": "s11",

            "technique": (
                "scaling_study"
            ),

            "scale": target_name,
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
            "scaling_plan"
        ] = plan

        summary[
            "scale_parameter_report"
        ] = actual_report

        save_json(
            run_dir
            / "run_summary.json",
            summary,
        )

        evaluation = (
            evaluate_scaling_target(
                summary
            )
        )

        save_json(
            run_dir
            / "status.json",
            {
                "status": "completed",

                "stage": "s11",

                "technique": (
                    "scaling_study"
                ),

                "scale": target_name,

                "final_checkpoint": (
                    summary[
                        "final_checkpoint"
                    ]
                ),
            },
        )

        return {
            "training": summary,
            "evaluation": evaluation,
        }

    except Exception as error:

        save_json(
            run_dir
            / "status.json",
            {
                "status": "failed",

                "stage": "s11",

                "scale": target_name,

                "error": str(error),
            },
        )

        raise


# ============================================================
# FULL SCALING STUDY
# ============================================================

def run_scaling_study(
    config_path=CONFIG_PATH,
):

    cfg = load_config(
        config_path
    )

    set_seed(
        cfg["experiment"]["seed"]
    )

    plan = build_scaling_plan(
        cfg
    )

    root = Path(
        cfg["experiment"].get(
            "runs_root",
            "runs",
        )
    )

    if not root.is_absolute():
        root = PROJECT_ROOT / root

    study_root = (
        root
        / "scaling_study"
    )

    study_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        study_root
        / "scaling_plan.json",
        {
            "target_basis": (
                cfg["scaling"][
                    "target_basis"
                ]
            ),

            "plans": plan,
        },
    )

    smoke = cfg.get(
        "smoke",
        {},
    )

    if smoke.get(
        "enabled",
        False,
    ):

        selected = set(
            smoke.get(
                "targets",
                ["16m"],
            )
        )

        plan = [
            item
            for item in plan
            if item["name"] in selected
        ]

    results = []

    for target_plan in plan:

        results.append(
            train_one_scale(
                cfg,
                target_plan,
            )
        )

    save_scaling_summary(
        cfg,
        results,
    )

    return {
        "plan": plan,
        "results": results,
    }