import math
import random

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, cfg):
    return AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )


def make_scheduler(optimizer, cfg, max_steps):
    warmup_steps = max(
        1,
        int(max_steps * cfg["warmup_ratio"]),
    )

    min_ratio = (
        cfg["min_learning_rate"]
        / cfg["learning_rate"]
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (
            step - warmup_steps
        ) / max(
            1,
            max_steps - warmup_steps,
        )

        progress = min(progress, 1.0)

        cosine = 0.5 * (
            1.0
            + math.cos(math.pi * progress)
        )

        return (
            min_ratio
            + (1.0 - min_ratio) * cosine
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


def load_parent(
    model,
    checkpoint_path,
    cfg,
    strict=False,
):
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    parent_state = checkpoint[
        "model_state_dict"
    ]

    if strict:
        model.load_state_dict(
            parent_state,
            strict=True,
        )

        return {
            "strategy": "strict",
            "checkpoint": checkpoint_path,
        }

    strategy = (
        cfg.get("initialization", {})
        .get("strategy", "strict")
    )

    if hasattr(
        model,
        "load_parent_state_dict",
    ):
        report = model.load_parent_state_dict(
            parent_state,
            parent_config=checkpoint.get(
                "config",
                {},
            ),
            strategy=strategy,
        )

    else:
        model.load_state_dict(
            parent_state,
            strict=True,
        )

        report = {
            "strategy": "strict",
        }

    report["checkpoint"] = (
        checkpoint_path
    )

    return report