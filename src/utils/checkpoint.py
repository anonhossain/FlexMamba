import json
from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    config,
    evaluation=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "parameter_report": model.parameter_report(),
        "evaluation": evaluation,
    }

    torch.save(payload, path)

    meta_path = path.with_suffix(".json")

    with meta_path.open("w") as f:
        json.dump(
            {
                "step": step,
                "parameter_report": model.parameter_report(),
                "evaluation": evaluation,
                "checkpoint": str(path),
            },
            f,
            indent=2,
        )