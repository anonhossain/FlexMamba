import math

import torch


@torch.inference_mode()
def validate(
    model,
    loader,
    device,
    max_batches,
):
    was_training = model.training

    model.eval()

    losses = []
    depths = []

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break

        batch = batch.to(device)

        x = batch[:, :-1]
        y = batch[:, 1:]

        output = model(
            x,
            labels=y,
        )

        losses.append(
            output["lm_loss"]
            .detach()
            .float()
            .cpu()
        )

        depth = output.get(
            "avg_recursion_depth",
            output.get("avg_depth"),
        )

        if depth is not None:
            if torch.is_tensor(depth):
                depth = depth.item()

            depths.append(float(depth))

    if was_training:
        model.train()

    if not losses:
        return {
            "nll": float("nan"),
            "ppl": float("nan"),
        }

    nll = (
        torch.stack(losses)
        .mean()
        .item()
    )

    result = {
        "nll": nll,
        "ppl": math.exp(
            min(nll, 20.0)
        ),
    }

    if depths:
        result["avg_depth"] = (
            sum(depths)
            / len(depths)
        )

    return result