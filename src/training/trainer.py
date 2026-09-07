import time
from pathlib import Path

import torch
from tqdm import tqdm

from training.validation import validate
from utils.checkpoint import save_checkpoint


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    cfg,
    run_dir,
    max_steps,
):
    train_cfg = cfg["training"]
    ckpt_cfg = cfg.get("checkpoint", {})

    grad_accum = train_cfg[
        "grad_accum_steps"
    ]

    checkpoint_dir = (
        Path(run_dir)
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_steps = set(
        ckpt_cfg.get(
            "save_steps",
            [],
        )
    )

    if not save_steps:
        save_every = train_cfg.get(
            "save_every",
            500,
        )

        save_steps = set(
            range(
                save_every,
                max_steps + 1,
                save_every,
            )
        )

    # Guarantees a checkpoint for smoke tests.
    save_steps.add(max_steps)

    model.train()

    train_iter = iter(train_loader)

    optimizer.zero_grad(
        set_to_none=True
    )

    latest_eval = None
    start_time = time.perf_counter()

    progress = tqdm(
        range(1, max_steps + 1),
        desc="training",
    )

    for step in progress:
        step_loss = 0.0
        step_depth = []

        for _ in range(grad_accum):
            try:
                batch = next(train_iter)

            except StopIteration:
                train_iter = iter(
                    train_loader
                )
                batch = next(train_iter)

            batch = batch.to(device)

            x = batch[:, :-1]
            y = batch[:, 1:]

            output = model(
                x,
                labels=y,
            )

            loss = output["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Invalid loss at step {step}"
                )

            (
                loss / grad_accum
            ).backward()

            step_loss += (
                loss.detach().item()
                / grad_accum
            )

            depth = output.get(
                "avg_recursion_depth",
                output.get("avg_depth"),
            )

            if depth is not None:
                if torch.is_tensor(depth):
                    depth = depth.item()

                step_depth.append(depth)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_cfg["grad_clip"],
        )

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        if (
            step == 1
            or step
            % train_cfg["log_every"]
            == 0
        ):
            info = {
                "loss": f"{step_loss:.3f}",
                "lr":
                    f"{scheduler.get_last_lr()[0]:.2e}",
            }

            if step_depth:
                info["depth"] = (
                    f"{sum(step_depth) / len(step_depth):.2f}"
                )

            progress.set_postfix(info)

        if (
            step
            % train_cfg["eval_every"]
            == 0
            or step in save_steps
        ):
            latest_eval = validate(
                model,
                val_loader,
                device,
                train_cfg["eval_batches"],
            )

            latest_eval["step"] = step

            print(
                f"\n[eval {step}] "
                f"NLL={latest_eval['nll']:.4f} "
                f"PPL={latest_eval['ppl']:.2f}"
            )

        if step in save_steps:
            path = (
                checkpoint_dir
                / f"step_{step:07d}.pt"
            )

            save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                step,
                cfg,
                evaluation=latest_eval,
            )

            print(
                f"Saved: {path}"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    final_checkpoint = (
        checkpoint_dir
        / f"step_{max_steps:07d}.pt"
    )

    return {
        "steps": max_steps,
        "training_seconds": elapsed,
        "final_checkpoint":
            str(final_checkpoint),
        "final_evaluation":
            latest_eval,
        "parameter_report":
            model.parameter_report(),
    }