import argparse
import math
import os
import random
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data.fineweb import PackedFineWebDataset
from models.mor_small import MoRConfig, MoRCausalLM
from models.mamba_small import MambaSmallConfig, MambaSmallCausalLM
from utils.checkpoint import save_checkpoint
from models.recursive_mamba import (
    RecursiveMambaConfig,
    RecursiveMambaCausalLM,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def make_scheduler(optimizer, max_steps, warmup_ratio, min_lr_ratio):
    warmup_steps = max(1, int(max_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    model.eval()
    losses = []
    depths = []
    it = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(it).to(device)
        except StopIteration:
            break
        x, y = batch[:, :-1], batch[:, 1:]
        out = model(x, labels=y)
        losses.append(out["lm_loss"].float().cpu())
        depths.append(out["avg_depth"].float().cpu())
    model.train()
    if not losses:
        return float("nan"), float("nan"), float("nan")
    nll = torch.stack(losses).mean().item()
    ppl = math.exp(min(nll, 20.0))
    avg_depth = torch.stack(depths).mean().item()
    return nll, ppl, avg_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mor_16m.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps for a quick smoke test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = choose_device()
    print(f"Device: {device}")

    if cfg["model_type"] == "mor":

        model_cfg = MoRConfig(**cfg["model"])
        model = MoRCausalLM(model_cfg).to(device)

    elif cfg["model_type"] == "mamba":

        model_cfg = MambaSmallConfig(**cfg["model"])
        model = MambaSmallCausalLM(model_cfg).to(device)
        
    elif cfg["model_type"]== "recursive_mamba":
        model_cfg= RecursiveMambaConfig(**cfg["model"])
        model= RecursiveMambaCausalLM(model_cfg).to(device)

    else:
        raise ValueError(
            f"Unknown model_type: {cfg['model_type']}"
        )
    report = model.parameter_report()
    print("Parameter report:")
    for k, v in report.items():
        print(f"  {k}: {v:,} ({v / 1e6:.3f}M)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"])
    assert len(tokenizer) == model_cfg.vocab_size, (
        f"Tokenizer vocab ({len(tokenizer)}) != model vocab ({model_cfg.vocab_size})"
    )

    train_ds = PackedFineWebDataset(
        tokenizer=tokenizer,
        dataset_name=cfg["data"]["dataset_name"],
        dataset_config=cfg["data"]["dataset_config"],
        seq_len=model_cfg.max_seq_len,
        mode="train",
        validation_docs=cfg["data"]["validation_docs"],
        shuffle_buffer=cfg["data"]["shuffle_buffer"],
        seed=cfg["seed"],
    )
    val_ds = PackedFineWebDataset(
        tokenizer=tokenizer,
        dataset_name=cfg["data"]["dataset_name"],
        dataset_config=cfg["data"]["dataset_config"],
        seq_len=model_cfg.max_seq_len,
        mode="validation",
        validation_docs=cfg["data"]["validation_docs"],
        shuffle_buffer=0,
        seed=cfg["seed"],
    )

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    train_cfg = cfg["training"]
    max_steps = args.steps or train_cfg["max_steps"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    min_lr_ratio = train_cfg["min_learning_rate"] / train_cfg["learning_rate"]
    scheduler = make_scheduler(
        optimizer,
        max_steps=max_steps,
        warmup_ratio=train_cfg["warmup_ratio"],
        min_lr_ratio=min_lr_ratio,
    )

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    train_iter = iter(train_loader)
    grad_accum = train_cfg["grad_accum_steps"]
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(1, max_steps + 1), desc="training")
    latest_eval = None
    for step in pbar:
        loss_sum = 0.0
        lm_sum = 0.0
        depth_sum = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            out = model(x, labels=y)
            loss = out["loss"] / grad_accum
            loss.backward()

            loss_sum += out["loss"].detach().float().item() / grad_accum
            lm_sum += out["lm_loss"].float().item() / grad_accum
            depth_sum += out["avg_depth"].float().item() / grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_cfg["log_every"] == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_sum:.3f}", lm=f"{lm_sum:.3f}", depth=f"{depth_sum:.2f}", lr=f"{lr:.2e}")

        if step % train_cfg["eval_every"] == 0:
            val_nll, val_ppl, val_depth = evaluate(
                model,
                val_loader,
                device,
                max_batches=train_cfg["eval_batches"]
            )

            latest_eval = {
                "step": step,
                "nll": val_nll,
                "ppl": val_ppl,
                "avg_depth": val_depth
            }

            print(
                f"\n[eval step {step}] "
                f"NLL={val_nll:.4f} "
                f"PPL={val_ppl:.2f} "
                f"avg_depth={val_depth:.3f}"
            )

        if step % train_cfg["save_every"] == 0:

            # Evaluate exactly at this checkpoint if not already evaluated
            if latest_eval is None or latest_eval["step"] != step:
                val_nll, val_ppl, val_depth = evaluate(
                    model,
                    val_loader,
                    device,
                    max_batches=train_cfg["eval_batches"]
                )

                latest_eval = {
                    "step": step,
                    "nll": val_nll,
                    "ppl": val_ppl,
                    "avg_depth": val_depth
                }

            ckpt = output_dir / f"step_{step:07d}.pt"

            save_checkpoint(
                str(ckpt),
                model,
                optimizer,
                scheduler,
                step,
                cfg,
                evaluation=latest_eval
            )

            save_checkpoint(
                str(output_dir / "last.pt"),
                model,
                optimizer,
                scheduler,
                step,
                cfg,
                evaluation=latest_eval
            )

            print(f"\nSaved checkpoint: {ckpt}")

    final_path = output_dir / "final.pt"

    val_nll, val_ppl, val_depth = evaluate(
        model,
        val_loader,
        device,
        max_batches=train_cfg["eval_batches"]
    )

    final_eval = {
        "step": max_steps,
        "nll": val_nll,
        "ppl": val_ppl,
        "avg_depth": val_depth
    }

    save_checkpoint(
        str(final_path),
        model,
        optimizer,
        scheduler,
        max_steps,
        cfg,
        evaluation=final_eval
    )


if __name__ == "__main__":
    main()
