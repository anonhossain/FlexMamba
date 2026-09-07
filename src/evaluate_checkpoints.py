import argparse
import json
import platform
import time
from pathlib import Path

import torch

from models.mor_small import MoRConfig, MoRCausalLM
from models.mamba_small import MambaSmallConfig, MambaSmallCausalLM
from models.recursive_mamba import (
    RecursiveMambaConfig,
    RecursiveMambaCausalLM,
)

# ============================================================
# DEVICE
# ============================================================

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):
    """
    Make sure device operations have actually finished before
    reading the timer.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# MODEL LOADING
# ============================================================

def detect_model_type(cfg, checkpoint_path):
    """
    First try to read model_type from YAML/checkpoint config.

    If older checkpoints don't contain model_type,
    infer it from the directory/file name.
    """

    if "model_type" in cfg:
        return cfg["model_type"].lower()

    path = str(checkpoint_path).lower()

    if "mamba" in path and "mor" not in path:
        return "mamba"

    if "mor" in path:
        return "mor"

    raise ValueError(
        f"Could not determine model type for {checkpoint_path}. "
        "Add model_type to the YAML config."
    )


def load_model(checkpoint_path, device):
    print(f"\nLoading: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    model_type = detect_model_type(
        cfg,
        checkpoint_path,
    )

    if model_type == "mor":

        model_cfg = MoRConfig(**cfg["model"])
        model = MoRCausalLM(model_cfg)

    elif model_type == "mamba":

        model_cfg = MambaSmallConfig(**cfg["model"])
        model = MambaSmallCausalLM(model_cfg)
    
    elif model_type == "recursive_mamba":
        model_cfg = RecursiveMambaConfig(**cfg["model"])
        model = RecursiveMambaCausalLM(model_cfg)

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    # Support slightly different checkpoint naming conventions
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]

    else:
        raise KeyError(
            f"No model state found inside {checkpoint_path}"
        )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model, cfg, model_type


# ============================================================
# MODEL OUTPUT
# ============================================================

def get_logits(model, input_ids):

    output = model(input_ids)

    if isinstance(output, dict):
        return output["logits"]

    if hasattr(output, "logits"):
        return output.logits

    raise ValueError(
        "Model output does not contain logits."
    )


# ============================================================
# PREFILL / FORWARD BENCHMARK
# ============================================================

@torch.inference_mode()
def benchmark_prefill(
    model,
    device,
    vocab_size,
    seq_len=128,
    batch_size=1,
    warmup_runs=2,
    measured_runs=5,
):

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )

    # -----------------------------
    # Warmup
    # -----------------------------

    for _ in range(warmup_runs):
        _ = get_logits(
            model,
            input_ids,
        )

    synchronize(device)

    # -----------------------------
    # Actual benchmark
    # -----------------------------

    times = []

    for _ in range(measured_runs):

        synchronize(device)

        start = time.perf_counter()

        _ = get_logits(
            model,
            input_ids,
        )

        synchronize(device)

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

    mean_time = sum(times) / len(times)

    tokens_processed = (
        batch_size
        * seq_len
    )

    tokens_per_second = (
        tokens_processed
        / mean_time
    )

    milliseconds_per_forward = (
        mean_time * 1000
    )

    return {
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "mean_forward_time_ms":
            milliseconds_per_forward,
        "tokens_per_second":
            tokens_per_second,
    }


# ============================================================
# TIME TO FIRST TOKEN
# ============================================================

@torch.inference_mode()
def benchmark_ttft(
    model,
    device,
    vocab_size,
    prompt_length=128,
    warmup_runs=2,
    measured_runs=5,
):

    input_ids = torch.randint(
        0,
        vocab_size,
        (1, prompt_length),
        device=device,
    )

    # Warmup

    for _ in range(warmup_runs):

        logits = get_logits(
            model,
            input_ids,
        )

        _ = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        )

    synchronize(device)

    times = []

    for _ in range(measured_runs):

        synchronize(device)

        start = time.perf_counter()

        logits = get_logits(
            model,
            input_ids,
        )

        _ = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        )

        synchronize(device)

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

    mean_time = sum(times) / len(times)

    return mean_time * 1000


# ============================================================
# AUTOREGRESSIVE DECODING
# ============================================================

@torch.inference_mode()
def benchmark_decode(
    model,
    device,
    vocab_size,
    prompt_length=64,
    generated_tokens=16,
    warmup_runs=1,
    measured_runs=3,
):

    base_prompt = torch.randint(
        0,
        vocab_size,
        (1, prompt_length),
        device=device,
    )

    def generate():

        input_ids = base_prompt.clone()

        for _ in range(generated_tokens):

            logits = get_logits(
                model,
                input_ids,
            )

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            )

        return input_ids

    # -----------------------------
    # Warmup
    # -----------------------------

    for _ in range(warmup_runs):
        generate()

    synchronize(device)

    # -----------------------------
    # Benchmark
    # -----------------------------

    times = []

    for _ in range(measured_runs):

        synchronize(device)

        start = time.perf_counter()

        generate()

        synchronize(device)

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

    mean_time = (
        sum(times)
        / len(times)
    )

    tokens_per_second = (
        generated_tokens
        / mean_time
    )

    ms_per_token = (
        mean_time
        / generated_tokens
        * 1000
    )

    return {
        "prompt_length": prompt_length,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "ms_per_token": ms_per_token,
        "total_generation_time_seconds":
            mean_time,
        "decode_mode":
            "naive_full_recompute",
    }


# ============================================================
# MEMORY
# ============================================================

def get_memory_mb(device):

    if device.type == "cuda":

        return (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    if device.type == "mps":

        try:
            return (
                torch.mps.current_allocated_memory()
                / (1024 ** 2)
            )
        except Exception:
            return None

    return None


# ============================================================
# JSON UPDATE
# ============================================================

def update_json(
    checkpoint_path,
    model,
    model_type,
    inference_results,
    device,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    json_path = (
        checkpoint_path
        .with_suffix(".json")
    )

    # Keep existing JSON if present
    if json_path.exists():

        with open(
            json_path,
            "r",
        ) as f:

            data = json.load(f)

    else:

        data = {
            "checkpoint":
                str(checkpoint_path)
        }

    # Add parameter information if needed
    if hasattr(
        model,
        "parameter_report",
    ):

        data["parameter_report"] = (
            model.parameter_report()
        )

    data["model_type"] = (
        model_type
    )

    data["inference"] = (
        inference_results
    )

    data["benchmark_environment"] = {
        "device":
            str(device),

        "torch_version":
            torch.__version__,

        "python_platform":
            platform.platform(),
    }

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

    print(
        f"Updated JSON: {json_path}"
    )


# ============================================================
# EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    checkpoint_path,
    device,
    args,
):

    model, cfg, model_type = (
        load_model(
            checkpoint_path,
            device,
        )
    )

    vocab_size = (
        cfg["model"]["vocab_size"]
    )

    # Reset CUDA peak memory measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------
    # Forward/prefill
    # --------------------------------

    prefill = benchmark_prefill(
        model=model,
        device=device,
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
    )

    # --------------------------------
    # TTFT
    # --------------------------------

    ttft = benchmark_ttft(
        model=model,
        device=device,
        vocab_size=vocab_size,
        prompt_length=args.prompt_length,
        warmup_runs=args.warmup_runs,
        measured_runs=args.runs,
    )

    # --------------------------------
    # Decode
    # --------------------------------

    decode = benchmark_decode(
        model=model,
        device=device,
        vocab_size=vocab_size,
        prompt_length=args.prompt_length,
        generated_tokens=args.generated_tokens,
        warmup_runs=1,
        measured_runs=max(
            1,
            args.runs // 2,
        ),
    )

    memory_mb = get_memory_mb(
        device
    )

    checkpoint_size_mb = (
        Path(checkpoint_path)
        .stat()
        .st_size
        / (1024 ** 2)
    )

    results = {

        "prefill": prefill,

        "decode": decode,

        "time_to_first_token_ms":
            ttft,

        "memory_mb":
            memory_mb,

        "checkpoint_size_mb":
            checkpoint_size_mb,
    }

    print("\nResults")

    print(
        f"Prefill tok/s: "
        f"{prefill['tokens_per_second']:.2f}"
    )

    print(
        f"Decode tok/s: "
        f"{decode['tokens_per_second']:.2f}"
    )

    print(
        f"Decode ms/token: "
        f"{decode['ms_per_token']:.2f}"
    )

    print(
        f"TTFT: "
        f"{ttft:.2f} ms"
    )

    if memory_mb is not None:

        print(
            f"Memory: "
            f"{memory_mb:.2f} MB"
        )

    print(
        f"Checkpoint size: "
        f"{checkpoint_size_mb:.2f} MB"
    )

    update_json(
        checkpoint_path,
        model,
        model_type,
        results,
        device,
    )

    # Free memory before next checkpoint
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        torch.mps.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir",
        required=True,
        help="Checkpoint directory",
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--prompt-length",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--generated-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    checkpoint_dir = Path(
        args.dir
    )

    if not checkpoint_dir.exists():

        raise FileNotFoundError(
            checkpoint_dir
        )

    device = choose_device()

    print(
        f"Benchmark device: {device}"
    )

    # -----------------------------------
    # step checkpoints
    # -----------------------------------

    checkpoints = sorted(
        checkpoint_dir.glob(
            "step_*.pt"
        )
    )

    # -----------------------------------
    # Add final.pt
    # -----------------------------------

    final_checkpoint = (
        checkpoint_dir
        / "final.pt"
    )

    if final_checkpoint.exists():
        checkpoints.append(
            final_checkpoint
        )

    if not checkpoints:

        raise RuntimeError(
            f"No checkpoints found in "
            f"{checkpoint_dir}"
        )

    print(
        f"Found {len(checkpoints)} checkpoints"
    )

    for i, checkpoint in enumerate(
        checkpoints,
        start=1,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"[{i}/{len(checkpoints)}] "
            f"{checkpoint.name}"
        )

        print(
            "=" * 70
        )

        try:

            evaluate_checkpoint(
                checkpoint,
                device,
                args,
            )

        except Exception as e:

            print(
                f"FAILED: "
                f"{checkpoint.name}"
            )

            print(
                f"Reason: {e}"
            )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()