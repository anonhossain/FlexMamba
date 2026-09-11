import copy
from pathlib import Path

import torch

from src.models.markov_head_model import (
    FlexMambaMarkov,
    load_flexmamba_target,
)

from src.training.common_16m import (
    PROJECT_ROOT,
    build_validation_batches,
    choose_device,
    save_json,
)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def current_memory_mb(device):

    if device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 ** 2)

    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            return None

    return None


# ============================================================
# EVALUATION
# ============================================================

def evaluate_markov_head(summary):

    checkpoint_path = resolve_path(
        summary["final_checkpoint"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    parent_checkpoint = resolve_path(
        checkpoint["parent_checkpoint"]
    )

    run_dir = resolve_path(
        summary["run_dir"]
    )

    device = choose_device()

    print(f"\nEvaluation device: {device}")
    print(f"Markov checkpoint: {checkpoint_path}")

    target, parent_cfg = load_flexmamba_target(
        parent_checkpoint,
        device,
    )

    model = FlexMambaMarkov(
        target,
        cfg,
    ).to(device)

    model.load_adapter_state_dict(
        checkpoint["adapter_state_dict"]
    )

    model.eval()

    eval_cfg = cfg["evaluation"]
    smoke = cfg.get("smoke", {})

    smoke_enabled = smoke.get(
        "enabled",
        False,
    )

    eval_batches = (
        smoke.get("eval_batches", 1)
        if smoke_enabled
        else eval_cfg["eval_batches"]
    )

    prompt_length = (
        smoke.get("prompt_length", 32)
        if smoke_enabled
        else eval_cfg["generation"]["prompt_length"]
    )

    generation_length = (
        smoke.get("generation_length", 16)
        if smoke_enabled
        else eval_cfg["generation"]["generation_length"]
    )

    temperature = float(
        eval_cfg["generation"].get(
            "temperature",
            1.0,
        )
    )

    block_size = int(
        eval_cfg["generation"].get(
            "block_size",
            cfg["markov_head"]["block_size"],
        )
    )

    if prompt_length + generation_length > target.cfg.max_seq_len:

        raise ValueError(
            f"prompt_length + generation_length = "
            f"{prompt_length + generation_length}, "
            f"but target max_seq_len={target.cfg.max_seq_len}."
        )

    data_cfg = copy.deepcopy(cfg)

    data_cfg["validation"] = {
        "enabled": True,
        "eval_batches": eval_batches,
    }

    data_cfg["training"]["batch_size"] = 1

    batches = build_validation_batches(
        data_cfg,
        target.cfg,
    )

    baseline_tokens = 0
    baseline_seconds = 0.0

    speculative_tokens = 0
    speculative_seconds = 0.0

    rounds = 0
    accepted_draft = 0
    proposed_draft = 0

    draft_seconds = 0.0
    verification_seconds = 0.0

    greedy_matches = 0

    peak_memory = current_memory_mb(
        device
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for index, batch in enumerate(batches):

        prompt = batch[
            0:1,
            :prompt_length,
        ].to(device)

        # ----------------------------------------------------
        # STANDARD TARGET DECODING
        # ----------------------------------------------------

        baseline_generator = torch.Generator(
            device=device
        )

        baseline_generator.manual_seed(
            cfg["experiment"]["seed"] + index
        )

        baseline = model.baseline_generate(
            prompt,
            max_new_tokens=generation_length,
            temperature=temperature,
            generator=baseline_generator,
        )

        baseline_tokens += generation_length
        baseline_seconds += baseline["seconds"]

        # ----------------------------------------------------
        # LOSSLESS SPECULATIVE DECODING
        # ----------------------------------------------------

        speculative_generator = torch.Generator(
            device=device
        )

        speculative_generator.manual_seed(
            cfg["experiment"]["seed"] + 10000 + index
        )

        speculative = model.speculative_generate(
            prompt,
            max_new_tokens=generation_length,
            block_size=block_size,
            temperature=temperature,
            generator=speculative_generator,
        )

        speculative_tokens += generation_length
        speculative_seconds += speculative["seconds"]

        rounds += speculative["rounds"]
        accepted_draft += speculative["accepted_draft_tokens"]
        proposed_draft += speculative["proposed_draft_tokens"]

        draft_seconds += speculative["draft_seconds"]
        verification_seconds += speculative["verification_seconds"]

        # ----------------------------------------------------
        # DETERMINISTIC QUALITY-EQUIVALENCE SANITY CHECK
        # ----------------------------------------------------

        if eval_cfg.get(
            "greedy_equivalence_check",
            True,
        ):

            greedy_baseline = model.baseline_generate(
                prompt,
                max_new_tokens=generation_length,
                temperature=0.0,
            )

            greedy_speculative = model.speculative_generate(
                prompt,
                max_new_tokens=generation_length,
                block_size=block_size,
                temperature=0.0,
            )

            match = torch.equal(
                greedy_baseline["tokens"],
                greedy_speculative["tokens"],
            )

            greedy_matches += int(match)

        memory = current_memory_mb(device)

        if memory is not None:
            peak_memory = max(
                peak_memory or 0.0,
                memory,
            )

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    baseline_tps = (
        baseline_tokens
        / baseline_seconds
    )

    speculative_tps = (
        speculative_tokens
        / speculative_seconds
    )

    speedup = (
        speculative_tps
        / baseline_tps
    )

    acceptance_rate = (
        accepted_draft
        / proposed_draft
        if proposed_draft
        else 0.0
    )

    accepted_per_round = (
        accepted_draft
        / rounds
        if rounds
        else 0.0
    )

    # DSpark reports accepted length including the target-generated token.
    tau = (
        (accepted_draft + rounds)
        / rounds
        if rounds
        else 0.0
    )

    greedy_match_rate = (
        greedy_matches
        / len(batches)
        if batches
        else 0.0
    )

    report = {
        "identity": {
            "stage": "s12",
            "technique": "markov_head",
            "model_type": "flexmamba_markov",

            "checkpoint": str(checkpoint_path),
            "parent_checkpoint": str(parent_checkpoint),

            "device": str(device),
        },

        "architecture": model.architecture_report(),

        "parameters": model.parameter_report(),

        "baseline": {
            "generated_tokens": baseline_tokens,
            "seconds": baseline_seconds,
            "tokens_per_second": baseline_tps,
        },

        "speculative": {
            "generated_tokens": speculative_tokens,

            "seconds": speculative_seconds,
            "tokens_per_second": speculative_tps,

            "rounds": rounds,

            "proposed_draft_tokens": proposed_draft,
            "accepted_draft_tokens": accepted_draft,

            "draft_acceptance_rate": acceptance_rate,

            "accepted_tokens_per_draft": accepted_per_round,

            "accepted_length_tau": tau,

            "rejected_draft_tokens": (
                proposed_draft
                - accepted_draft
            ),

            "draft_seconds": draft_seconds,
            "verification_seconds": verification_seconds,

            "average_draft_latency_ms": (
                1000 * draft_seconds / rounds
                if rounds
                else 0.0
            ),

            "average_verification_latency_ms": (
                1000 * verification_seconds / rounds
                if rounds
                else 0.0
            ),
        },

        "speedup_ratio": speedup,

        "peak_memory_mb": peak_memory,

        "quality_equivalence": {
            "verification_algorithm": (
                "standard speculative rejection sampling"
            ),

            "target_distribution_preserved_by_algorithm": True,

            "greedy_sequence_match": (
                greedy_match_rate == 1.0
            ),

            "greedy_match_rate": (
                greedy_match_rate
            ),

            "note": (
                "Sampling-mode outputs are not expected to be "
                "token-identical because baseline and speculative "
                "decoding consume random draws differently. "
                "The rejection-sampling correction preserves the "
                "target distribution."
            ),
        },

        "evaluation_setup": {
            "eval_batches": len(batches),
            "prompt_length": prompt_length,
            "generation_length": generation_length,
            "temperature": temperature,
            "block_size": block_size,
        },

        "limitations": {
            "production_dspark_backbone": False,

            "target_feature_source": "final_hidden",

            "hardware_aware_scheduler": False,

            "incremental_target_cache": False,

            "wall_clock_speedup_is_prototype_only": True,

            "note": (
                "FlexMamba currently recomputes the target prefix to "
                "recover anchor features and does not use a production "
                "incremental serving engine. Acceptance metrics are "
                "meaningful; CPU wall-clock speedup is not directly "
                "comparable with DSpark production results."
            ),
        },
    }

    save_json(
        run_dir / "evaluation.json",
        report,
    )

    print("\nMarkov speculative evaluation completed.")

    print(
        f"Baseline tokens/sec: "
        f"{baseline_tps:.2f}"
    )

    print(
        f"Speculative tokens/sec: "
        f"{speculative_tps:.2f}"
    )

    print(
        f"Measured speedup: "
        f"{speedup:.3f}x"
    )

    print(
        f"Draft acceptance rate: "
        f"{acceptance_rate * 100:.2f}%"
    )

    print(
        f"Accepted draft tokens/round: "
        f"{accepted_per_round:.3f}"
    )

    print(
        f"Accepted length τ: "
        f"{tau:.3f}"
    )

    print(
        f"Rejected draft tokens: "
        f"{proposed_draft - accepted_draft}"
    )

    print(
        f"Greedy quality equivalence: "
        f"{greedy_match_rate == 1.0}"
    )

    print(
        f"Adapter parameters: "
        f"{report['parameters']['adapter_trainable_parameters']:,}"
    )

    print(
        f"Markov parameters: "
        f"{report['parameters']['markov_head_parameters']:,}"
    )

    print(
        f"Saved: "
        f"{run_dir / 'evaluation.json'}"
    )

    return report