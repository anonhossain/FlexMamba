# 16M MoR Local Prototype

A small, Mac-friendly **Mixture-of-Recursions (MoR)** language-model prototype designed to reproduce the *architecture ideas* of the MoR paper at a much smaller scale before moving to the 360M experiment.

## What is paper-aligned

- Llama-style causal Transformer blocks
- RMSNorm, RoPE, GQA, SwiGLU
- **Middle-Cycle** sharing: unique first/last blocks, shared middle pool
- **3 recursions**
- **Token-choice routing**: one routing decision assigns each token a recursion depth
- Token-choice balancing loss
- Router z-loss
- FineWeb-Edu streaming
- Cosine schedule for non-isoFLOP experimentation
- Tied token embedding / LM head

This is intentionally a **prototype**, not a byte-for-byte reproduction of the authors' optimized code. The official MoR implementation indexes only selected tokens and scatters them back for real sparse compute. This local version uses masking so it is easier to debug on Apple Silicon; therefore it validates routing/model behavior but is not intended to demonstrate the final throughput gain.

## Parameter target

The config uses:

- vocab: 49,152
- d_model: 256
- heads: 4
- KV heads: 2
- FFN: 768
- shared middle layers: 3
- recursions: 3

Run:

```bash
python count_params.py
```

The model should be around **16M unique parameters**.

## Setup on Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test first

```bash
python train.py --steps 5
```

This verifies dataset loading, forward/backward, router, optimizer, and checkpoint logic.

## Prototype training

```bash
python train.py --config configs/mor_16m.yaml
```

Checkpoints are written to:

```text
checkpoints/mor_16m/
```

including `last.pt` and `final.pt`.

## Important for the research roadmap

Treat this as **Phase 0: small MoR baseline**. Do not add Mamba/LTE/grouped layers/TurboQuant yet. First make this training and evaluation pipeline reproducible. Then create a new model file for each technique so the baseline remains untouched.
