from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.final_joint_model import (
    FinalJointModelConfig,
    FinalJointCausalLM,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MarkovHeadModelConfig:
    block_size: int = 7
    maximum_block_size: int = 7
    draft_layers: int = 5
    draft_heads: int = 4
    draft_ffn_multiplier: int = 2
    dropout: float = 0.0
    markov_rank: int = 32
    anchors_per_sequence: int = 8


# ============================================================
# PARALLEL DRAFT BACKBONE
# ============================================================

class ParallelDraftBackbone(nn.Module):

    def __init__(self, d_model, max_block_size, layers, heads, ffn_mult, dropout):
        super().__init__()

        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by draft_heads={heads}.")

        self.d_model = d_model
        self.max_block_size = max_block_size

        self.mask_embedding = nn.Parameter(torch.empty(d_model))
        self.position_embedding = nn.Parameter(torch.empty(max_block_size, d_model))

        self.context_proj = nn.Linear(d_model, d_model, bias=False)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * ffn_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        ])

        self.norm = nn.RMSNorm(d_model)

        nn.init.normal_(self.mask_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)


    def forward(self, anchor_embedding, target_hidden, block_size):

        if block_size > self.max_block_size:
            raise ValueError(
                f"block_size={block_size} exceeds maximum_block_size={self.max_block_size}."
            )

        batch_size = anchor_embedding.size(0)

        anchor = anchor_embedding.unsqueeze(1)

        if block_size > 1:
            masks = self.mask_embedding.view(1, 1, -1).expand(batch_size, block_size - 1, -1)
            hidden = torch.cat([anchor, masks], dim=1)
        else:
            hidden = anchor

        hidden = hidden + self.position_embedding[:block_size].unsqueeze(0)
        hidden = hidden + self.context_proj(target_hidden).unsqueeze(1)

        for layer in self.layers:
            hidden = layer(hidden)

        return self.norm(hidden)


# ============================================================
# PAPER MARKOV HEAD: B = W1 W2
# ============================================================

class VanillaMarkovHead(nn.Module):

    def __init__(self, vocab_size, rank):
        super().__init__()

        if rank <= 0:
            raise ValueError("markov_rank must be > 0.")

        self.vocab_size = vocab_size
        self.rank = rank

        self.w1 = nn.Embedding(vocab_size, rank)
        self.w2 = nn.Linear(rank, vocab_size, bias=False)


    def previous_embedding(self, token_ids):
        return self.w1(token_ids.long())


    def transition_bias(self, token_ids):
        return self.w2(self.previous_embedding(token_ids))


    def apply_teacher_forcing(self, base_logits, previous_tokens):
        return base_logits + self.transition_bias(previous_tokens)


# ============================================================
# CONFIDENCE HEAD
# ============================================================

class AcceptanceConfidenceHead(nn.Module):

    def __init__(self, d_model, markov_rank):
        super().__init__()
        self.proj = nn.Linear(d_model + markov_rank, 1)


    def forward(self, draft_hidden, previous_embedding):
        features = torch.cat([draft_hidden, previous_embedding], dim=-1)
        return self.proj(features).squeeze(-1)


# ============================================================
# FLEXMAMBA + MARKOV DRAFTER
# ============================================================

class FlexMambaMarkov(nn.Module):

    def __init__(self, target, cfg):
        super().__init__()

        self.target = target
        self.cfg = cfg

        markov = cfg["markov_head"]

        self.block_size = int(markov["block_size"])
        self.max_block_size = int(markov["maximum_block_size"])
        self.markov_rank = int(markov["markov_rank"])

        d_model = target.cfg.d_model
        vocab_size = target.cfg.vocab_size

        self.target_state_mode = markov.get("target_state_mode", "quantized")

        self.draft_backbone = ParallelDraftBackbone(
            d_model=d_model,
            max_block_size=self.max_block_size,
            layers=int(markov["draft_layers"]),
            heads=int(markov["draft_heads"]),
            ffn_mult=int(markov["draft_ffn_multiplier"]),
            dropout=float(markov.get("dropout", 0.0)),
        )

        self.markov_head = VanillaMarkovHead(vocab_size, self.markov_rank)
        self.confidence_head = AcceptanceConfidenceHead(d_model, self.markov_rank)

        self.loss_decay_gamma = float(markov.get("loss_decay_gamma", 4.0))
        self.ce_coeff = float(markov.get("ce_loss_coeff", 0.1))
        self.tv_coeff = float(markov.get("tv_loss_coeff", 0.9))
        self.conf_coeff = float(markov.get("confidence_loss_coeff", 1.0))

        for parameter in self.target.parameters():
            parameter.requires_grad = False

        self.target.eval()


    # ========================================================
    # TRAIN / FREEZE
    # ========================================================

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self


    def adapter_parameters(self):
        return [p for name, p in self.named_parameters()
                if not name.startswith("target.") and p.requires_grad]


    # ========================================================
    # TARGET FEATURES
    # ========================================================

    def _set_target_mode(self):
        if hasattr(self.target, "set_state_mode"):
            self.target.set_state_mode(self.target_state_mode)


    @torch.no_grad()
    def target_forward_with_hidden(self, input_ids):

        self._set_target_mode()

        holder = {}

        def capture_hidden(module, args, output):
            holder["hidden"] = output.detach()

        handle = self.target.model.backbone.norm_f.register_forward_hook(capture_hidden)

        try:
            output = self.target(input_ids)
        finally:
            handle.remove()

        if "hidden" not in holder:
            raise RuntimeError("Failed to capture FlexMamba final hidden states.")

        return output["logits"].detach(), holder["hidden"]


    @torch.no_grad()
    def target_logits(self, input_ids):
        self._set_target_mode()
        return self.target(input_ids)["logits"].detach()


    # ========================================================
    # SHARED FROZEN EMBEDDING / LM HEAD
    # ========================================================

    def _anchor_embedding(self, anchor_ids):
        with torch.no_grad():
            return self.target.model.backbone.embeddings(anchor_ids).detach()


    def _shared_lm_head(self, hidden):
        # Target LM-head weights remain frozen, but gradients still flow to hidden.
        return self.target.model.lm_head(hidden)


    # ========================================================
    # PARALLEL BASE DRAFT
    # ========================================================

    def parallel_draft(self, anchor_ids, anchor_hidden, block_size):

        anchor_embedding = self._anchor_embedding(anchor_ids)

        draft_hidden = self.draft_backbone(
            anchor_embedding,
            anchor_hidden,
            block_size,
        )

        base_logits = self._shared_lm_head(draft_hidden)

        return draft_hidden, base_logits


    # ========================================================
    # TRAINING BLOCKS
    # ========================================================

    def _sample_training_blocks(self, input_ids, target_hidden, target_logits, anchors):

        batch_size, seq_len = input_ids.shape
        gamma = self.block_size

        max_anchor = seq_len - gamma - 1

        if max_anchor < 0:
            raise ValueError(
                f"Sequence length {seq_len} is too short for block_size={gamma}."
            )

        anchor_positions = torch.randint(
            0,
            max_anchor + 1,
            (batch_size, anchors),
            device=input_ids.device,
        )

        batch_index = torch.arange(batch_size, device=input_ids.device).view(-1, 1)
        batch_index = batch_index.expand(batch_size, anchors)

        anchor_ids = input_ids[batch_index, anchor_positions]
        anchor_hidden = target_hidden[batch_index, anchor_positions]

        offsets = torch.arange(gamma, device=input_ids.device).view(1, 1, -1)

        token_positions = anchor_positions.unsqueeze(-1) + offsets
        batch_3d = batch_index.unsqueeze(-1).expand(-1, -1, gamma)

        target_ids = input_ids[batch_3d, token_positions + 1]
        teacher_logits = target_logits[batch_3d, token_positions]

        return anchor_ids, anchor_hidden, target_ids, teacher_logits


    # ========================================================
    # DSPARK LOSS
    # ========================================================

    def training_forward(self, input_ids, anchors_per_sequence=None):

        anchors = anchors_per_sequence or int(
            self.cfg["markov_head"]["anchors_per_sequence"]
        )

        target_logits, target_hidden = self.target_forward_with_hidden(input_ids)

        anchor_ids, anchor_hidden, target_ids, teacher_logits = (
            self._sample_training_blocks(
                input_ids,
                target_hidden,
                target_logits,
                anchors,
            )
        )

        batch_size, num_anchors = anchor_ids.shape
        gamma = self.block_size

        flat_anchor_ids = anchor_ids.reshape(-1)
        flat_anchor_hidden = anchor_hidden.reshape(-1, anchor_hidden.size(-1))

        draft_hidden, base_logits = self.parallel_draft(
            flat_anchor_ids,
            flat_anchor_hidden,
            gamma,
        )

        draft_hidden = draft_hidden.reshape(
            batch_size, num_anchors, gamma, -1
        )

        base_logits = base_logits.reshape(
            batch_size, num_anchors, gamma, -1
        )

        previous_tokens = torch.cat(
            [anchor_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )

        previous_embeddings = self.markov_head.previous_embedding(previous_tokens)

        draft_logits = self.markov_head.apply_teacher_forcing(
            base_logits,
            previous_tokens,
        )

        confidence_logits = self.confidence_head(
            draft_hidden,
            previous_embeddings,
        )

        positions = torch.arange(
            gamma,
            device=input_ids.device,
            dtype=torch.float32,
        )

        weights = torch.exp(
            -positions / self.loss_decay_gamma
        ).view(1, 1, gamma)

        weight_den = weights.expand(
            batch_size, num_anchors, -1
        ).sum().clamp_min(1e-6)

        ce = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.size(-1)),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(batch_size, num_anchors, gamma)

        draft_prob = torch.softmax(draft_logits.float(), dim=-1)
        target_prob = torch.softmax(teacher_logits.float(), dim=-1)

        tv_l1 = (draft_prob - target_prob).abs().sum(dim=-1)

        # Analytical speculative acceptance probability:
        # c* = 1 - 0.5 ||p_d - p_t||_1
        acceptance_target = (
            1.0 - 0.5 * tv_l1
        ).clamp(0.0, 1.0).detach()

        confidence_bce = F.binary_cross_entropy_with_logits(
            confidence_logits.float(),
            acceptance_target,
            reduction="none",
        )

        ce_loss = (ce * weights).sum() / weight_den
        tv_loss = (tv_l1 * weights).sum() / weight_den
        confidence_loss = (confidence_bce * weights).sum() / weight_den

        loss = (
            self.ce_coeff * ce_loss
            + self.tv_coeff * tv_loss
            + self.conf_coeff * confidence_loss
        )

        confidence_prob = torch.sigmoid(confidence_logits.float())

        confidence_mae = (
            (confidence_prob - acceptance_target).abs() * weights
        ).sum() / weight_den

        tau_probabilistic = (
            1.0
            + acceptance_target.cumprod(dim=-1).sum(dim=-1)
        ).mean()

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "tv_loss": tv_loss,
            "confidence_loss": confidence_loss,
            "confidence_mae": confidence_mae,
            "tau_probabilistic": tau_probabilistic,
            "mean_acceptance_target": acceptance_target.mean(),
        }


    # ========================================================
    # DRAFT GENERATION
    # ========================================================

    @staticmethod
    def _distribution(logits, temperature):

        if temperature <= 0:
            return torch.softmax(logits.float(), dim=-1)

        return torch.softmax(
            logits.float() / temperature,
            dim=-1,
        )


    @staticmethod
    def _sample(probs, temperature, generator=None):

        if temperature <= 0:
            return probs.argmax(dim=-1)

        return torch.multinomial(
            probs,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)


    @torch.inference_mode()
    def draft_block(self, prefix, block_size=None, temperature=1.0, generator=None):

        if prefix.size(0) != 1:
            raise ValueError("Prototype speculative generation currently requires batch_size=1.")

        gamma = block_size or self.block_size

        if gamma > self.max_block_size:
            raise ValueError("Requested draft block exceeds maximum_block_size.")

        _, target_hidden = self.target_forward_with_hidden(prefix)

        anchor_id = prefix[:, -1]
        anchor_hidden = target_hidden[:, -1]

        draft_hidden, base_logits = self.parallel_draft(
            anchor_id,
            anchor_hidden,
            gamma,
        )

        previous = anchor_id
        sampled = []
        distributions = []
        confidences = []

        for position in range(gamma):

            previous_embedding = self.markov_head.previous_embedding(previous)

            logits = (
                base_logits[:, position]
                + self.markov_head.w2(previous_embedding)
            )

            probs = self._distribution(logits, temperature)
            token = self._sample(probs, temperature, generator)

            confidence_logit = self.confidence_head(
                draft_hidden[:, position],
                previous_embedding,
            )

            sampled.append(token)
            distributions.append(probs)
            confidences.append(torch.sigmoid(confidence_logit))

            previous = token

        return {
            "tokens": torch.stack(sampled, dim=1),
            "probabilities": torch.stack(distributions, dim=1),
            "confidences": torch.stack(confidences, dim=1),
        }


    # ========================================================
    # EXACT / LOSSLESS VERIFICATION
    # ========================================================

    @torch.inference_mode()
    def verify_draft(self, prefix, draft, temperature=1.0, generator=None):

        draft_tokens = draft["tokens"]
        draft_probs = draft["probabilities"]

        gamma = draft_tokens.size(1)
        prompt_len = prefix.size(1)

        verification_input = torch.cat(
            [prefix, draft_tokens],
            dim=1,
        )

        logits = self.target_logits(verification_input)

        target_logits = logits[:, prompt_len - 1:prompt_len - 1 + gamma]
        bonus_logits = logits[:, prompt_len - 1 + gamma]

        target_probs = self._distribution(target_logits, temperature)
        bonus_probs = self._distribution(bonus_logits, temperature)

        output_tokens = []
        accepted = 0
        rejection_position = None

        # Deterministic greedy equivalence path.
        if temperature <= 0:

            for k in range(gamma):

                target_token = target_probs[:, k].argmax(dim=-1)
                proposed = draft_tokens[:, k]

                if torch.equal(proposed, target_token):
                    output_tokens.append(proposed)
                    accepted += 1
                    continue

                output_tokens.append(target_token)
                rejection_position = k
                break

            if rejection_position is None:
                output_tokens.append(bonus_probs.argmax(dim=-1))

        # Standard speculative rejection sampling.
        else:

            for k in range(gamma):

                proposed = draft_tokens[0, k]

                p = target_probs[0, k]
                q = draft_probs[0, k]

                p_token = p[proposed]
                q_token = q[proposed].clamp_min(1e-12)

                accept_probability = torch.minimum(
                    torch.ones_like(p_token),
                    p_token / q_token,
                )

                uniform = torch.rand(
                    (),
                    device=p.device,
                    generator=generator,
                )

                if uniform <= accept_probability:
                    output_tokens.append(proposed.view(1))
                    accepted += 1
                    continue

                residual = (p - q).clamp_min(0.0)

                if residual.sum() <= 1e-12:
                    residual = p
                else:
                    residual = residual / residual.sum()

                replacement = torch.multinomial(
                    residual,
                    num_samples=1,
                    generator=generator,
                )

                output_tokens.append(replacement)
                rejection_position = k
                break

            if rejection_position is None:

                bonus = torch.multinomial(
                    bonus_probs[0],
                    num_samples=1,
                    generator=generator,
                )

                output_tokens.append(bonus)

        tokens = torch.cat(output_tokens, dim=0)

        return {
            "tokens": tokens,
            "accepted_draft_tokens": accepted,
            "proposed_draft_tokens": gamma,
            "rejection_position": rejection_position,
            "all_draft_tokens_accepted": rejection_position is None,
        }


    # ========================================================
    # GENERATION
    # ========================================================

    @staticmethod
    def _sync(device):

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()


    @torch.inference_mode()
    def baseline_generate(self, prompt, max_new_tokens, temperature=1.0, generator=None):

        sequence = prompt.clone()
        device = sequence.device

        self._sync(device)
        start = time.perf_counter()

        for _ in range(max_new_tokens):

            logits = self.target_logits(sequence)[:, -1]
            probs = self._distribution(logits, temperature)
            token = self._sample(probs, temperature, generator)

            sequence = torch.cat(
                [sequence, token.view(1, 1)],
                dim=1,
            )

        self._sync(device)

        elapsed = time.perf_counter() - start

        return {
            "tokens": sequence[:, prompt.size(1):],
            "seconds": elapsed,
            "tokens_per_second": max_new_tokens / elapsed,
            "target_forward_calls": max_new_tokens,
        }


    @torch.inference_mode()
    def speculative_generate(
        self,
        prompt,
        max_new_tokens,
        block_size=None,
        temperature=1.0,
        generator=None,
    ):

        sequence = prompt.clone()
        device = sequence.device

        generated = 0
        rounds = 0

        accepted_total = 0
        proposed_total = 0

        draft_seconds = 0.0
        verification_seconds = 0.0

        while generated < max_new_tokens:

            remaining = max_new_tokens - generated
            gamma = min(block_size or self.block_size, remaining)

            self._sync(device)
            start = time.perf_counter()

            draft = self.draft_block(
                sequence,
                block_size=gamma,
                temperature=temperature,
                generator=generator,
            )

            self._sync(device)
            draft_seconds += time.perf_counter() - start

            self._sync(device)
            start = time.perf_counter()

            verified = self.verify_draft(
                sequence,
                draft,
                temperature=temperature,
                generator=generator,
            )

            self._sync(device)
            verification_seconds += time.perf_counter() - start

            new_tokens = verified["tokens"][:remaining]

            sequence = torch.cat(
                [sequence, new_tokens.view(1, -1)],
                dim=1,
            )

            generated += new_tokens.numel()
            rounds += 1

            accepted_total += verified["accepted_draft_tokens"]
            proposed_total += verified["proposed_draft_tokens"]

        total_seconds = draft_seconds + verification_seconds

        return {
            "tokens": sequence[:, prompt.size(1):],

            "seconds": total_seconds,
            "tokens_per_second": max_new_tokens / total_seconds,

            "rounds": rounds,

            "accepted_draft_tokens": accepted_total,
            "proposed_draft_tokens": proposed_total,

            "draft_acceptance_rate": (
                accepted_total / proposed_total
                if proposed_total
                else 0.0
            ),

            # Paper-style accepted length includes one target token per round.
            "accepted_length_tau": (
                (accepted_total + rounds) / rounds
                if rounds
                else 0.0
            ),

            "accepted_draft_tokens_per_round": (
                accepted_total / rounds
                if rounds
                else 0.0
            ),

            "rejected_draft_tokens": (
                proposed_total - accepted_total
            ),

            "draft_seconds": draft_seconds,
            "verification_seconds": verification_seconds,

            # This prototype needs one target pass to recover
            # anchor features plus one verification pass per round.
            "target_forward_calls": 2 * rounds,
        }


    # ========================================================
    # CHECKPOINT / REPORT
    # ========================================================

    def adapter_state_dict(self):

        return {
            "draft_backbone": self.draft_backbone.state_dict(),
            "markov_head": self.markov_head.state_dict(),
            "confidence_head": self.confidence_head.state_dict(),
        }


    def load_adapter_state_dict(self, state):

        self.draft_backbone.load_state_dict(state["draft_backbone"], strict=True)
        self.markov_head.load_state_dict(state["markov_head"], strict=True)
        self.confidence_head.load_state_dict(state["confidence_head"], strict=True)


    def parameter_report(self):

        target_parameters = sum(p.numel() for p in self.target.parameters())
        adapter_parameters = sum(p.numel() for p in self.adapter_parameters())

        markov_parameters = sum(p.numel() for p in self.markov_head.parameters())
        draft_parameters = sum(p.numel() for p in self.draft_backbone.parameters())
        confidence_parameters = sum(p.numel() for p in self.confidence_head.parameters())

        return {
            "target_parameters": target_parameters,
            "target_trainable_parameters": 0,

            "adapter_trainable_parameters": adapter_parameters,

            "parallel_drafter_parameters": draft_parameters,
            "markov_head_parameters": markov_parameters,
            "confidence_head_parameters": confidence_parameters,

            "combined_parameters": target_parameters + adapter_parameters,
            "adapter_fraction_of_target": adapter_parameters / target_parameters,

            "markov_rank": self.markov_rank,
        }


    def architecture_report(self):

        return {
            "architecture": "flexmamba_markov_speculator",

            "target_architecture": "flexmamba",
            "target_frozen": True,

            "parallel_drafter": True,
            "semi_autoregressive": True,

            "markov_head": True,
            "markov_type": "first_order_low_rank",
            "markov_rank": self.markov_rank,

            "confidence_head": True,

            "block_size": self.block_size,

            "verification": "standard_speculative_rejection_sampling",
            "lossless_verification": True,

            "target_feature_source": "final_hidden",

            "dspark_exact_backbone_reproduction": False,
            "dspark_inspired_flexmamba_adaptation": True,
        }


# ============================================================
# TARGET LOADER
# ============================================================

def load_flexmamba_target(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = checkpoint["config"]

    model_cfg = FinalJointModelConfig(
        **cfg["model"]
    )

    target = FinalJointCausalLM(
        model_cfg,
        cfg["recursion"],
        cfg["state"],
        cfg["routing"],
        cfg["grouped_parameterization"],
        cfg["lte"],
        cfg["state_quantization"],
        cfg["quantization"],
        seed=cfg["experiment"]["seed"],
    )

    target.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    target = target.to(device)
    target.eval()

    for parameter in target.parameters():
        parameter.requires_grad = False

    return target, cfg