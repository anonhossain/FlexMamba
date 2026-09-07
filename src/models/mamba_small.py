from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MambaConfig, MambaForCausalLM


@dataclass
class MambaSmallConfig:
    vocab_size: int = 49152
    d_model: int = 256
    n_layers: int = 8

    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    max_seq_len: int = 256

    tie_word_embeddings: bool = True
    use_bias: bool = False
    use_conv_bias: bool = True

    residual_in_fp32: bool = True


class MambaSmallCausalLM(nn.Module):
    """
    Small standard Mamba language model.

    No recursion.
    No MoR router.
    No LTE.
    No grouped parameter reduction.

    This is the clean Mamba baseline.
    """

    def __init__(self, config: MambaSmallConfig):
        super().__init__()

        self.cfg = config

        hf_config = MambaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            state_size=config.d_state,
            num_hidden_layers=config.n_layers,

            expand=config.expand,
            conv_kernel=config.d_conv,

            use_bias=config.use_bias,
            use_conv_bias=config.use_conv_bias,

            hidden_act="silu",

            residual_in_fp32=config.residual_in_fp32,

            # Important because otherwise the LM head adds
            # another ~12.6M parameters.
            tie_word_embeddings=config.tie_word_embeddings,

            # We do not need inference cache during pretraining.
            use_cache=False,

            # Safe fallback for local Mac/MPS development.
            use_mambapy=False,
            use_associative_scan=False,
        )

        self.model = MambaForCausalLM(hf_config)

    def forward(self, input_ids, labels=None):
        """
        train.py already prepares:

            x = tokens[:, :-1]
            y = tokens[:, 1:]

        Therefore we calculate CE ourselves instead of giving
        labels directly to Hugging Face, avoiding an additional
        internal causal shift.
        """

        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits

        lm_loss = None

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        # Standard Mamba has no dynamic recursion router.
        # Keep this value only so the current train.py interface
        # remains compatible.
        avg_depth = torch.tensor(
            float(self.cfg.n_layers),
            device=logits.device,
        )

        return {
            "logits": logits,
            "loss": lm_loss,
            "lm_loss": lm_loss,
            "avg_depth": avg_depth,
        }

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        embedding = self.model.backbone.embeddings.weight.numel()

        non_embedding = total - embedding

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embedding,
            "non_embedding_parameters": non_embedding,
        }