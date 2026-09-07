from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    MambaConfig as HFMambaConfig,
    MambaForCausalLM,
)


@dataclass
class MambaModelConfig:
    """
    Standard Mamba configuration.

    Values are loaded from mamba.yaml.
    Only values missing from YAML should have defaults here.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    d_state: int
    expand: int
    d_conv: int
    max_seq_len: int
    tie_word_embeddings: bool
    use_bias: bool
    use_conv_bias: bool
    residual_in_fp32: bool
    layer_norm_eps: float


class MambaCausalLM(nn.Module):
    """
    Standard Mamba baseline.

    No:
        - recursion
        - routing
        - LTE
        - grouped parameterization
        - Markov head
        - TurboQuant

    This model is trained from scratch and becomes the parent
    checkpoint for Recursive-Mamba.
    """

    def __init__(
        self,
        config: MambaModelConfig,
    ):
        super().__init__()

        self.cfg = config

        hf_config = HFMambaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            state_size=config.d_state,
            num_hidden_layers=config.n_layers,
            expand=config.expand,
            conv_kernel=config.d_conv,
            use_bias=config.use_bias,
            use_conv_bias=config.use_conv_bias,
            residual_in_fp32=config.residual_in_fp32,
            layer_norm_epsilon=config.layer_norm_eps,
            hidden_act="silu",
            tie_word_embeddings=config.tie_word_embeddings,
            # No recurrent cache needed during pretraining.
            use_cache=False,
            # Safe development fallback for Mac/MPS.
            use_mambapy=False,
            use_associative_scan=False,
        )

        self.model = MambaForCausalLM(
            hf_config
        )

    def forward(self,input_ids,labels=None,):
        """
        train.py prepares:

            x = tokens[:, :-1]
            y = tokens[:, 1:]

        Therefore we calculate CE ourselves instead of letting
        Hugging Face perform another internal causal shift.
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
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
            )

        # Compatibility with the current trainer.
        #
        # Later we will remove this ambiguous metric and store:
        #
        # physical_layers
        # effective_layers
        # avg_recursion_depth
        #
        # separately.
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

    def parameter_report(
        self,
    ) -> Dict[str, int]:

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        embedding = (
            self.model
            .backbone
            .embeddings
            .weight
            .numel()
        )

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embedding,
            "non_embedding_parameters":
                total - embedding,
        }

    def architecture_report(
        self,
    ) -> Dict:

        return {
            "architecture": "mamba",
            "physical_layers":
                self.cfg.n_layers,
            "effective_layers":
                self.cfg.n_layers,
            "recursive": False,
        }