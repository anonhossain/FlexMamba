# src/models/mamba.py

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

    layer_norm_eps: float = 1e-5

class MambaCausalLM(nn.Module):

    def __init__(self, config: MambaModelConfig):
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
            use_cache=False,
            # Safe generic implementation.
            use_mambapy=False,
            use_associative_scan=False,
        )

        self.model = MambaForCausalLM(
            hf_config
        )
    # ========================================================
    # FORWARD
    # ========================================================
    def forward(
        self,
        input_ids,
        labels=None,
    ):
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

        return {
            "logits": logits,
            "loss": lm_loss,
            "lm_loss": lm_loss,
        }
    # ========================================================
    # LOAD KNOWLEDGE FROM MoR
    # ========================================================

    def load_mor_parent_state_dict(
        self,
        parent_state,
        ):
        target_state = self.state_dict()

        mapping = {
            "model.backbone.embeddings.weight":
                "embed_tokens.weight",

            "model.lm_head.weight":
                "lm_head.weight",

            "model.backbone.norm_f.weight":
                "final_norm.weight",
        }

        mapped_state = {}

        # Includes aliases for tied parameters.
        target_parameters = dict(
            self.named_parameters(
                remove_duplicate=False
            )
        )

        transferred_parameter_ids = set()
        transferred_parameter_count = 0

        for target_key, parent_key in mapping.items():

            if parent_key not in parent_state:
                continue

            if target_key not in target_state:
                continue

            parent_tensor = parent_state[parent_key]
            target_tensor = target_state[target_key]

            if parent_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Incompatible tensor:\n"
                    f"{parent_key}: {parent_tensor.shape}\n"
                    f"{target_key}: {target_tensor.shape}"
                )

            mapped_state[target_key] = (
                parent_tensor.clone()
            )

            # Count tied parameters only once.
            parameter = target_parameters.get(
                target_key
            )

            if (
                parameter is not None
                and id(parameter)
                not in transferred_parameter_ids
            ):
                transferred_parameter_ids.add(
                    id(parameter)
                )

                transferred_parameter_count += (
                    parameter.numel()
                )

        if not mapped_state:
            raise RuntimeError(
                "No compatible MoR parameters "
                "were transferred."
            )

        missing_keys, unexpected_keys = (
            self.load_state_dict(
                mapped_state,
                strict=False,
            )
        )

        total_parameters = sum(
            p.numel()
            for p in self.parameters()
        )

        return {
            "strategy":
                "mor_compatible_transfer",

            "transferred_tensors":
                list(mapped_state.keys()),

            "transferred_parameter_count":
                transferred_parameter_count,

            "total_parameter_count":
                total_parameters,

            "transferred_parameter_fraction":
                transferred_parameter_count
                / total_parameters,

            "new_parameter_count":
                total_parameters
                - transferred_parameter_count,

            "missing_keys":
                list(missing_keys),

            "unexpected_keys":
                list(unexpected_keys),
        }

    # ========================================================
    # REPORTS
    # ========================================================

    def parameter_report(self) -> Dict[str, int]:

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

    def architecture_report(self) -> Dict:

        return {
            "architecture": "mamba",

            "physical_layers":
                self.cfg.n_layers,

            "effective_layers":
                self.cfg.n_layers,

            "d_state":
                self.cfg.d_state,

            "expand":
                self.cfg.expand,

            "d_conv":
                self.cfg.d_conv,

            "recursive":
                False,
        }