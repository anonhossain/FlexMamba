from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MambaConfig as HFMambaConfig
from transformers import MambaForCausalLM

try:
    from transformers.cache_utils import MambaCache
except ImportError:
    MambaCache = None

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


@dataclass
class RecursionWiseStateModelConfig:
    vocab_size: int
    d_model: int

    input_layers: int
    shared_middle_layers: int
    num_recursions: int
    output_layers: int

    d_state: int
    expand: int
    d_conv: int
    max_seq_len: int

    tie_word_embeddings: bool
    use_bias: bool
    use_conv_bias: bool
    residual_in_fp32: bool

    layer_norm_eps: float = 1e-5

    @property
    def physical_layers(self):
        return self.input_layers + self.shared_middle_layers + self.output_layers

    @property
    def effective_layers(self):
        return (
            self.input_layers
            + self.shared_middle_layers * self.num_recursions
            + self.output_layers
        )


class RecursionWiseStateCausalLM(nn.Module):

    def __init__(self, config: RecursionWiseStateModelConfig, state_cfg):
        super().__init__()

        self.cfg = config
        self.state_cfg = state_cfg

        if state_cfg["mode"] != "recursion_wise":
            raise ValueError("state.mode must be 'recursion_wise'.")

        if state_cfg.get("share_state_across_recursions", False):
            raise ValueError("S03 requires independent state per recursion.")

        hf_config = HFMambaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            state_size=config.d_state,
            num_hidden_layers=config.physical_layers,
            expand=config.expand,
            conv_kernel=config.d_conv,
            use_bias=config.use_bias,
            use_conv_bias=config.use_conv_bias,
            residual_in_fp32=config.residual_in_fp32,
            layer_norm_epsilon=config.layer_norm_eps,
            hidden_act="silu",
            tie_word_embeddings=config.tie_word_embeddings,
            use_cache=False,
            use_mambapy=False,
            use_associative_scan=False,
        )

        self.model = MambaForCausalLM(hf_config)

    def _new_cache(self, hidden):

        batch_size = hidden.size(0)
        device = hidden.device
        dtype = hidden.dtype

        if MambaCache is not None:
            cache = MambaCache(
                self.model.config,
                batch_size,
                device=device,
                dtype=dtype,
            )

            cache_position = torch.arange(
                0,
                self.cfg.d_conv,
                device=device,
            )

            return cache, cache_position

        if DynamicCache is not None:

            try:
                cache = DynamicCache(config=self.model.config)
            except TypeError:
                cache = DynamicCache()

            return cache, None

        raise RuntimeError("No compatible Mamba cache class found.")

    def _run_layer(self, layer, hidden, cache=None, cache_position=None):

        kwargs = {}

        if cache is not None:
            kwargs["cache_params"] = cache

        if cache_position is not None:
            kwargs["cache_position"] = cache_position

        output = layer(hidden, **kwargs)

        return output[0] if isinstance(output, (tuple, list)) else output

    def forward(self, input_ids, labels=None):

        backbone = self.model.backbone
        layers = backbone.layers
        hidden = backbone.embeddings(input_ids)

        index = 0

        # Unique input layer
        for _ in range(self.cfg.input_layers):
            hidden = self._run_layer(layers[index], hidden)
            index += 1

        shared_start = index

        # Independent state/cache for every recursion.
        state_bank = [
            self._new_cache(hidden)
            for _ in range(self.cfg.num_recursions)
        ]

        for recursion in range(self.cfg.num_recursions):

            cache, cache_position = state_bank[recursion]

            for j in range(self.cfg.shared_middle_layers):
                hidden = self._run_layer(
                    layers[shared_start + j],
                    hidden,
                    cache=cache,
                    cache_position=cache_position,
                )

        # Unique output layer
        output_start = shared_start + self.cfg.shared_middle_layers

        for j in range(self.cfg.output_layers):
            hidden = self._run_layer(layers[output_start + j], hidden)

        hidden = backbone.norm_f(hidden)
        logits = self.model.lm_head(hidden)

        lm_loss = None

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        depth = torch.tensor(
            float(self.cfg.num_recursions),
            device=logits.device,
        )

        return {
            "logits": logits,
            "loss": lm_loss,
            "lm_loss": lm_loss,
            "avg_depth": depth,
            "avg_recursion_depth": depth,
            "state_report": self.state_report(
                batch_size=input_ids.size(0),
                dtype=hidden.dtype,
            ),
        }

    def load_parent_state_dict(self, parent_state):

        missing, unexpected = self.load_state_dict(
            parent_state,
            strict=True,
        )

        return {
            "strategy": "copy_parent_weights",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "physical_layers": self.cfg.physical_layers,
            "effective_layers": self.cfg.effective_layers,
            "num_recursions": self.cfg.num_recursions,
        }

    def state_report(self, batch_size=1, dtype=torch.float32):

        intermediate = self.cfg.d_model * self.cfg.expand

        state_elements = (
            batch_size
            * self.cfg.num_recursions
            * self.cfg.shared_middle_layers
            * intermediate
            * self.cfg.d_state
        )

        conv_elements = (
            batch_size
            * self.cfg.num_recursions
            * self.cfg.shared_middle_layers
            * intermediate
            * self.cfg.d_conv
        )

        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        total_bytes = (state_elements + conv_elements) * bytes_per_element

        return {
            "mode": "recursion_wise",
            "state_banks": self.cfg.num_recursions,
            "share_block_weights": True,
            "share_state_across_recursions": False,
            "reset_state_between_sequences": True,
            "ssm_state_elements": state_elements,
            "conv_state_elements": conv_elements,
            "theoretical_shared_state_memory_mb": total_bytes / (1024 ** 2),
        }

    def parameter_report(self) -> Dict[str, int]:

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = self.model.backbone.embeddings.weight.numel()

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embedding,
            "non_embedding_parameters": total - embedding,
        }

    def architecture_report(self) -> Dict:

        return {
            "architecture": "recursion_wise_state_mamba",
            "physical_layers": self.cfg.physical_layers,
            "effective_layers": self.cfg.effective_layers,
            "input_layers": self.cfg.input_layers,
            "shared_middle_layers": self.cfg.shared_middle_layers,
            "num_recursions": self.cfg.num_recursions,
            "output_layers": self.cfg.output_layers,
            "recursive": True,
            "dynamic_routing": False,
            "recursion_wise_state": True,
        }