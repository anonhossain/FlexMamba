from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MambaConfig as HFMambaConfig
from transformers import MambaForCausalLM


@dataclass
class RecursiveMambaModelConfig:
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


class RecursiveMambaCausalLM(nn.Module):

    def __init__(self, config: RecursiveMambaModelConfig):
        super().__init__()
        self.cfg = config

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

    def _run_layer(self, layer, hidden):
        output = layer(hidden, cache_params=None)
        return output[0] if isinstance(output, (tuple, list)) else output

    def forward(self, input_ids, labels=None):
        backbone = self.model.backbone
        layers = backbone.layers
        hidden = backbone.embeddings(input_ids)

        index = 0

        for _ in range(self.cfg.input_layers):
            hidden = self._run_layer(layers[index], hidden)
            index += 1

        shared_start = index

        for _ in range(self.cfg.num_recursions):
            for j in range(self.cfg.shared_middle_layers):
                hidden = self._run_layer(layers[shared_start + j], hidden)

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

        depth = torch.tensor(float(self.cfg.num_recursions), device=logits.device)

        return {
            "logits": logits,
            "loss": lm_loss,
            "lm_loss": lm_loss,
            "avg_depth": depth,
            "avg_recursion_depth": depth,
        }

    def load_mamba_parent_state_dict(self, parent_state, parent_n_layers, strategy="cycle_average"):

        if strategy != "cycle_average":
            raise ValueError("Only cycle_average is supported.")

        if parent_n_layers != self.cfg.effective_layers:
            raise ValueError(
                f"Parent has {parent_n_layers} layers but recursive model "
                f"requires {self.cfg.effective_layers} effective layers."
            )

        target_state = self.state_dict()
        mapped_state = {}
        layer_mapping = {}

        # Copy embeddings, LM head, norm and other compatible non-layer tensors.
        for key, target in target_state.items():

            if "model.backbone.layers." in key:
                continue

            if key in parent_state and parent_state[key].shape == target.shape:
                mapped_state[key] = parent_state[key].clone()

        def map_layer(target_layer, source_layers):

            layer_mapping[str(target_layer)] = source_layers
            prefix = f"model.backbone.layers.{target_layer}."

            for target_key in target_state:

                if not target_key.startswith(prefix):
                    continue

                suffix = target_key[len(prefix):]

                source_values = []

                for source_layer in source_layers:
                    source_key = f"model.backbone.layers.{source_layer}.{suffix}"

                    if source_key not in parent_state:
                        raise KeyError(f"Missing parent tensor: {source_key}")

                    source_values.append(parent_state[source_key])

                if len(source_values) == 1:
                    value = source_values[0].clone()

                elif source_values[0].dtype.is_floating_point:
                    value = torch.stack(
                        [x.float() for x in source_values]
                    ).mean(0).to(source_values[0].dtype)

                else:
                    value = source_values[0].clone()

                mapped_state[target_key] = value

        # Unique input layers.
        for i in range(self.cfg.input_layers):
            map_layer(i, [i])

        # Shared middle layers.
        for j in range(self.cfg.shared_middle_layers):

            sources = [
                self.cfg.input_layers
                + recursion * self.cfg.shared_middle_layers
                + j
                for recursion in range(self.cfg.num_recursions)
            ]

            target = self.cfg.input_layers + j
            map_layer(target, sources)

        # Unique output layers.
        target_start = self.cfg.input_layers + self.cfg.shared_middle_layers
        source_start = self.cfg.effective_layers - self.cfg.output_layers

        for j in range(self.cfg.output_layers):
            map_layer(target_start + j, [source_start + j])

        missing, unexpected = self.load_state_dict(mapped_state, strict=False)

        if missing:
            raise RuntimeError("Missing initialized parameters:\n" + "\n".join(missing))

        return {
            "strategy": strategy,
            "parent_layers": parent_n_layers,
            "physical_layers": self.cfg.physical_layers,
            "effective_layers": self.cfg.effective_layers,
            "layer_mapping": layer_mapping,
            "mapped_tensor_count": len(mapped_state),
            "unexpected_keys": list(unexpected),
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
            "architecture": "static_recursive_mamba",
            "physical_layers": self.cfg.physical_layers,
            "effective_layers": self.cfg.effective_layers,
            "input_layers": self.cfg.input_layers,
            "shared_middle_layers": self.cfg.shared_middle_layers,
            "num_recursions": self.cfg.num_recursions,
            "output_layers": self.cfg.output_layers,
            "recursive": True,
            "dynamic_routing": False,
        }