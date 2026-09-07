# will be deleted
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
class RecursiveMambaConfig:
    """
    Static Recursive-Mamba configuration.

    Example:

        input_layers = 1
        shared_middle_layers = 2
        num_recursions = 3
        output_layers = 1

    Physical layers:

        1 + 2 + 1 = 4

    Effective layers:

        1 + (2 * 3) + 1 = 8
    """

    vocab_size: int = 49152

    d_model: int = 256

    input_layers: int = 1

    shared_middle_layers: int = 2

    num_recursions: int = 3

    output_layers: int = 1

    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    max_seq_len: int = 256

    tie_word_embeddings: bool = True

    use_bias: bool = False
    use_conv_bias: bool = True

    residual_in_fp32: bool = True

    layer_norm_eps: float = 1e-5

    def __post_init__(self):

        if self.input_layers < 1:
            raise ValueError(
                "input_layers must be >= 1"
            )

        if self.shared_middle_layers < 1:
            raise ValueError(
                "shared_middle_layers must be >= 1"
            )

        if self.num_recursions < 1:
            raise ValueError(
                "num_recursions must be >= 1"
            )

        if self.output_layers < 1:
            raise ValueError(
                "output_layers must be >= 1"
            )

    @property
    def physical_layers(
        self,
    ) -> int:

        return (
            self.input_layers
            + self.shared_middle_layers
            + self.output_layers
        )

    @property
    def effective_layers(
        self,
    ) -> int:

        return (
            self.input_layers
            + (
                self.shared_middle_layers
                * self.num_recursions
            )
            + self.output_layers
        )


class RecursiveMambaCausalLM(nn.Module):
    """
    Static Recursive-Mamba.

    Weight structure:

        Input block(s)
             ↓
        Shared middle pool
             ↓
        repeated R times
             ↓
        Output block(s)

    Example:

        Physical blocks:

            0  A  B  3

        Effective computation:

            0
            A B
            A B
            A B
            3

    There is NO dynamic token router yet.

    That is deliberately reserved for dynamic_routing.py.
    """

    def __init__(
        self,
        config: RecursiveMambaConfig,
    ):
        super().__init__()

        self.cfg = config

        hf_config = HFMambaConfig(
            vocab_size=config.vocab_size,

            hidden_size=config.d_model,

            state_size=config.d_state,

            # Only unique physical blocks
            # are actually instantiated.
            num_hidden_layers=(
                config.physical_layers
            ),

            expand=config.expand,

            conv_kernel=config.d_conv,

            use_bias=config.use_bias,

            use_conv_bias=config.use_conv_bias,

            residual_in_fp32=(
                config.residual_in_fp32
            ),

            layer_norm_epsilon=(
                config.layer_norm_eps
            ),

            hidden_act="silu",

            tie_word_embeddings=(
                config.tie_word_embeddings
            ),

            use_cache=False,

            use_mambapy=False,
            use_associative_scan=False,
        )

        self.model = MambaForCausalLM(
            hf_config
        )

    def forward(
        self,
        input_ids,
        labels=None,
    ):

        backbone = self.model.backbone

        hidden = backbone.embeddings(
            input_ids
        )

        layers = backbone.layers

        layer_index = 0

        # ====================================================
        # UNIQUE INPUT BLOCKS
        # ====================================================

        for _ in range(
            self.cfg.input_layers
        ):

            hidden = layers[
                layer_index
            ](
                hidden,
                cache_params=None,
            )

            layer_index += 1

        # ====================================================
        # SHARED MIDDLE BLOCKS
        # ====================================================

        shared_start = layer_index

        for _ in range(
            self.cfg.num_recursions
        ):

            for j in range(
                self.cfg.shared_middle_layers
            ):

                block = layers[
                    shared_start + j
                ]

                hidden = block(
                    hidden,
                    cache_params=None,
                )

        # ====================================================
        # UNIQUE OUTPUT BLOCKS
        # ====================================================

        output_start = (
            shared_start
            + self.cfg.shared_middle_layers
        )

        for j in range(
            self.cfg.output_layers
        ):

            hidden = layers[
                output_start + j
            ](
                hidden,
                cache_params=None,
            )

        # ====================================================
        # FINAL NORMALIZATION + LM HEAD
        # ====================================================

        hidden = backbone.norm_f(
            hidden
        )

        logits = self.model.lm_head(
            hidden
        )

        lm_loss = None

        if labels is not None:

            lm_loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
            )

        recursion_depth = torch.tensor(
            float(
                self.cfg.num_recursions
            ),
            device=logits.device,
        )

        return {
            "logits": logits,

            "loss": lm_loss,

            "lm_loss": lm_loss,

            # Current trainer compatibility.
            "avg_depth":
                recursion_depth,

            # Better explicit metric for the
            # future training/evaluation system.
            "avg_recursion_depth":
                recursion_depth,
        }

    # ========================================================
    # PARAMETER REPORT
    # ========================================================

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
            "total_parameters":
                total,

            "trainable_parameters":
                trainable,

            "embedding_parameters":
                embedding,

            "non_embedding_parameters":
                total - embedding,
        }

    # ========================================================
    # ARCHITECTURE REPORT
    # ========================================================

    def architecture_report(
        self,
    ) -> Dict:

        return {
            "architecture":
                "recursive_mamba",

            "physical_layers":
                self.cfg.physical_layers,

            "effective_layers":
                self.cfg.effective_layers,

            "input_layers":
                self.cfg.input_layers,

            "shared_middle_layers":
                self.cfg.shared_middle_layers,

            "num_recursions":
                self.cfg.num_recursions,

            "output_layers":
                self.cfg.output_layers,

            "recursive":
                True,
        }

    # ========================================================
    # INITIALIZE FROM STANDARD MAMBA CHECKPOINT
    # ========================================================

    def load_parent_state_dict(
        self,
        parent_state,
        parent_config,
        strategy="cycle_average",
    ):
        """
        Convert an ordinary Mamba checkpoint into
        Recursive-Mamba.

        Example parent:

            layer 0
            layer 1
            layer 2
            layer 3
            layer 4
            layer 5
            layer 6
            layer 7

        Recursive architecture:

            input:
                layer 0

            shared A:
                average(1, 3, 5)

            shared B:
                average(2, 4, 6)

            output:
                layer 7
        """

        if strategy != "cycle_average":

            raise ValueError(
                "Recursive-Mamba currently supports "
                "only strategy='cycle_average'."
            )

        parent_model_cfg = (
            parent_config["model"]
        )

        parent_layers = (
            parent_model_cfg["n_layers"]
        )

        expected_layers = (
            self.cfg.effective_layers
        )

        if parent_layers != expected_layers:

            raise ValueError(
                "\nParent depth does not match "
                "Recursive-Mamba effective depth.\n"
                f"Parent layers: {parent_layers}\n"
                f"Recursive effective layers: "
                f"{expected_layers}"
            )

        target_state = (
            self.state_dict()
        )

        mapped_state = {}

        # ====================================================
        # COPY NON-LAYER PARAMETERS
        #
        # embeddings
        # final norm
        # LM head
        # ====================================================

        for key, value in (
            target_state.items()
        ):

            if (
                "model.backbone.layers."
                in key
            ):
                continue

            if key not in parent_state:
                continue

            if (
                parent_state[key].shape
                != value.shape
            ):
                continue

            mapped_state[key] = (
                parent_state[key]
                .clone()
            )

        # ====================================================
        # HELPER FOR COPYING / AVERAGING LAYERS
        # ====================================================

        def map_layer(
            target_layer,
            source_layers,
        ):

            target_prefix = (
                "model.backbone.layers."
                f"{target_layer}."
            )

            for target_key in (
                target_state
            ):

                if not target_key.startswith(
                    target_prefix
                ):
                    continue

                suffix = target_key[
                    len(target_prefix):
                ]

                source_values = []

                for source_layer in (
                    source_layers
                ):

                    source_key = (
                        "model.backbone.layers."
                        f"{source_layer}."
                        f"{suffix}"
                    )

                    if (
                        source_key
                        not in parent_state
                    ):

                        raise KeyError(
                            f"Missing parent key: "
                            f"{source_key}"
                        )

                    source_values.append(
                        parent_state[
                            source_key
                        ]
                    )

                first = (
                    source_values[0]
                )

                if (
                    first.dtype
                    .is_floating_point
                ):

                    stacked = (
                        torch.stack(
                            [
                                tensor.float()
                                for tensor
                                in source_values
                            ],
                            dim=0,
                        )
                    )

                    mapped_value = (
                        stacked
                        .mean(dim=0)
                        .to(first.dtype)
                    )

                else:

                    mapped_value = (
                        first.clone()
                    )

                mapped_state[
                    target_key
                ] = mapped_value

        # ====================================================
        # INPUT LAYERS
        # ====================================================

        for i in range(
            self.cfg.input_layers
        ):

            map_layer(
                target_layer=i,
                source_layers=[i],
            )

        # ====================================================
        # SHARED MIDDLE POOL
        # ====================================================

        for j in range(
            self.cfg.shared_middle_layers
        ):

            source_layers = [

                (
                    self.cfg.input_layers
                    + (
                        recursion
                        * self.cfg.shared_middle_layers
                    )
                    + j
                )

                for recursion in range(
                    self.cfg.num_recursions
                )
            ]

            target_layer = (
                self.cfg.input_layers
                + j
            )

            map_layer(
                target_layer=
                    target_layer,

                source_layers=
                    source_layers,
            )

        # ====================================================
        # OUTPUT LAYERS
        # ====================================================

        target_output_start = (
            self.cfg.input_layers
            + self.cfg.shared_middle_layers
        )

        source_output_start = (
            self.cfg.effective_layers
            - self.cfg.output_layers
        )

        for j in range(
            self.cfg.output_layers
        ):

            map_layer(
                target_layer=(
                    target_output_start
                    + j
                ),

                source_layers=[
                    source_output_start
                    + j
                ],
            )

        # ====================================================
        # LOAD
        # ====================================================

        missing_keys, unexpected_keys = (
            self.load_state_dict(
                mapped_state,
                strict=False,
            )
        )

        loaded_elements = sum(
            tensor.numel()
            for tensor
            in mapped_state.values()
        )

        total_elements = sum(
            tensor.numel()
            for tensor
            in target_state.values()
        )

        return {
            "strategy":
                strategy,

            "loaded_tensor_count":
                len(mapped_state),

            "loaded_parameter_fraction":
                loaded_elements
                / total_elements,

            "missing_keys":
                list(missing_keys),

            "unexpected_keys":
                list(unexpected_keys),
        }