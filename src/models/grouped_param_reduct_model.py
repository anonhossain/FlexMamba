from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.models.dynamic_mor_models import (
    DynamicMoRModelConfig,
    DynamicMoRCausalLM,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class GroupedParamReductionModelConfig(DynamicMoRModelConfig):
    pass


# ============================================================
# GROUPED LINEAR
# ============================================================

class GroupedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        groups,
        section_sizes=None,
        bias=False,
        shuffle_input=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.shuffle_input = shuffle_input

        if in_features % groups != 0:
            raise ValueError(
                f"in_features={in_features} must be divisible by groups={groups}"
            )

        self.in_per_group = in_features // groups

        if section_sizes is None:
            section_sizes = [out_features]

        if sum(section_sizes) != out_features:
            raise ValueError(
                f"section_sizes sum {sum(section_sizes)} != out_features {out_features}"
            )

        for size in section_sizes:
            if size % groups != 0:
                raise ValueError(
                    f"Section size {size} must be divisible by groups={groups}"
                )

        self.section_sizes = list(section_sizes)

        self.section_weights = nn.ParameterList()

        for size in self.section_sizes:
            out_per_group = size // groups

            weight = nn.Parameter(
                torch.empty(
                    groups,
                    out_per_group,
                    self.in_per_group,
                )
            )

            self.section_weights.append(weight)

        if bias:

            self.section_biases = nn.ParameterList()

            for size in self.section_sizes:
                out_per_group = size // groups

                self.section_biases.append(
                    nn.Parameter(
                        torch.empty(
                            groups,
                            out_per_group,
                        )
                    )
                )

        else:
            self.section_biases = None

        permutation = self._build_shuffle_permutation(
            in_features,
            groups,
        )

        self.register_buffer(
            "_input_permutation",
            permutation,
            persistent=False,
        )

        self.reset_parameters()


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def reset_parameters(self):

        for section in self.section_weights:

            for group_index in range(self.groups):
                nn.init.kaiming_uniform_(
                    section[group_index],
                    a=math.sqrt(5),
                )

        if self.section_biases is not None:

            bound = 1 / math.sqrt(self.in_per_group)

            for section in self.section_biases:
                nn.init.uniform_(section, -bound, bound)


    # ========================================================
    # CHANNEL SHUFFLE
    # ========================================================

    @staticmethod
    def _build_shuffle_permutation(channels, groups):

        if groups <= 1:
            return torch.arange(channels)

        channels_per_group = channels // groups

        return (
            torch.arange(channels)
            .reshape(groups, channels_per_group)
            .transpose(0, 1)
            .reshape(-1)
        )


    def _shuffle(self, x):

        if not self.shuffle_input or self.groups <= 1:
            return x

        return x.index_select(
            -1,
            self._input_permutation,
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        prefix = x.shape[:-1]

        x = self._shuffle(x)

        grouped_input = x.reshape(
            *prefix,
            self.groups,
            self.in_per_group,
        )

        outputs = []

        for section_index, weight in enumerate(self.section_weights):

            output = torch.einsum(
                "...gi,goi->...go",
                grouped_input,
                weight,
            )

            if self.section_biases is not None:
                output = output + self.section_biases[section_index]

            outputs.append(
                output.reshape(
                    *prefix,
                    -1,
                )
            )

        return torch.cat(outputs, dim=-1)


    # ========================================================
    # DENSE COMPATIBILITY
    # ========================================================

    @property
    def weight(self):

        sections = []

        for section in self.section_weights:

            blocks = [
                section[group_index]
                for group_index in range(self.groups)
            ]

            dense_section = torch.block_diag(*blocks)
            sections.append(dense_section)

        dense_weight = torch.cat(
            sections,
            dim=0,
        )

        if self.shuffle_input and self.groups > 1:

            inverse = torch.argsort(
                self._input_permutation
            )

            dense_weight = dense_weight.index_select(
                1,
                inverse,
            )

        return dense_weight


    @property
    def bias(self):

        if self.section_biases is None:
            return None

        return torch.cat(
            [
                section.reshape(-1)
                for section in self.section_biases
            ],
            dim=0,
        )


    # ========================================================
    # LOAD DENSE PARENT
    # ========================================================

    @torch.no_grad()
    def load_from_dense(self, dense_weight, dense_bias=None):

        if dense_weight.shape != (
            self.out_features,
            self.in_features,
        ):
            raise ValueError(
                f"Dense weight shape {dense_weight.shape} != "
                f"({self.out_features}, {self.in_features})"
            )

        source = dense_weight

        if self.shuffle_input and self.groups > 1:
            source = source.index_select(
                1,
                self._input_permutation,
            )

        output_offset = 0

        for section_index, section_size in enumerate(self.section_sizes):

            out_per_group = section_size // self.groups
            target_weight = self.section_weights[section_index]

            for group_index in range(self.groups):

                input_start = (
                    group_index
                    * self.in_per_group
                )

                input_end = (
                    input_start
                    + self.in_per_group
                )

                output_start = (
                    output_offset
                    + group_index
                    * out_per_group
                )

                output_end = (
                    output_start
                    + out_per_group
                )

                target_weight[group_index].copy_(
                    source[
                        output_start:output_end,
                        input_start:input_end,
                    ]
                )

                if (
                    self.section_biases is not None
                    and dense_bias is not None
                ):
                    self.section_biases[
                        section_index
                    ][group_index].copy_(
                        dense_bias[
                            output_start:output_end
                        ]
                    )

            output_offset += section_size


    # ========================================================
    # PARAMETER COUNTS
    # ========================================================

    def grouped_parameter_count(self):

        total = sum(
            p.numel()
            for p in self.section_weights
        )

        if self.section_biases is not None:
            total += sum(
                p.numel()
                for p in self.section_biases
            )

        return total


    def dense_parameter_count(self):

        total = (
            self.in_features
            * self.out_features
        )

        if self.section_biases is not None:
            total += self.out_features

        return total


# ============================================================
# GROUPED DYNAMIC MAMBA
# ============================================================

class GroupedParamReductionCausalLM(DynamicMoRCausalLM):

    def __init__(
        self,
        config,
        recursion_cfg,
        state_cfg,
        routing_cfg,
        grouped_cfg,
    ):

        super().__init__(
            config,
            recursion_cfg,
            state_cfg,
            routing_cfg,
        )

        self.grouped_cfg = grouped_cfg

        if not grouped_cfg.get("enabled", True):
            raise ValueError(
                "S05 requires grouped_parameterization.enabled=true"
            )

        self.groups = grouped_cfg.get(
            "groups",
            4,
        )

        if self.groups < 1:
            raise ValueError("groups must be >= 1.")

        self._grouped_projections = []
        self._dense_projection_parameters = 0
        self._grouped_projection_parameters = 0

        self._apply_grouped_parameterization()


    # ========================================================
    # REPLACE PROJECTIONS
    # ========================================================

    def _replace_projection(
        self,
        layer_index,
        mixer,
        projection_name,
        section_sizes,
    ):

        dense = getattr(
            mixer,
            projection_name,
        )

        if not isinstance(dense, nn.Linear):
            raise TypeError(
                f"{projection_name} is not nn.Linear."
            )

        grouped = GroupedLinear(
            in_features=dense.in_features,
            out_features=dense.out_features,
            groups=self.groups,
            section_sizes=section_sizes,
            bias=dense.bias is not None,
            shuffle_input=self.grouped_cfg.get(
                "channel_shuffle",
                False,
            ),
        )

        setattr(
            mixer,
            projection_name,
            grouped,
        )

        prefix = (
            f"model.backbone.layers.{layer_index}"
            f".mixer.{projection_name}"
        )

        self._grouped_projections.append(
            {
                "prefix": prefix,
                "name": projection_name,
                "layer": layer_index,
                "module": grouped,
            }
        )

        self._dense_projection_parameters += (
            grouped.dense_parameter_count()
        )

        self._grouped_projection_parameters += (
            grouped.grouped_parameter_count()
        )


    def _apply_grouped_parameterization(self):

        for layer_index, layer in enumerate(
            self.model.backbone.layers
        ):

            mixer = layer.mixer

            # ------------------------------------------------
            # MAMBA IN PROJECTION
            #
            # Preserves the two semantic halves:
            # hidden-state branch + gate branch.
            # ------------------------------------------------

            if self.grouped_cfg.get(
                "group_input_projection",
                True,
            ):

                self._replace_projection(
                    layer_index,
                    mixer,
                    "in_proj",
                    [
                        mixer.intermediate_size,
                        mixer.intermediate_size,
                    ],
                )

            # ------------------------------------------------
            # SSM PROJECTION
            #
            # Preserve:
            # dt | B | C
            # ------------------------------------------------

            if self.grouped_cfg.get(
                "group_ssm_projection",
                True,
            ):

                self._replace_projection(
                    layer_index,
                    mixer,
                    "x_proj",
                    [
                        mixer.time_step_rank,
                        mixer.ssm_state_size,
                        mixer.ssm_state_size,
                    ],
                )

            # ------------------------------------------------
            # OUTPUT PROJECTION
            # ------------------------------------------------

            if self.grouped_cfg.get(
                "group_output_projection",
                True,
            ):

                self._replace_projection(
                    layer_index,
                    mixer,
                    "out_proj",
                    [
                        mixer.out_proj.out_features,
                    ],
                )


    # ========================================================
    # LOAD S04 PARENT
    # ========================================================

    def load_parent_state_dict(self, parent_state):

        target_state = self.state_dict()

        grouped_prefixes = [
            item["prefix"]
            for item in self._grouped_projections
        ]

        mapped_state = {}

        # ----------------------------------------------------
        # COPY ALL UNCHANGED PARAMETERS
        # ----------------------------------------------------

        for key, target_tensor in target_state.items():

            if any(
                key.startswith(prefix + ".")
                for prefix in grouped_prefixes
            ):
                continue

            if key not in parent_state:
                continue

            source_tensor = parent_state[key]

            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Parent/child tensor mismatch for {key}: "
                    f"{source_tensor.shape} != {target_tensor.shape}"
                )

            mapped_state[key] = source_tensor.clone()

        missing, unexpected = self.load_state_dict(
            mapped_state,
            strict=False,
        )

        invalid_missing = [
            key
            for key in missing
            if not any(
                key.startswith(prefix + ".")
                for prefix in grouped_prefixes
            )
        ]

        if invalid_missing:
            raise RuntimeError(
                "Unexpected missing parent parameters:\n"
                + "\n".join(invalid_missing)
            )

        # ----------------------------------------------------
        # CONVERT DENSE PROJECTIONS → GROUPED PROJECTIONS
        # ----------------------------------------------------

        for item in self._grouped_projections:

            prefix = item["prefix"]
            module = item["module"]

            weight_key = prefix + ".weight"
            bias_key = prefix + ".bias"

            if weight_key not in parent_state:
                raise KeyError(
                    f"Parent projection not found: {weight_key}"
                )

            dense_weight = parent_state[
                weight_key
            ]

            dense_bias = parent_state.get(
                bias_key,
                None,
            )

            module.load_from_dense(
                dense_weight,
                dense_bias,
            )

        child_total = sum(
            p.numel()
            for p in self.parameters()
        )

        grouped_total = (
            self._grouped_projection_parameters
        )

        directly_copied = (
            child_total
            - grouped_total
        )

        parent_equivalent = (
            child_total
            + self.parameter_savings
        )

        return {
            "strategy": "group_parent_projections",

            "groups": self.groups,

            "directly_copied_parameter_count": directly_copied,

            "grouped_parameters_initialized_from_parent": grouped_total,

            "child_initialized_parameter_count": child_total,

            "child_initialized_fraction": 1.0,

            "parent_equivalent_parameter_count": parent_equivalent,

            "child_parameter_count": child_total,

            "parameter_savings": self.parameter_savings,

            "parameter_reduction_fraction": (
                self.parameter_savings
                / parent_equivalent
            ),

            "unexpected_keys": list(unexpected),
        }


    # ========================================================
    # GROUPING REPORT
    # ========================================================

    @property
    def parameter_savings(self):

        return (
            self._dense_projection_parameters
            - self._grouped_projection_parameters
        )


    def grouping_report(self):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        parent_equivalent = (
            total
            + self.parameter_savings
        )

        projection_reduction = (
            1.0
            - (
                self._grouped_projection_parameters
                / self._dense_projection_parameters
            )
        )

        overall_reduction = (
            self.parameter_savings
            / parent_equivalent
        )

        return {
            "enabled": True,

            "groups": self.groups,

            "channel_shuffle": self.grouped_cfg.get(
                "channel_shuffle",
                False,
            ),

            "grouped_projection_count": len(
                self._grouped_projections
            ),

            "dense_projection_parameters": (
                self._dense_projection_parameters
            ),

            "grouped_projection_parameters": (
                self._grouped_projection_parameters
            ),

            "projection_parameter_reduction": (
                projection_reduction
            ),

            "parameter_savings": (
                self.parameter_savings
            ),

            "parent_equivalent_parameters": (
                parent_equivalent
            ),

            "grouped_model_parameters": total,

            "overall_parameter_reduction": (
                overall_reduction
            ),
        }


    # ========================================================
    # PARAMETER REPORT
    # ========================================================

    def parameter_report(self):

        base = super().parameter_report()

        grouping = self.grouping_report()

        parent_non_embedding = (
            grouping["parent_equivalent_parameters"]
            - base["embedding_parameters"]
        )

        child_non_embedding = (
            base["total_parameters"]
            - base["embedding_parameters"]
        )

        base.update(
            {
                "grouped_projection_parameters": (
                    grouping[
                        "grouped_projection_parameters"
                    ]
                ),

                "dense_parent_equivalent_parameters": (
                    grouping[
                        "parent_equivalent_parameters"
                    ]
                ),

                "parameter_savings": (
                    grouping[
                        "parameter_savings"
                    ]
                ),

                "overall_parameter_reduction": (
                    grouping[
                        "overall_parameter_reduction"
                    ]
                ),

                "non_embedding_parameter_reduction": (
                    1.0
                    - (
                        child_non_embedding
                        / parent_non_embedding
                    )
                ),
            }
        )

        return base


    # ========================================================
    # ARCHITECTURE REPORT
    # ========================================================

    def architecture_report(self):

        report = super().architecture_report()

        report.update(
            {
                "architecture": "grouped_dynamic_mamba",

                "grouped_parameterization": True,

                "groups": self.groups,

                "group_input_projection": (
                    self.grouped_cfg.get(
                        "group_input_projection",
                        True,
                    )
                ),

                "group_ssm_projection": (
                    self.grouped_cfg.get(
                        "group_ssm_projection",
                        True,
                    )
                ),

                "group_output_projection": (
                    self.grouped_cfg.get(
                        "group_output_projection",
                        True,
                    )
                ),

                "channel_shuffle": (
                    self.grouped_cfg.get(
                        "channel_shuffle",
                        False,
                    )
                ),
            }
        )

        return report