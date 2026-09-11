from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.dynamic_mor_models import (
    DynamicMoRModelConfig,
    DynamicMoRCausalLM,
)

from src.models.grouped_param_reduct_model import GroupedLinear

from src.models.lte_adaptive_width_model import (
    LTEInputProjection,
    LTEOutputProjection,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class GroupedLTEModelConfig(DynamicMoRModelConfig):
    pass


# ============================================================
# GROUPED + LTE DYNAMIC MAMBA
# ============================================================

class GroupedLTECausalLM(DynamicMoRCausalLM):

    def __init__(
        self,
        config,
        recursion_cfg,
        state_cfg,
        routing_cfg,
        grouped_cfg,
        lte_cfg,
    ):
        super().__init__(
            config,
            recursion_cfg,
            state_cfg,
            routing_cfg,
        )

        self.grouped_cfg = grouped_cfg
        self.lte_cfg = lte_cfg

        if not grouped_cfg.get("enabled", True):
            raise ValueError("S07 requires grouped parameterization.")

        if not lte_cfg.get("enabled", True):
            raise ValueError("S07 requires LTE.")

        if lte_cfg.get("apply_to", "shared_middle") != "shared_middle":
            raise ValueError("S07 LTE currently applies to shared_middle.")

        self.groups = grouped_cfg.get("groups", 4)

        self._grouped_projections = []
        self._lte_layers = []

        self._dense_projection_parameters = 0
        self._grouped_projection_parameters = 0

        # Order matters:
        # Dense S04 -> Grouped projections -> LTE wrappers
        self._apply_grouped_parameterization()
        self._apply_lte()


    # ========================================================
    # GROUPED PARAMETERIZATION
    # ========================================================

    def _replace_projection(
        self,
        layer_index,
        mixer,
        projection_name,
        section_sizes,
    ):

        dense = getattr(mixer, projection_name)

        if not isinstance(dense, nn.Linear):
            raise TypeError(
                f"Expected dense nn.Linear for {projection_name}, "
                f"received {type(dense).__name__}"
            )

        grouped = GroupedLinear(
            in_features=dense.in_features,
            out_features=dense.out_features,
            groups=self.groups,
            section_sizes=section_sizes,
            bias=dense.bias is not None,
            shuffle_input=self.grouped_cfg.get("channel_shuffle", False),
        )

        setattr(mixer, projection_name, grouped)

        prefix = (
            f"model.backbone.layers.{layer_index}"
            f".mixer.{projection_name}"
        )

        self._grouped_projections.append(
            {
                "prefix": prefix,
                "layer": layer_index,
                "name": projection_name,
                "module": grouped,
            }
        )

        self._dense_projection_parameters += grouped.dense_parameter_count()
        self._grouped_projection_parameters += grouped.grouped_parameter_count()


    def _apply_grouped_parameterization(self):

        for layer_index, layer in enumerate(self.model.backbone.layers):

            mixer = layer.mixer

            if self.grouped_cfg.get("group_input_projection", True):
                self._replace_projection(
                    layer_index,
                    mixer,
                    "in_proj",
                    [
                        mixer.intermediate_size,
                        mixer.intermediate_size,
                    ],
                )

            if self.grouped_cfg.get("group_ssm_projection", True):
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

            if self.grouped_cfg.get("group_output_projection", True):
                self._replace_projection(
                    layer_index,
                    mixer,
                    "out_proj",
                    [mixer.out_proj.out_features],
                )


    # ========================================================
    # LTE
    # ========================================================

    def _apply_lte(self):

        shared_start = self.cfg.input_layers
        shared_end = shared_start + self.cfg.shared_middle_layers

        for layer_index in range(shared_start, shared_end):

            mixer = self.model.backbone.layers[layer_index].mixer

            if not isinstance(mixer.in_proj, GroupedLinear):
                raise TypeError(
                    f"Layer {layer_index} in_proj must be GroupedLinear."
                )

            if not isinstance(mixer.out_proj, GroupedLinear):
                raise TypeError(
                    f"Layer {layer_index} out_proj must be GroupedLinear."
                )

            lte_input = LTEInputProjection(
                linear=mixer.in_proj,
                intermediate_size=mixer.intermediate_size,
                lte_cfg=self.lte_cfg,
            )

            lte_output = LTEOutputProjection(
                linear=mixer.out_proj,
                input_projection=lte_input,
            )

            mixer.in_proj = lte_input
            mixer.out_proj = lte_output

            self._lte_layers.append(
                (layer_index, lte_input)
            )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, input_ids, labels=None):

        for _, lte_layer in self._lte_layers:
            lte_layer.reset_metrics()

        output = super().forward(
            input_ids,
            labels=labels,
        )

        efficiency_losses = []
        hard_active_bins = []
        soft_active_bins = []

        layer_metrics = {}
        total_token_decisions = 0

        for layer_index, lte_layer in self._lte_layers:

            metrics = lte_layer.collect_metrics()

            efficiency_losses.append(
                metrics["efficiency_loss"]
            )

            hard_active_bins.append(
                metrics["hard_active_bins"]
            )

            soft_active_bins.append(
                metrics["soft_active_bins"]
            )

            total_token_decisions += metrics["token_decisions"]

            layer_metrics[str(layer_index)] = {
                "average_active_bins": metrics["average_active_bins"].detach(),
                "active_fraction": metrics["active_fraction"].detach(),
                "soft_active_fraction": metrics["soft_active_fraction"].detach(),
                "token_decisions": metrics["token_decisions"],
            }

        lte_efficiency_loss = torch.stack(
            efficiency_losses
        ).mean()

        hard_active_bins = torch.stack(
            hard_active_bins
        ).sum()

        soft_active_bins = torch.stack(
            soft_active_bins
        ).sum()

        average_active_bins = (
            hard_active_bins
            / total_token_decisions
        )

        active_fraction = (
            average_active_bins
            / self.lte_cfg["num_bins"]
        )

        soft_active_fraction = (
            soft_active_bins
            / (
                total_token_decisions
                * self.lte_cfg["num_bins"]
            )
        )

        lte_aux_loss = (
            self.lte_cfg.get(
                "efficiency_loss_coeff",
                0.0,
            )
            * lte_efficiency_loss
        )

        if output["loss"] is not None:
            output["loss"] = output["loss"] + lte_aux_loss

        output.update(
            {
                "lte_efficiency_loss": lte_efficiency_loss,
                "lte_aux_loss": lte_aux_loss,

                "lte_average_active_bins": average_active_bins.detach(),
                "lte_active_fraction": active_fraction.detach(),
                "lte_soft_active_fraction": soft_active_fraction.detach(),

                "lte_active_bin_count": hard_active_bins.detach(),
                "lte_token_decisions": total_token_decisions,

                "lte_layer_stats": layer_metrics,
            }
        )

        return output


    # ========================================================
    # LOAD S04 DYNAMIC MoR PARENT
    # ========================================================

    def load_parent_state_dict(self, parent_state):

        target_state = self.state_dict()
        mapped_state = {}

        # ----------------------------------------------------
        # COPY UNCHANGED S04 PARAMETERS
        # ----------------------------------------------------

        for target_key, target_tensor in target_state.items():

            if ".section_weights." in target_key:
                continue

            if ".section_biases." in target_key:
                continue

            if ".lte_router." in target_key:
                continue

            if target_key not in parent_state:
                continue

            source_tensor = parent_state[target_key]

            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Parent/child mismatch for {target_key}: "
                    f"{source_tensor.shape} != {target_tensor.shape}"
                )

            mapped_state[target_key] = source_tensor.clone()

        missing, unexpected = self.load_state_dict(
            mapped_state,
            strict=False,
        )

        invalid_missing = [
            key
            for key in missing
            if (
                ".section_weights." not in key
                and ".section_biases." not in key
                and ".lte_router." not in key
            )
        ]

        if invalid_missing:
            raise RuntimeError(
                "Unexpected missing parent parameters:\n"
                + "\n".join(invalid_missing)
            )

        # ----------------------------------------------------
        # DENSE S04 -> GROUPED S07
        # ----------------------------------------------------

        for item in self._grouped_projections:

            prefix = item["prefix"]
            grouped = item["module"]

            weight_key = prefix + ".weight"
            bias_key = prefix + ".bias"

            if weight_key not in parent_state:
                raise KeyError(
                    f"Parent projection not found: {weight_key}"
                )

            grouped.load_from_dense(
                parent_state[weight_key],
                parent_state.get(bias_key),
            )

        child_total = sum(
            p.numel()
            for p in self.parameters()
        )

        lte_router_parameters = self.lte_router_parameter_count()

        directly_copied = (
            child_total
            - self._grouped_projection_parameters
            - lte_router_parameters
        )

        source_parent_parameters = (
            child_total
            - lte_router_parameters
            + self.parameter_savings
        )

        dense_combined_equivalent = (
            child_total
            + self.parameter_savings
        )

        parent_derived = (
            directly_copied
            + self._grouped_projection_parameters
        )

        return {
            "strategy": "initialize_combined_from_parent",

            "groups": self.groups,

            "source_parent_parameters": source_parent_parameters,

            "directly_copied_parameter_count": directly_copied,

            "grouped_parameters_initialized_from_parent": (
                self._grouped_projection_parameters
            ),

            "new_lte_router_parameters": lte_router_parameters,

            "child_parameter_count": child_total,

            "parent_derived_parameter_count": parent_derived,

            "parent_derived_fraction": (
                parent_derived / child_total
            ),

            "grouped_parameter_savings": self.parameter_savings,

            "dense_combined_equivalent_parameters": (
                dense_combined_equivalent
            ),

            "stored_parameter_reduction_fraction": (
                self.parameter_savings
                / dense_combined_equivalent
            ),

            "unexpected_keys": list(unexpected),
        }


    # ========================================================
    # COUNTS
    # ========================================================

    @property
    def parameter_savings(self):
        return (
            self._dense_projection_parameters
            - self._grouped_projection_parameters
        )


    def lte_router_parameter_count(self):

        return sum(
            p.numel()
            for _, lte_layer in self._lte_layers
            for p in lte_layer.lte_router.parameters()
        )


    def grouping_report(self):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        dense_equivalent = (
            total
            + self.parameter_savings
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
                1.0
                - self._grouped_projection_parameters
                / self._dense_projection_parameters
            ),

            "parameter_savings": self.parameter_savings,

            "grouped_model_parameters": total,

            "dense_combined_equivalent_parameters": (
                dense_equivalent
            ),

            "overall_parameter_reduction": (
                self.parameter_savings
                / dense_equivalent
            ),
        }


    # ========================================================
    # WIDTH-SENSITIVE PARAMETERS
    # ========================================================

    def width_sensitive_parameter_count(self):

        total = 0

        for layer_index, _ in self._lte_layers:

            mixer = (
                self.model
                .backbone
                .layers[layer_index]
                .mixer
            )

            modules = [
                mixer.in_proj.linear,
                mixer.conv1d,
                mixer.x_proj,
                mixer.dt_proj,
                mixer.out_proj.linear,
            ]

            for module in modules:
                total += sum(
                    p.numel()
                    for p in module.parameters()
                )

            total += mixer.A_log.numel()
            total += mixer.D.numel()

        return total


    def active_parameter_report(self, active_fraction):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        width_sensitive = (
            self.width_sensitive_parameter_count()
        )

        fixed = total - width_sensitive

        theoretical_active = (
            fixed
            + width_sensitive * active_fraction
        )

        return {
            "stored_parameters": total,

            "width_sensitive_parameters": width_sensitive,
            "fixed_parameters": fixed,

            "active_width_fraction": active_fraction,

            "theoretical_active_parameters": theoretical_active,

            "theoretical_active_parameter_fraction": (
                theoretical_active / total
            ),

            "note": (
                "Grouping reduces stored parameters. LTE changes "
                "token-dependent active width. Current LTE implementation "
                "still uses dense kernels."
            ),
        }


    # ========================================================
    # PARAMETER REPORT
    # ========================================================

    def parameter_report(self):

        report = super().parameter_report()
        grouping = self.grouping_report()

        report.update(
            {
                "lte_router_parameters": (
                    self.lte_router_parameter_count()
                ),

                "grouped_projection_parameters": (
                    self._grouped_projection_parameters
                ),

                "dense_projection_equivalent_parameters": (
                    self._dense_projection_parameters
                ),

                "grouped_parameter_savings": (
                    self.parameter_savings
                ),

                "overall_parameter_reduction": (
                    grouping[
                        "overall_parameter_reduction"
                    ]
                ),

                "width_sensitive_parameters": (
                    self.width_sensitive_parameter_count()
                ),
            }
        )

        return report


    # ========================================================
    # ARCHITECTURE REPORT
    # ========================================================

    def architecture_report(self):

        report = super().architecture_report()

        report.update(
            {
                "architecture": "grouped_lte_dynamic_mamba",

                "grouped_parameterization": True,
                "groups": self.groups,

                "group_input_projection": self.grouped_cfg.get(
                    "group_input_projection",
                    True,
                ),

                "group_ssm_projection": self.grouped_cfg.get(
                    "group_ssm_projection",
                    True,
                ),

                "group_output_projection": self.grouped_cfg.get(
                    "group_output_projection",
                    True,
                ),

                "channel_shuffle": self.grouped_cfg.get(
                    "channel_shuffle",
                    False,
                ),

                "lte_adaptive_width": True,
                "lte_apply_to": "shared_middle",

                "lte_layer_indices": [
                    layer_index
                    for layer_index, _
                    in self._lte_layers
                ],

                "lte_num_bins": self.lte_cfg["num_bins"],

                "lte_minimum_active_bins": (
                    self.lte_cfg[
                        "minimum_active_bins"
                    ]
                ),

                "lte_maximum_active_bins": (
                    self.lte_cfg[
                        "maximum_active_bins"
                    ]
                ),

                "lte_target_active_fraction": (
                    self.lte_cfg[
                        "target_active_fraction"
                    ]
                ),

                "lte_dense_compute_fallback": True,
            }
        )

        return report