from dataclasses import dataclass
import weakref

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
class LTEAdaptiveWidthModelConfig(DynamicMoRModelConfig):
    pass


# ============================================================
# LTE INPUT PROJECTION
# ============================================================

class LTEInputProjection(nn.Module):

    def __init__(self, linear, intermediate_size, lte_cfg):
        super().__init__()

        self.linear = linear
        self.intermediate_size = intermediate_size

        self.num_bins = lte_cfg["num_bins"]
        self.min_bins = lte_cfg["minimum_active_bins"]
        self.max_bins = lte_cfg["maximum_active_bins"]

        self.threshold = lte_cfg.get("threshold", 0.5)
        self.straight_through = lte_cfg.get("straight_through", True)
        self.target_active_fraction = lte_cfg.get("target_active_fraction", 0.5)

        if lte_cfg.get("router_type") != "sigmoid":
            raise ValueError("S06 LTE requires router_type='sigmoid'.")

        if intermediate_size % self.num_bins != 0:
            raise ValueError(
                f"Intermediate size {intermediate_size} must be "
                f"divisible by num_bins={self.num_bins}."
            )

        if not 1 <= self.min_bins <= self.max_bins <= self.num_bins:
            raise ValueError(
                "LTE bins must satisfy "
                "1 <= minimum_active_bins <= maximum_active_bins <= num_bins."
            )

        self.bin_size = intermediate_size // self.num_bins

        self.lte_router = nn.Linear(
            linear.in_features,
            self.num_bins,
        )

        self._initialize_router(lte_cfg)

        self._current_channel_gate = None
        self.reset_metrics()


    # ========================================================
    # COMPATIBILITY
    # ========================================================

    @property
    def weight(self):
        return self.linear.weight


    @property
    def bias(self):
        return self.linear.bias


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_router(self, lte_cfg):

        nn.init.zeros_(self.lte_router.weight)
        nn.init.zeros_(self.lte_router.bias)

        if lte_cfg.get("initialize_all_active", True):
            initial_logit = lte_cfg.get("initial_logit", 4.0)

            with torch.no_grad():
                self.lte_router.bias.fill_(initial_logit)


    # ========================================================
    # HARD BIN SELECTION
    # ========================================================

    def _hard_bin_mask(self, probabilities):

        threshold_mask = probabilities >= self.threshold

        if self.max_bins < self.num_bins:
            top_max = probabilities.topk(self.max_bins, dim=-1).indices

            maximum_mask = torch.zeros_like(
                threshold_mask,
                dtype=torch.bool,
            )

            maximum_mask.scatter_(
                -1,
                top_max,
                True,
            )

            hard_mask = threshold_mask & maximum_mask

        else:
            hard_mask = threshold_mask

        top_min = probabilities.topk(
            self.min_bins,
            dim=-1,
        ).indices

        minimum_mask = torch.zeros_like(
            threshold_mask,
            dtype=torch.bool,
        )

        minimum_mask.scatter_(
            -1,
            top_min,
            True,
        )

        return hard_mask | minimum_mask


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, hidden):

        logits = self.lte_router(hidden)
        probabilities = torch.sigmoid(logits)

        hard_mask = self._hard_bin_mask(
            probabilities
        )

        hard_gate = hard_mask.to(
            probabilities.dtype
        )

        if self.straight_through:
            bin_gate = (
                hard_gate
                + probabilities
                - probabilities.detach()
            )
        else:
            bin_gate = probabilities

        channel_gate = bin_gate.repeat_interleave(
            self.bin_size,
            dim=-1,
        )

        projected = self.linear(hidden)

        if projected.size(-1) != self.intermediate_size * 2:
            raise ValueError(
                "Mamba in_proj must output exactly "
                "2 × intermediate_size."
            )

        hidden_branch, gate_branch = projected.split(
            self.intermediate_size,
            dim=-1,
        )

        hidden_branch = hidden_branch * channel_gate
        gate_branch = gate_branch * channel_gate

        projected = torch.cat(
            [hidden_branch, gate_branch],
            dim=-1,
        )

        self._current_channel_gate = channel_gate

        soft_fraction = probabilities.mean()

        efficiency_loss = (
            soft_fraction
            - self.target_active_fraction
        ).pow(2)

        token_count = (
            hard_mask.numel()
            // self.num_bins
        )

        self._loss_records.append(
            efficiency_loss
        )

        self._hard_records.append(
            hard_gate.detach().sum()
        )

        self._soft_records.append(
            probabilities.detach().sum()
        )

        self._token_count += token_count

        return projected


    # ========================================================
    # OUTPUT GATE
    # ========================================================

    def current_channel_gate(self):
        return self._current_channel_gate


    def clear_current_gate(self):
        self._current_channel_gate = None


    # ========================================================
    # METRICS
    # ========================================================

    def reset_metrics(self):

        self._loss_records = []
        self._hard_records = []
        self._soft_records = []
        self._token_count = 0
        self._current_channel_gate = None


    def collect_metrics(self):

        device = self.linear.weight.device

        if not self._loss_records:
            zero = torch.tensor(0.0, device=device)

            return {
                "efficiency_loss": zero,
                "hard_active_bins": zero.detach(),
                "soft_active_bins": zero.detach(),
                "token_decisions": 0,
                "average_active_bins": zero.detach(),
                "active_fraction": zero.detach(),
                "soft_active_fraction": zero.detach(),
            }

        efficiency_loss = torch.stack(
            self._loss_records
        ).mean()

        hard_active_bins = torch.stack(
            self._hard_records
        ).sum()

        soft_active_bins = torch.stack(
            self._soft_records
        ).sum()

        token_decisions = self._token_count

        average_active_bins = (
            hard_active_bins
            / token_decisions
        )

        active_fraction = (
            average_active_bins
            / self.num_bins
        )

        soft_active_fraction = (
            soft_active_bins
            / (
                token_decisions
                * self.num_bins
            )
        )

        self._loss_records = []
        self._hard_records = []
        self._soft_records = []
        self._token_count = 0

        return {
            "efficiency_loss": efficiency_loss,
            "hard_active_bins": hard_active_bins,
            "soft_active_bins": soft_active_bins,
            "token_decisions": token_decisions,
            "average_active_bins": average_active_bins,
            "active_fraction": active_fraction,
            "soft_active_fraction": soft_active_fraction,
        }


# ============================================================
# LTE OUTPUT PROJECTION
# ============================================================

class LTEOutputProjection(nn.Module):

    def __init__(self, linear, input_projection):
        super().__init__()

        self.linear = linear

        object.__setattr__(
            self,
            "_input_projection_ref",
            weakref.ref(input_projection),
        )


    @property
    def weight(self):
        return self.linear.weight


    @property
    def bias(self):
        return self.linear.bias


    def forward(self, hidden):

        input_projection = (
            self._input_projection_ref()
        )

        if input_projection is None:
            raise RuntimeError(
                "LTE input projection reference was lost."
            )

        channel_gate = (
            input_projection.current_channel_gate()
        )

        if channel_gate is None:
            raise RuntimeError(
                "LTE output projection did not receive "
                "a channel gate."
            )

        hidden = hidden * channel_gate
        output = self.linear(hidden)

        input_projection.clear_current_gate()

        return output


# ============================================================
# LTE DYNAMIC MAMBA
# ============================================================

class LTEAdaptiveWidthCausalLM(DynamicMoRCausalLM):

    def __init__(
        self,
        config,
        recursion_cfg,
        state_cfg,
        routing_cfg,
        lte_cfg,
    ):
        super().__init__(
            config,
            recursion_cfg,
            state_cfg,
            routing_cfg,
        )

        self.lte_cfg = lte_cfg

        if not lte_cfg.get("enabled", True):
            raise ValueError(
                "S06 requires lte.enabled=true."
            )

        if lte_cfg.get("apply_to", "shared_middle") != "shared_middle":
            raise ValueError(
                "Current S06 implementation applies LTE "
                "to shared_middle layers."
            )

        self._lte_layers = []

        self._apply_lte()


    # ========================================================
    # APPLY LTE
    # ========================================================

    def _apply_lte(self):

        shared_start = self.cfg.input_layers
        shared_end = (
            shared_start
            + self.cfg.shared_middle_layers
        )

        for layer_index in range(
            shared_start,
            shared_end,
        ):

            mixer = (
                self.model
                .backbone
                .layers[layer_index]
                .mixer
            )

            if not isinstance(
                mixer.in_proj,
                nn.Linear,
            ):
                raise TypeError(
                    f"Layer {layer_index} in_proj "
                    "must be nn.Linear before LTE wrapping."
                )

            if not isinstance(
                mixer.out_proj,
                nn.Linear,
            ):
                raise TypeError(
                    f"Layer {layer_index} out_proj "
                    "must be nn.Linear before LTE wrapping."
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
                (
                    layer_index,
                    lte_input,
                )
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

        layer_metrics = {}

        efficiency_losses = []
        hard_active_bins = []
        soft_active_bins = []

        total_token_decisions = 0

        for layer_index, lte_layer in self._lte_layers:

            metrics = (
                lte_layer.collect_metrics()
            )

            efficiency_losses.append(
                metrics["efficiency_loss"]
            )

            hard_active_bins.append(
                metrics["hard_active_bins"]
            )

            soft_active_bins.append(
                metrics["soft_active_bins"]
            )

            total_token_decisions += (
                metrics["token_decisions"]
            )

            layer_metrics[
                str(layer_index)
            ] = {
                "average_active_bins": (
                    metrics[
                        "average_active_bins"
                    ].detach()
                ),

                "active_fraction": (
                    metrics[
                        "active_fraction"
                    ].detach()
                ),

                "soft_active_fraction": (
                    metrics[
                        "soft_active_fraction"
                    ].detach()
                ),

                "token_decisions": (
                    metrics[
                        "token_decisions"
                    ]
                ),
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
            output["loss"] = (
                output["loss"]
                + lte_aux_loss
            )

        output.update(
            {
                "lte_efficiency_loss": (
                    lte_efficiency_loss
                ),

                "lte_aux_loss": (
                    lte_aux_loss
                ),

                "lte_average_active_bins": (
                    average_active_bins.detach()
                ),

                "lte_active_fraction": (
                    active_fraction.detach()
                ),

                "lte_soft_active_fraction": (
                    soft_active_fraction.detach()
                ),

                "lte_active_bin_count": (
                    hard_active_bins.detach()
                ),

                "lte_token_decisions": (
                    total_token_decisions
                ),

                "lte_layer_stats": (
                    layer_metrics
                ),
            }
        )

        return output


    # ========================================================
    # LOAD S04 PARENT
    # ========================================================

    def load_parent_state_dict(
        self,
        parent_state,
    ):

        target_state = self.state_dict()
        mapped_state = {}

        target_parameters = dict(
            self.named_parameters(
                remove_duplicate=False
            )
        )

        transferred_ids = set()
        transferred_count = 0

        for target_key, target_tensor in target_state.items():

            if ".in_proj.lte_router." in target_key:
                continue

            source_key = target_key

            source_key = source_key.replace(
                ".mixer.in_proj.linear.",
                ".mixer.in_proj.",
            )

            source_key = source_key.replace(
                ".mixer.out_proj.linear.",
                ".mixer.out_proj.",
            )

            if source_key not in parent_state:
                continue

            source_tensor = (
                parent_state[source_key]
            )

            if (
                source_tensor.shape
                != target_tensor.shape
            ):
                raise ValueError(
                    f"Parent/child tensor mismatch: "
                    f"{source_key} -> {target_key}: "
                    f"{source_tensor.shape} != "
                    f"{target_tensor.shape}"
                )

            mapped_state[target_key] = (
                source_tensor.clone()
            )

            parameter = (
                target_parameters.get(
                    target_key
                )
            )

            if parameter is not None:

                parameter_id = id(parameter)

                if (
                    parameter_id
                    not in transferred_ids
                ):
                    transferred_ids.add(
                        parameter_id
                    )

                    transferred_count += (
                        parameter.numel()
                    )

        missing, unexpected = (
            self.load_state_dict(
                mapped_state,
                strict=False,
            )
        )

        invalid_missing = [
            key
            for key in missing
            if ".in_proj.lte_router." not in key
        ]

        if invalid_missing:
            raise RuntimeError(
                "Unexpected missing parent parameters:\n"
                + "\n".join(
                    invalid_missing
                )
            )

        total_parameters = sum(
            p.numel()
            for p in self.parameters()
        )

        lte_router_parameters = sum(
            lte_layer.lte_router.weight.numel()
            + lte_layer.lte_router.bias.numel()
            for _, lte_layer
            in self._lte_layers
        )

        return {
            "strategy": "copy_parent_init_lte",

            "transferred_parameter_count": (
                transferred_count
            ),

            "new_lte_router_parameters": (
                lte_router_parameters
            ),

            "total_parameter_count": (
                total_parameters
            ),

            "transferred_parameter_fraction": (
                transferred_count
                / total_parameters
            ),

            "missing_lte_keys": [
                key
                for key in missing
                if ".in_proj.lte_router." in key
            ],

            "unexpected_keys": list(
                unexpected
            ),
        }


    # ========================================================
    # WIDTH-SENSITIVE PARAMETER COUNT
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


    # ========================================================
    # ACTIVE PARAMETER REPORT
    # ========================================================

    def active_parameter_report(
        self,
        active_fraction,
    ):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        width_sensitive = (
            self.width_sensitive_parameter_count()
        )

        fixed = (
            total
            - width_sensitive
        )

        theoretical_active = (
            fixed
            + width_sensitive
            * active_fraction
        )

        return {
            "stored_parameters": total,

            "width_sensitive_parameters": (
                width_sensitive
            ),

            "fixed_parameters": (
                fixed
            ),

            "active_width_fraction": (
                active_fraction
            ),

            "theoretical_active_parameters": (
                theoretical_active
            ),

            "theoretical_active_parameter_fraction": (
                theoretical_active
                / total
            ),

            "stored_parameter_reduction": 0.0,

            "note": (
                "LTE changes token-dependent active width, "
                "not stored model parameter count."
            ),
        }


    # ========================================================
    # PARAMETER REPORT
    # ========================================================

    def parameter_report(self):

        report = super().parameter_report()

        lte_router_parameters = sum(
            lte_layer.lte_router.weight.numel()
            + lte_layer.lte_router.bias.numel()
            for _, lte_layer
            in self._lte_layers
        )

        report.update(
            {
                "lte_router_parameters": (
                    lte_router_parameters
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
                "architecture": (
                    "lte_dynamic_mamba"
                ),

                "lte_adaptive_width": True,

                "lte_apply_to": (
                    "shared_middle"
                ),

                "lte_layer_indices": [
                    layer_index
                    for layer_index, _
                    in self._lte_layers
                ],

                "lte_num_bins": (
                    self.lte_cfg[
                        "num_bins"
                    ]
                ),

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