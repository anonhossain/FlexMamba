from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.final_joint_model import (
    FinalJointModelConfig,
    FinalJointCausalLM,
)

from src.models.quantization_aware_tuning_model import (
    QATStateQuantizer,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ScalingStudyModelConfig(FinalJointModelConfig):
    pass


# ============================================================
# CALIBRATABLE QAT QUANTIZER
# ============================================================

class ScalingStateQuantizer(QATStateQuantizer):

    def __init__(self, channels, state_dim, state_quant_cfg, qat_cfg, seed):
        super().__init__(
            channels,
            state_dim,
            state_quant_cfg,
            qat_cfg,
            seed,
        )

        self.calibration_percentile = state_quant_cfg.get(
            "calibration_percentile",
            99.9,
        )

        self.max_samples = state_quant_cfg.get(
            "max_samples_per_channel",
            16384,
        )

        self._calibration_samples = []


    def start_calibration(self):

        self._calibration_samples = []
        self.ready.fill_(False)


    @torch.no_grad()
    def observe(self, state):

        rotated = self.rotate(
            state.detach()
        ).abs()

        if self.per_channel:

            values = (
                rotated
                .movedim(1, 0)
                .reshape(
                    self.channels,
                    -1,
                )
                .cpu()
            )

        else:

            values = (
                rotated
                .reshape(1, -1)
                .cpu()
            )

        self._calibration_samples.append(
            values
        )

        count = sum(
            sample.size(1)
            for sample in self._calibration_samples
        )

        if count > self.max_samples:

            merged = torch.cat(
                self._calibration_samples,
                dim=1,
            )

            indices = torch.linspace(
                0,
                merged.size(1) - 1,
                self.max_samples,
            ).long()

            self._calibration_samples = [
                merged[:, indices]
            ]


    @torch.no_grad()
    def finish_calibration(self):

        if not self._calibration_samples:
            raise RuntimeError(
                "No state samples collected for scaling calibration."
            )

        samples = torch.cat(
            self._calibration_samples,
            dim=1,
        ).float()

        clip = torch.quantile(
            samples,
            self.calibration_percentile / 100.0,
            dim=1,
        )

        scale = (
            clip / self.qmax
        ).clamp_min(1e-8)

        if self.per_channel:
            scale = scale.reshape(
                1,
                self.channels,
                1,
            )
        else:
            scale = scale.reshape(
                1,
                1,
                1,
            )

        self.scale.copy_(
            scale.to(
                self.scale.device
            )
        )

        self.ready.fill_(True)
        self._calibration_samples = []


    def quantize_dequantize(
        self,
        state,
        mode,
    ):

        if mode == "calibrate":
            self.observe(state)
            return state

        return super().quantize_dequantize(
            state,
            mode,
        )


# ============================================================
# SCALING FLEXMAMBA
# ============================================================

class ScalingStudyCausalLM(FinalJointCausalLM):

    def __init__(
        self,
        config,
        recursion_cfg,
        state_cfg,
        routing_cfg,
        grouped_cfg,
        lte_cfg,
        state_quant_cfg,
        qat_cfg,
        seed=42,
    ):

        super().__init__(
            config,
            recursion_cfg,
            state_cfg,
            routing_cfg,
            grouped_cfg,
            lte_cfg,
            state_quant_cfg,
            qat_cfg,
            seed=seed,
        )

        intermediate = (
            self.cfg.d_model
            * self.cfg.expand
        )

        quantizers = nn.ModuleDict()

        for recursion_index in range(
            self.max_recursions
        ):

            for shared_index in range(
                self.cfg.shared_middle_layers
            ):

                layer_index = (
                    self.cfg.input_layers
                    + shared_index
                )

                key = self._quantizer_key(
                    recursion_index,
                    layer_index,
                )

                quantizers[key] = (
                    ScalingStateQuantizer(
                        channels=intermediate,
                        state_dim=self.cfg.d_state,
                        state_quant_cfg=state_quant_cfg,
                        qat_cfg=qat_cfg,
                        seed=(
                            seed
                            + recursion_index * 100
                            + layer_index
                        ),
                    )
                )

        self.state_quantizers = quantizers


    # ========================================================
    # CALIBRATION
    # ========================================================

    def set_state_mode(self, mode):

        if mode == "calibrate":
            self.state_mode = mode
            return

        super().set_state_mode(mode)


    def start_state_calibration(self):

        for quantizer in self.state_quantizers.values():
            quantizer.start_calibration()

        self.set_state_mode(
            "calibrate"
        )


    def finish_state_calibration(self):

        for quantizer in self.state_quantizers.values():
            quantizer.finish_calibration()

        self.set_state_mode(
            "fake_quant"
        )


    # ========================================================
    # SCALE PARAMETER ACCOUNTING
    # ========================================================

    def scale_parameter_report(self):

        stored = sum(
            p.numel()
            for p in self.parameters()
        )

        grouping = (
            self.grouping_report()
        )

        physical_dense = grouping[
            "dense_combined_equivalent_parameters"
        ]

        extra_unrolled = 0

        shared_start = (
            self.cfg.input_layers
        )

        shared_end = (
            shared_start
            + self.cfg.shared_middle_layers
        )

        for layer_index in range(
            shared_start,
            shared_end,
        ):

            layer = (
                self.model
                .backbone
                .layers[layer_index]
            )

            layer_stored = sum(
                p.numel()
                for p in layer.parameters()
            )

            grouped_savings = 0

            for item in self._grouped_projections:

                if item["layer"] != layer_index:
                    continue

                module = item["module"]

                grouped_savings += (
                    module.dense_parameter_count()
                    - module.grouped_parameter_count()
                )

            dense_layer = (
                layer_stored
                + grouped_savings
            )

            extra_unrolled += (
                dense_layer
                * (
                    self.max_recursions
                    - 1
                )
            )

        unrolled_dense = (
            physical_dense
            + extra_unrolled
        )

        return {
            "stored_parameters": stored,

            "physical_dense_equivalent_parameters": (
                physical_dense
            ),

            "unrolled_dense_equivalent_parameters": (
                unrolled_dense
            ),

            "recursive_parameter_savings": (
                unrolled_dense
                - physical_dense
            ),

            "grouped_parameter_savings": (
                grouping[
                    "parameter_savings"
                ]
            ),

            "stored_vs_unrolled_fraction": (
                stored
                / unrolled_dense
            ),

            "stored_reduction_vs_unrolled": (
                1.0
                - stored
                / unrolled_dense
            ),
        }


    def architecture_report(self):

        report = super().architecture_report()

        report.update(
            {
                "architecture": (
                    "flexmamba_scaling"
                ),

                "scaling_study": True,

                "scaling_axis": (
                    "d_model"
                ),

                "from_scratch_scaling": True,
            }
        )

        return report