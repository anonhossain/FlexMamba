from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.grouped_lte_model import (
    GroupedLTEModelConfig,
    GroupedLTECausalLM,
)

from src.models.turbo_state_compression_model import fwht


# ============================================================
# CONFIG
# ============================================================

@dataclass
class QuantizationAwareTuningModelConfig(GroupedLTEModelConfig):
    pass


# ============================================================
# QAT RECURRENT-STATE QUANTIZER
# ============================================================

class QATStateQuantizer(nn.Module):

    def __init__(self, channels, state_dim, state_quant_cfg, qat_cfg, seed):
        super().__init__()

        self.channels = channels
        self.state_dim = state_dim

        self.bits = state_quant_cfg["bits"]
        self.per_channel = state_quant_cfg.get("per_channel", True)
        self.symmetric = state_quant_cfg.get("symmetric", True)
        self.rotation = state_quant_cfg.get("rotation", "randomized_hadamard")

        self.use_ste = qat_cfg.get("straight_through_estimator", True)

        if not self.symmetric:
            raise ValueError("S09 currently requires symmetric state quantization.")

        if self.rotation != "randomized_hadamard":
            raise ValueError("S09 currently requires randomized_hadamard rotation.")

        if self.bits < 2 or self.bits > 8:
            raise ValueError("S09 currently supports 2-8 bit fake quantization.")

        if state_dim <= 0 or (state_dim & (state_dim - 1)) != 0:
            raise ValueError("d_state must be a power of two for Hadamard rotation.")

        self.qmax = (2 ** (self.bits - 1)) - 1

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        signs = torch.randint(
            0,
            2,
            (state_dim,),
            generator=generator,
            dtype=torch.int8,
        )

        signs = signs * 2 - 1

        scale_shape = (1, channels, 1) if self.per_channel else (1, 1, 1)

        self.register_buffer("signs", signs)
        self.register_buffer("scale", torch.ones(scale_shape, dtype=torch.float32))
        self.register_buffer("ready", torch.tensor(False, dtype=torch.bool))

        self.reset_statistics()


    # ========================================================
    # PARENT QUANTIZER STATE
    # ========================================================

    @torch.no_grad()
    def load_external_state(self, state):

        if state.get("bits") != self.bits:
            raise ValueError(
                f"Parent quantizer is {state.get('bits')} bit, "
                f"but S09 requests {self.bits} bit."
            )

        signs = state["signs"]
        scale = state["scale"]

        if scale is None:
            raise ValueError("Parent S08 quantizer has no calibrated scale.")

        if signs.shape != self.signs.shape:
            raise ValueError(
                f"Sign shape mismatch: {signs.shape} != {self.signs.shape}"
            )

        if scale.shape != self.scale.shape:
            raise ValueError(
                f"Scale shape mismatch: {scale.shape} != {self.scale.shape}"
            )

        self.signs.copy_(signs.to(self.signs.device))
        self.scale.copy_(scale.to(self.scale.device))
        self.ready.fill_(True)


    def export_state(self):

        return {
            "bits": self.bits,
            "channels": self.channels,
            "state_dim": self.state_dim,
            "per_channel": self.per_channel,
            "signs": self.signs.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "calibrated": bool(self.ready.item()),
        }


    # ========================================================
    # ROTATION
    # ========================================================

    def rotate(self, state):

        signs = self.signs.to(
            device=state.device,
            dtype=state.dtype,
        )

        return fwht(state * signs)


    def inverse_rotate(self, rotated):

        signs = self.signs.to(
            device=rotated.device,
            dtype=rotated.dtype,
        )

        return fwht(rotated) * signs


    # ========================================================
    # QAT QUANTIZE / DEQUANTIZE
    # ========================================================

    def quantize_dequantize(self, state, mode):

        if mode == "float":
            return state

        if not bool(self.ready.item()):
            raise RuntimeError("QAT state quantizer is not initialized from S08.")

        rotated = self.rotate(state)

        scale = self.scale.to(
            device=state.device,
            dtype=rotated.dtype,
        )

        normalized = rotated / scale

        clipped = normalized.clamp(
            -self.qmax,
            self.qmax,
        )

        rounded = torch.round(clipped)

        dequantized_rotated = rounded * scale

        restored = self.inverse_rotate(
            dequantized_rotated
        ).to(state.dtype)

        self._record_statistics(
            original=state,
            restored=restored,
            normalized=normalized,
        )

        if mode == "fake_quant":

            if not self.use_ste:
                return restored

            # Forward  = quantized/dequantized value.
            # Backward = identity gradient.
            return state + (
                restored - state
            ).detach()

        if mode == "quantized":
            return restored

        raise ValueError(
            f"Unknown recurrent-state mode: {mode}"
        )


    # ========================================================
    # METRICS
    # ========================================================

    def reset_statistics(self):

        self._squared_error = 0.0
        self._absolute_error = 0.0
        self._values = 0
        self._clipped = 0


    @torch.no_grad()
    def _record_statistics(
        self,
        original,
        restored,
        normalized,
    ):

        error = (
            restored.detach().float()
            - original.detach().float()
        )

        self._squared_error += (
            error.pow(2).sum().item()
        )

        self._absolute_error += (
            error.abs().sum().item()
        )

        self._values += error.numel()

        self._clipped += (
            normalized.detach().abs()
            > self.qmax
        ).sum().item()


    def statistics(self):

        if self._values == 0:
            return {
                "mse": 0.0,
                "mae": 0.0,
                "clip_fraction": 0.0,
                "values": 0,
            }

        return {
            "mse": self._squared_error / self._values,
            "mae": self._absolute_error / self._values,
            "clip_fraction": self._clipped / self._values,
            "values": self._values,
        }


# ============================================================
# QAT JOINT MODEL
# ============================================================

class QuantizationAwareTuningCausalLM(GroupedLTECausalLM):

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
        )

        self.state_quant_cfg = state_quant_cfg
        self.qat_cfg = qat_cfg
        self.state_mode = "fake_quant"

        if not qat_cfg.get("enabled", True):
            raise ValueError("S09 requires quantization.enabled=true.")

        if qat_cfg.get("target") != "recurrent_ssm_state":
            raise ValueError("S09 currently targets recurrent_ssm_state only.")

        if not qat_cfg.get("fake_quantization", True):
            raise ValueError("S09 requires fake_quantization=true.")

        if qat_cfg["state_bits"] != state_quant_cfg["bits"]:
            raise ValueError(
                "quantization.state_bits must match state_quantization.bits."
            )

        intermediate = self.cfg.d_model * self.cfg.expand

        self.state_quantizers = nn.ModuleDict()

        for recursion_index in range(self.max_recursions):

            for shared_index in range(self.cfg.shared_middle_layers):

                layer_index = self.cfg.input_layers + shared_index
                key = self._quantizer_key(recursion_index, layer_index)

                quantizer_seed = (
                    seed
                    + recursion_index * 100
                    + layer_index
                )

                self.state_quantizers[key] = QATStateQuantizer(
                    channels=intermediate,
                    state_dim=self.cfg.d_state,
                    state_quant_cfg=state_quant_cfg,
                    qat_cfg=qat_cfg,
                    seed=quantizer_seed,
                )


    # ========================================================
    # QUANTIZER HELPERS
    # ========================================================

    @staticmethod
    def _quantizer_key(recursion_index, layer_index):
        return f"r{recursion_index}_l{layer_index}"


    def set_state_mode(self, mode):

        if mode not in {
            "float",
            "fake_quant",
            "quantized",
        }:
            raise ValueError(
                f"Unknown state mode: {mode}"
            )

        self.state_mode = mode


    def reset_quantization_statistics(self):

        for quantizer in self.state_quantizers.values():
            quantizer.reset_statistics()


    def quantization_statistics(self):

        total_squared_error = 0.0
        total_absolute_error = 0.0
        total_clipped = 0.0
        total_values = 0

        per_quantizer = {}

        for key, quantizer in self.state_quantizers.items():

            stats = quantizer.statistics()

            per_quantizer[key] = stats

            total_squared_error += stats["mse"] * stats["values"]
            total_absolute_error += stats["mae"] * stats["values"]
            total_clipped += stats["clip_fraction"] * stats["values"]
            total_values += stats["values"]

        if total_values == 0:
            return {
                "mse": 0.0,
                "mae": 0.0,
                "clip_fraction": 0.0,
                "values": 0,
                "per_quantizer": per_quantizer,
            }

        return {
            "mse": total_squared_error / total_values,
            "mae": total_absolute_error / total_values,
            "clip_fraction": total_clipped / total_values,
            "values": total_values,
            "per_quantizer": per_quantizer,
        }


    def quantization_state_dict(self):

        return {
            key: quantizer.export_state()
            for key, quantizer
            in self.state_quantizers.items()
        }


    # ========================================================
    # LOAD S08 PARENT
    # ========================================================

    def load_parent_state_dict(
        self,
        parent_state,
        parent_quantization_state,
    ):

        missing, unexpected = self.load_state_dict(
            parent_state,
            strict=False,
        )

        invalid_missing = [
            key
            for key in missing
            if not key.startswith(
                "state_quantizers."
            )
        ]

        if invalid_missing:
            raise RuntimeError(
                "Unexpected missing parent parameters/buffers:\n"
                + "\n".join(invalid_missing)
            )

        if unexpected:
            raise RuntimeError(
                "Unexpected S08 parent keys:\n"
                + "\n".join(unexpected)
            )

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

                if key not in parent_quantization_state:
                    raise KeyError(
                        f"Missing S08 quantizer state: {key}"
                    )

                self.state_quantizers[
                    key
                ].load_external_state(
                    parent_quantization_state[key]
                )

        total_parameters = sum(
            p.numel()
            for p in self.parameters()
        )

        return {
            "strategy": "copy_s08_weights_reuse_quantizer_state",
            "transferred_parameter_count": total_parameters,
            "new_trainable_parameters": 0,
            "total_parameter_count": total_parameters,
            "transferred_parameter_fraction": 1.0,
            "quantizer_count": len(self.state_quantizers),
            "quantizer_scales_reused": True,
        }


    # ========================================================
    # LTE METRICS
    # ========================================================

    def _reset_lte_metrics(self):

        for _, lte_layer in self._lte_layers:
            lte_layer.reset_metrics()


    def _collect_lte_metrics(self):

        losses = []
        hard_bins = []
        soft_bins = []

        layer_stats = {}
        token_decisions = 0

        for layer_index, lte_layer in self._lte_layers:

            metrics = lte_layer.collect_metrics()

            losses.append(
                metrics["efficiency_loss"]
            )

            hard_bins.append(
                metrics["hard_active_bins"]
            )

            soft_bins.append(
                metrics["soft_active_bins"]
            )

            token_decisions += (
                metrics["token_decisions"]
            )

            layer_stats[
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

        efficiency_loss = torch.stack(
            losses
        ).mean()

        hard_bins = torch.stack(
            hard_bins
        ).sum()

        soft_bins = torch.stack(
            soft_bins
        ).sum()

        average_active_bins = (
            hard_bins / token_decisions
        )

        active_fraction = (
            average_active_bins
            / self.lte_cfg["num_bins"]
        )

        soft_active_fraction = (
            soft_bins
            / (
                token_decisions
                * self.lte_cfg["num_bins"]
            )
        )

        return {
            "efficiency_loss": efficiency_loss,
            "hard_active_bins": hard_bins,
            "average_active_bins": average_active_bins,
            "active_fraction": active_fraction,
            "soft_active_fraction": soft_active_fraction,
            "token_decisions": token_decisions,
            "layer_stats": layer_stats,
        }


    # ========================================================
    # DIFFERENTIABLE QAT MAMBA SCAN
    # ========================================================

    def _qat_mamba_block(
        self,
        layer,
        hidden,
        quantizer,
    ):

        residual = hidden

        hidden = layer.norm(
            hidden.to(
                dtype=layer.norm.weight.dtype
            )
        )

        if layer.residual_in_fp32:
            residual = residual.float()

        mixer = layer.mixer

        batch_size, seq_len, _ = (
            hidden.shape
        )

        intermediate = (
            mixer.intermediate_size
        )

        # ----------------------------------------------------
        # MAMBA INPUT + LTE
        # ----------------------------------------------------

        projected = mixer.in_proj(
            hidden
        )

        if projected.size(-1) != (
            intermediate * 2
        ):
            raise ValueError(
                "Mamba in_proj output size mismatch."
            )

        hidden_b_c, gate = projected.split(
            intermediate,
            dim=-1,
        )

        # ----------------------------------------------------
        # CAUSAL DEPTHWISE CONVOLUTION
        # ----------------------------------------------------

        conv_input = hidden_b_c.transpose(
            1,
            2,
        )

        conv_output = F.conv1d(
            conv_input.to(
                mixer.conv1d.weight.dtype
            ),
            weight=mixer.conv1d.weight,
            bias=mixer.conv1d.bias,
            padding=mixer.conv_kernel_size - 1,
            groups=intermediate,
        )

        conv_output = conv_output[
            :,
            :,
            :seq_len,
        ]

        conv_output = mixer.act(
            conv_output
        )

        conv_output = conv_output.transpose(
            1,
            2,
        ).to(hidden.dtype)

        # ----------------------------------------------------
        # SELECTIVE PARAMETERS
        # ----------------------------------------------------

        projected_ssm = mixer.x_proj(
            conv_output
        )

        time_step, B, C = torch.split(
            projected_ssm,
            [
                mixer.time_step_rank,
                mixer.ssm_state_size,
                mixer.ssm_state_size,
            ],
            dim=-1,
        )

        time_step = F.linear(
            time_step,
            mixer.dt_proj.weight,
            bias=None,
        )

        A = -torch.exp(
            mixer.A_log.float()
        )

        D = mixer.D.float()

        dt_bias = (
            mixer.dt_proj.bias.float()
            if mixer.dt_proj.bias is not None
            else None
        )

        # ----------------------------------------------------
        # RECURRENT SCAN WITH FAKE-QUANTIZED STATE
        # ----------------------------------------------------

        state = torch.zeros(
            batch_size,
            intermediate,
            mixer.ssm_state_size,
            device=hidden.device,
            dtype=torch.float32,
        )

        outputs = []

        for token_index in range(
            seq_len
        ):

            dt = time_step[
                :,
                token_index,
                :
            ].float()

            if dt_bias is not None:
                dt = dt + dt_bias

            dt = F.softplus(dt)

            B_t = B[
                :,
                token_index,
                :
            ].float()

            C_t = C[
                :,
                token_index,
                :
            ]

            u_t = conv_output[
                :,
                token_index,
                :
            ].float()

            dA = torch.exp(
                dt.unsqueeze(-1)
                * A.unsqueeze(0)
            )

            dB = (
                dt.unsqueeze(-1)
                * B_t.unsqueeze(1)
            )

            new_state = (
                state * dA
                + dB
                * u_t.unsqueeze(-1)
            )

            # Current token output uses the newly computed
            # full-precision state, matching storage quantization:
            # quantization affects the NEXT recurrent step.
            out = torch.matmul(
                new_state.to(
                    C_t.dtype
                ),
                C_t.unsqueeze(-1),
            ).squeeze(-1)

            out = (
                out
                + u_t.to(out.dtype)
                * D.to(out.dtype)
            )

            out = (
                out
                * F.silu(
                    gate[
                        :,
                        token_index,
                        :
                    ].to(out.dtype)
                )
            )

            outputs.append(
                out.to(hidden.dtype)
            )

            # This is the important S09 operation.
            state = quantizer.quantize_dequantize(
                new_state,
                mode=self.state_mode,
            )

        scan_output = torch.stack(
            outputs,
            dim=1,
        )

        contextualized = mixer.out_proj(
            scan_output
        )

        return residual + contextualized


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        input_ids,
        labels=None,
    ):

        self._reset_lte_metrics()
        self.reset_quantization_statistics()

        backbone = self.model.backbone
        layers = backbone.layers

        hidden = backbone.embeddings(
            input_ids
        )

        layer_index = 0

        # ----------------------------------------------------
        # UNIQUE INPUT LAYER
        # ----------------------------------------------------

        for _ in range(
            self.cfg.input_layers
        ):

            hidden = self._run_layer(
                layers[layer_index],
                hidden,
            )

            layer_index += 1

        shared_start = layer_index

        # ----------------------------------------------------
        # DYNAMIC DEPTH ROUTER
        # ----------------------------------------------------

        routing = self._route(
            hidden
        )

        router_probs = routing["probs"]
        depths = routing["depths"]

        depth_values = torch.arange(
            self.min_recursions,
            self.max_recursions + 1,
            device=hidden.device,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # RECURSIVE SHARED MIDDLE
        # ----------------------------------------------------

        for recursion_index in range(
            self.max_recursions
        ):

            recursion_number = (
                recursion_index + 1
            )

            candidate = hidden

            for shared_index in range(
                self.cfg.shared_middle_layers
            ):

                physical_layer_index = (
                    shared_start
                    + shared_index
                )

                quantizer_key = (
                    self._quantizer_key(
                        recursion_index,
                        physical_layer_index,
                    )
                )

                candidate = (
                    self._qat_mamba_block(
                        layers[
                            physical_layer_index
                        ],
                        candidate,
                        self.state_quantizers[
                            quantizer_key
                        ],
                    )
                )

            hard_active = (
                depths
                >= recursion_number
            ).to(hidden.dtype)

            valid_depths = (
                depth_values
                >= recursion_number
            )

            soft_active = router_probs[
                ...,
                valid_depths
            ].sum(dim=-1)

            if self.routing_cfg.get(
                "hard_routing",
                True,
            ):

                gate = hard_active

                if self.routing_cfg.get(
                    "straight_through",
                    True,
                ):

                    gate = (
                        hard_active
                        + soft_active
                        - soft_active.detach()
                    )

            else:
                gate = soft_active

            hidden = (
                hidden
                + gate.unsqueeze(-1)
                * (
                    candidate
                    - hidden
                )
            )

        # ----------------------------------------------------
        # UNIQUE OUTPUT LAYER
        # ----------------------------------------------------

        output_start = (
            shared_start
            + self.cfg.shared_middle_layers
        )

        for output_index in range(
            self.cfg.output_layers
        ):

            hidden = self._run_layer(
                layers[
                    output_start
                    + output_index
                ],
                hidden,
            )

        hidden = backbone.norm_f(
            hidden
        )

        logits = self.model.lm_head(
            hidden
        )

        # ----------------------------------------------------
        # LOSSES
        # ----------------------------------------------------

        balance_coeff = (
            self.routing_cfg.get(
                "balance_loss_coeff",
                0.0,
            )
        )

        z_coeff = (
            self.routing_cfg.get(
                "z_loss_coeff",
                0.0,
            )
        )

        router_aux_loss = (
            balance_coeff
            * routing[
                "balance_loss"
            ]
            + z_coeff
            * routing[
                "z_loss"
            ]
        )

        lte = self._collect_lte_metrics()

        lte_aux_loss = (
            self.lte_cfg.get(
                "efficiency_loss_coeff",
                0.0,
            )
            * lte[
                "efficiency_loss"
            ]
        )

        lm_loss = None
        total_loss = None

        if labels is not None:

            lm_loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
            )

            total_loss = (
                lm_loss
                + router_aux_loss
                + lte_aux_loss
            )

        quant_stats = (
            self.quantization_statistics()
        )

        return {
            "logits": logits,

            "loss": total_loss,
            "lm_loss": lm_loss,

            "avg_depth": depths.float().mean(),
            "avg_recursion_depth": depths.float().mean(),

            "routing_depths": depths.detach(),
            "routing_entropy": routing["entropy"].detach(),

            "router_balance_loss": routing["balance_loss"],
            "router_z_loss": routing["z_loss"],
            "router_aux_loss": router_aux_loss,

            "lte_efficiency_loss": lte["efficiency_loss"],
            "lte_aux_loss": lte_aux_loss,

            "lte_average_active_bins": (
                lte[
                    "average_active_bins"
                ].detach()
            ),

            "lte_active_fraction": (
                lte[
                    "active_fraction"
                ].detach()
            ),

            "lte_soft_active_fraction": (
                lte[
                    "soft_active_fraction"
                ].detach()
            ),

            "lte_active_bin_count": (
                lte[
                    "hard_active_bins"
                ].detach()
            ),

            "lte_token_decisions": (
                lte[
                    "token_decisions"
                ]
            ),

            "lte_layer_stats": (
                lte[
                    "layer_stats"
                ]
            ),

            "state_quantization_mse": (
                quant_stats["mse"]
            ),

            "state_quantization_mae": (
                quant_stats["mae"]
            ),

            "state_quantization_clip_fraction": (
                quant_stats[
                    "clip_fraction"
                ]
            ),
        }


    # ========================================================
    # STATE COMPRESSION REPORT
    # ========================================================

    def state_compression_report(
        self,
        batch_size,
        dtype=torch.float32,
    ):

        bytes_per_float = torch.tensor(
            [],
            dtype=dtype,
        ).element_size()

        intermediate = (
            self.cfg.d_model
            * self.cfg.expand
        )

        recurrent_elements = (
            batch_size
            * self.max_recursions
            * self.cfg.shared_middle_layers
            * intermediate
            * self.cfg.d_state
        )

        conv_elements = (
            batch_size
            * self.max_recursions
            * self.cfg.shared_middle_layers
            * intermediate
            * self.cfg.d_conv
        )

        full_recurrent_bytes = (
            recurrent_elements
            * bytes_per_float
        )

        quantized_recurrent_bytes = (
            recurrent_elements
            * self.state_quant_cfg["bits"]
            / 8
        )

        conv_bytes = (
            conv_elements
            * bytes_per_float
        )

        scale_count = (
            self.max_recursions
            * self.cfg.shared_middle_layers
            * (
                intermediate
                if self.state_quant_cfg.get(
                    "per_channel",
                    True,
                )
                else 1
            )
        )

        scale_bytes = (
            scale_count * 4
        )

        sign_bytes = (
            self.max_recursions
            * self.cfg.shared_middle_layers
            * self.cfg.d_state
        )

        full_total = (
            full_recurrent_bytes
            + conv_bytes
        )

        compressed_total = (
            quantized_recurrent_bytes
            + conv_bytes
            + scale_bytes
            + sign_bytes
        )

        return {
            "bits": self.state_quant_cfg["bits"],

            "full_precision_recurrent_mb": (
                full_recurrent_bytes
                / (1024 ** 2)
            ),

            "quantized_recurrent_mb": (
                quantized_recurrent_bytes
                / (1024 ** 2)
            ),

            "conv_state_mb": (
                conv_bytes
                / (1024 ** 2)
            ),

            "full_state_total_mb": (
                full_total
                / (1024 ** 2)
            ),

            "compressed_state_total_mb": (
                compressed_total
                / (1024 ** 2)
            ),

            "recurrent_compression_ratio": (
                full_recurrent_bytes
                / quantized_recurrent_bytes
            ),

            "total_state_compression_ratio": (
                full_total
                / compressed_total
            ),

            "total_state_memory_reduction": (
                1.0
                - compressed_total
                / full_total
            ),

            "physical_int8_storage_realized": (
                self.state_quant_cfg[
                    "bits"
                ] == 8
            ),

            "qat_forward_uses_fake_quantization": True,
        }


    # ========================================================
    # REPORTS
    # ========================================================

    def parameter_report(self):

        report = super().parameter_report()

        report.update(
            {
                "qat_trainable_parameters_added": 0,
                "state_quantizer_parameters": 0,
                "state_quantizer_count": len(
                    self.state_quantizers
                ),
                "state_quantization_bits": (
                    self.state_quant_cfg[
                        "bits"
                    ]
                ),
            }
        )

        return report


    def architecture_report(self):

        report = super().architecture_report()

        report.update(
            {
                "architecture": "qat_joint_mamba",

                "quantization_aware_tuning": True,

                "state_quantization_target": (
                    "recurrent_ssm_state"
                ),

                "state_quantization_bits": (
                    self.state_quant_cfg[
                        "bits"
                    ]
                ),

                "fake_quantization": True,

                "straight_through_estimator": (
                    self.qat_cfg.get(
                        "straight_through_estimator",
                        True,
                    )
                ),

                "quantizer_scales_frozen": (
                    self.qat_cfg.get(
                        "freeze_quantizer_scales",
                        True,
                    )
                ),

                "quantization_applied_per_token": True,

                "qat_is_turboquant_extension": True,
            }
        )

        return report