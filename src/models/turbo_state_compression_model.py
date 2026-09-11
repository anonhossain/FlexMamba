from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache, LinearAttentionLayer

from src.models.grouped_lte_model import (
    GroupedLTEModelConfig,
    GroupedLTECausalLM,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TurboStateCompressionModelConfig(GroupedLTEModelConfig):
    pass


# ============================================================
# NORMALIZED FAST WALSH-HADAMARD TRANSFORM
# ============================================================

def fwht(x):

    n = x.size(-1)

    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(
            f"Hadamard rotation requires power-of-two dimension, received {n}."
        )

    y = x.float()
    original_shape = y.shape

    h = 1

    while h < n:

        y = y.reshape(
            *original_shape[:-1],
            n // (2 * h),
            2 * h,
        )

        left = y[..., :h].clone()
        right = y[..., h:2 * h].clone()

        y = torch.cat(
            [
                left + right,
                left - right,
            ],
            dim=-1,
        )

        y = y.reshape(original_shape)

        h *= 2

    return y / math.sqrt(n)


# ============================================================
# TURBOQUANT-INSPIRED STATE QUANTIZER
# ============================================================

class TurboStateQuantizer:

    def __init__(
        self,
        channels,
        state_dim,
        cfg,
        seed,
    ):

        self.channels = channels
        self.state_dim = state_dim

        self.bits = cfg.get("bits", 8)
        self.symmetric = cfg.get("symmetric", True)
        self.per_channel = cfg.get("per_channel", True)

        self.percentile = cfg.get(
            "calibration_percentile",
            99.9,
        )

        self.max_samples = cfg.get(
            "max_samples_per_channel",
            16384,
        )

        self.rotation = cfg.get(
            "rotation",
            "randomized_hadamard",
        )

        if not self.symmetric:
            raise ValueError(
                "Current S08 implementation requires symmetric quantization."
            )

        if self.bits < 2 or self.bits > 8:
            raise ValueError(
                "Current S08 prototype supports 2-8 bit quantization."
            )

        if self.rotation != "randomized_hadamard":
            raise ValueError(
                "Current S08 requires rotation='randomized_hadamard'."
            )

        if (
            self.state_dim <= 0
            or (self.state_dim & (self.state_dim - 1)) != 0
        ):
            raise ValueError(
                "d_state must be a power of two for the Hadamard rotation."
            )

        self.qmax = (2 ** (self.bits - 1)) - 1

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(seed)

        signs = torch.randint(
            0,
            2,
            (state_dim,),
            generator=generator,
            dtype=torch.int8,
        )

        self.signs = (
            signs * 2 - 1
        )

        self.scale = None
        self.calibrated = False

        self.mode = "float"
        self._samples = []


    # ========================================================
    # ROTATION
    # ========================================================

    def rotate(self, state):

        signs = self.signs.to(
            device=state.device,
            dtype=state.dtype,
        )

        return fwht(
            state * signs
        )


    def inverse_rotate(self, state):

        signs = self.signs.to(
            device=state.device,
            dtype=state.dtype,
        )

        return (
            fwht(state)
            * signs
        )


    # ========================================================
    # CALIBRATION
    # ========================================================

    def reset_calibration(self):

        self.scale = None
        self.calibrated = False
        self._samples = []


    def observe(self, state):

        rotated = self.rotate(
            state.detach()
        )

        if self.per_channel:

            values = (
                rotated
                .abs()
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
                .abs()
                .reshape(1, -1)
                .cpu()
            )

        self._samples.append(
            values
        )

        current = sum(
            sample.size(1)
            for sample in self._samples
        )

        if current > self.max_samples:

            merged = torch.cat(
                self._samples,
                dim=1,
            )

            indices = torch.linspace(
                0,
                merged.size(1) - 1,
                self.max_samples,
            ).long()

            self._samples = [
                merged[:, indices]
            ]


    def finalize_calibration(self):

        if not self._samples:
            raise RuntimeError(
                "No recurrent-state samples were collected for calibration."
            )

        samples = torch.cat(
            self._samples,
            dim=1,
        )

        q = self.percentile / 100.0

        clip = torch.quantile(
            samples.float(),
            q,
            dim=1,
        )

        scale = (
            clip
            / self.qmax
        ).clamp_min(1e-8)

        if self.per_channel:

            self.scale = scale.reshape(
                1,
                self.channels,
                1,
            )

        else:

            self.scale = scale.reshape(
                1,
                1,
                1,
            )

        self.calibrated = True
        self._samples = []


    # ========================================================
    # QUANTIZE / DEQUANTIZE
    # ========================================================

    def quantize(self, state):

        if not self.calibrated:
            raise RuntimeError(
                "State quantizer has not been calibrated."
            )

        rotated = self.rotate(state)

        scale = self.scale.to(
            device=state.device,
            dtype=rotated.dtype,
        )

        quantized = torch.round(
            rotated / scale
        )

        quantized = quantized.clamp(
            -self.qmax,
            self.qmax,
        )

        # INT8 container. At 8 bits this is also physical storage.
        return quantized.to(torch.int8)


    def dequantize(
        self,
        quantized,
        dtype,
    ):

        scale = self.scale.to(
            device=quantized.device,
            dtype=torch.float32,
        )

        rotated = (
            quantized.float()
            * scale
        )

        restored = self.inverse_rotate(
            rotated
        )

        return restored.to(dtype)


    # ========================================================
    # SERIALIZATION
    # ========================================================

    def export_state(self):

        return {
            "bits": self.bits,
            "channels": self.channels,
            "state_dim": self.state_dim,
            "per_channel": self.per_channel,
            "percentile": self.percentile,

            "signs": self.signs.cpu(),

            "scale": (
                self.scale.cpu()
                if self.scale is not None
                else None
            ),

            "calibrated": self.calibrated,
        }


    def load_state(self, state):

        self.signs = state[
            "signs"
        ].cpu()

        self.scale = (
            state["scale"].cpu()
            if state["scale"] is not None
            else None
        )

        self.calibrated = state[
            "calibrated"
        ]


# ============================================================
# QUANTIZED RECURRENT-STATE STORAGE
# ============================================================

class QuantizedRecurrentStateStore:

    def __init__(self, quantizer):

        self.quantizer = quantizer

        self.float_states = {}
        self.quantized_states = {}
        self.original_dtypes = {}


    def __getitem__(self, index):

        if index in self.quantized_states:

            return self.quantizer.dequantize(
                self.quantized_states[index],
                self.original_dtypes[index],
            )

        return self.float_states.get(
            index,
            None,
        )


    def __setitem__(self, index, value):

        if value is None:

            self.float_states[index] = None

            self.quantized_states.pop(
                index,
                None,
            )

            return

        self.original_dtypes[index] = (
            value.dtype
        )

        if self.quantizer.mode == "calibrate":

            self.quantizer.observe(value)

            self.float_states[index] = (
                value.detach().clone()
            )

            self.quantized_states.pop(
                index,
                None,
            )

            return

        if self.quantizer.mode == "quantized":

            self.quantized_states[index] = (
                self.quantizer.quantize(
                    value.detach()
                )
            )

            self.float_states.pop(
                index,
                None,
            )

            return

        self.float_states[index] = (
            value.detach().clone()
        )

        self.quantized_states.pop(
            index,
            None,
        )


    def clear(self):

        self.float_states = {}
        self.quantized_states = {}
        self.original_dtypes = {}


# ============================================================
# MAMBA CACHE LAYER WITH QUANTIZED RECURRENT STATE
# ============================================================

class TurboQuantLinearAttentionLayer(
    LinearAttentionLayer
):

    def __init__(
        self,
        quantizer,
        number_of_states=1,
    ):

        super().__init__(
            number_of_states=number_of_states
        )

        self.state_quantizer = quantizer

        self.recurrent_states = (
            QuantizedRecurrentStateStore(
                quantizer
            )
        )


    def lazy_initialization(
        self,
        conv_states=None,
        recurrent_states=None,
        state_idx=0,
        conv_kernel_size=None,
    ):

        if conv_states is not None:

            super().lazy_initialization(
                conv_states=conv_states,
                recurrent_states=None,
                state_idx=state_idx,
                conv_kernel_size=conv_kernel_size,
            )

        if recurrent_states is not None:

            if self.device is None:

                self.dtype = recurrent_states.dtype
                self.device = recurrent_states.device

            self.is_recurrent_states_initialized[
                state_idx
            ] = True


    def update_recurrent_state(
        self,
        recurrent_states,
        state_idx=0,
        **kwargs,
    ):

        if not self.is_recurrent_states_initialized[
            state_idx
        ]:

            self.lazy_initialization(
                recurrent_states=recurrent_states,
                state_idx=state_idx,
            )

        self.recurrent_states[
            state_idx
        ] = recurrent_states

        return self.recurrent_states[
            state_idx
        ]


    def reset(self):

        for index in range(
            self.number_of_states
        ):

            if self.is_conv_states_initialized[
                index
            ]:

                self.conv_states[
                    index
                ].zero_()

            self.has_previous_state[
                index
            ] = False

            self.is_recurrent_states_initialized[
                index
            ] = False

        self.recurrent_states.clear()


# ============================================================
# TURBOQUANT-INSPIRED GROUPED + LTE MAMBA
# ============================================================

class TurboStateCompressionCausalLM(
    GroupedLTECausalLM
):

    def __init__(
        self,
        config,
        recursion_cfg,
        state_cfg,
        routing_cfg,
        grouped_cfg,
        lte_cfg,
        state_quant_cfg,
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

        self.state_quant_cfg = (
            state_quant_cfg
        )

        if not state_quant_cfg.get(
            "enabled",
            True,
        ):
            raise ValueError(
                "S08 requires state_quantization.enabled=true."
            )

        if (
            state_quant_cfg.get(
                "target"
            )
            != "recurrent_ssm_state"
        ):
            raise ValueError(
                "S08 currently quantizes recurrent_ssm_state only."
            )

        self.shared_layer_indices = list(
            range(
                self.cfg.input_layers,
                self.cfg.input_layers
                + self.cfg.shared_middle_layers,
            )
        )

        intermediate = (
            self.cfg.d_model
            * self.cfg.expand
        )

        self._state_quantizers = {}

        for recursion_index in range(
            self.max_recursions
        ):

            for layer_index in (
                self.shared_layer_indices
            ):

                quantizer_seed = (
                    seed
                    + recursion_index * 100
                    + layer_index
                )

                self._state_quantizers[
                    (
                        recursion_index,
                        layer_index,
                    )
                ] = TurboStateQuantizer(
                    channels=intermediate,
                    state_dim=self.cfg.d_state,
                    cfg=state_quant_cfg,
                    seed=quantizer_seed,
                )


    # ========================================================
    # QUANTIZATION MODE
    # ========================================================

    def set_state_quantization_mode(
        self,
        mode,
    ):

        if mode not in {
            "float",
            "calibrate",
            "quantized",
        }:
            raise ValueError(
                f"Unknown state quantization mode: {mode}"
            )

        if mode == "quantized":

            missing = [
                key
                for key, quantizer
                in self._state_quantizers.items()
                if not quantizer.calibrated
            ]

            if missing:
                raise RuntimeError(
                    "Cannot enable quantized state before calibration."
                )

        for quantizer in (
            self._state_quantizers.values()
        ):
            quantizer.mode = mode


    def reset_state_calibration(self):

        for quantizer in (
            self._state_quantizers.values()
        ):

            quantizer.reset_calibration()

        self.set_state_quantization_mode(
            "calibrate"
        )


    def finalize_state_calibration(self):

        for quantizer in (
            self._state_quantizers.values()
        ):

            quantizer.finalize_calibration()

        self.set_state_quantization_mode(
            "quantized"
        )


    # ========================================================
    # QUANTIZED CACHE
    # ========================================================

    def _new_turbo_cache(
        self,
        recursion_index,
    ):

        try:

            cache = DynamicCache(
                config=self.model.config
            )

        except TypeError:

            cache = DynamicCache()

        if not hasattr(cache, "layers"):

            raise RuntimeError(
                "DynamicCache does not expose cache layers."
            )

        for layer_index in (
            self.shared_layer_indices
        ):

            if layer_index >= len(
                cache.layers
            ):
                raise RuntimeError(
                    f"Cache has no layer {layer_index}."
                )

            old_layer = cache.layers[
                layer_index
            ]

            number_of_states = getattr(
                old_layer,
                "number_of_states",
                1,
            )

            cache.layers[
                layer_index
            ] = (
                TurboQuantLinearAttentionLayer(
                    quantizer=(
                        self._state_quantizers[
                            (
                                recursion_index,
                                layer_index,
                            )
                        ]
                    ),
                    number_of_states=number_of_states,
                )
            )

        return cache


    def new_state_bank(self):

        return [
            self._new_turbo_cache(
                recursion_index
            )
            for recursion_index in range(
                self.max_recursions
            )
        ]


    # ========================================================
    # LTE METRICS
    # ========================================================

    def _reset_lte_metrics(self):

        for _, lte_layer in (
            self._lte_layers
        ):
            lte_layer.reset_metrics()


    def _collect_lte_metrics(self):

        efficiency_losses = []

        hard_active_bins = []
        soft_active_bins = []

        layer_metrics = {}

        total_token_decisions = 0

        for layer_index, lte_layer in (
            self._lte_layers
        ):

            metrics = (
                lte_layer.collect_metrics()
            )

            efficiency_losses.append(
                metrics[
                    "efficiency_loss"
                ]
            )

            hard_active_bins.append(
                metrics[
                    "hard_active_bins"
                ]
            )

            soft_active_bins.append(
                metrics[
                    "soft_active_bins"
                ]
            )

            total_token_decisions += (
                metrics[
                    "token_decisions"
                ]
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

        lte_efficiency_loss = (
            torch.stack(
                efficiency_losses
            ).mean()
        )

        hard_active_bins = (
            torch.stack(
                hard_active_bins
            ).sum()
        )

        soft_active_bins = (
            torch.stack(
                soft_active_bins
            ).sum()
        )

        average_active_bins = (
            hard_active_bins
            / total_token_decisions
        )

        active_fraction = (
            average_active_bins
            / self.lte_cfg[
                "num_bins"
            ]
        )

        soft_active_fraction = (
            soft_active_bins
            / (
                total_token_decisions
                * self.lte_cfg[
                    "num_bins"
                ]
            )
        )

        return {
            "lte_efficiency_loss": (
                lte_efficiency_loss
            ),

            "lte_average_active_bins": (
                average_active_bins
            ),

            "lte_active_fraction": (
                active_fraction
            ),

            "lte_soft_active_fraction": (
                soft_active_fraction
            ),

            "lte_active_bin_count": (
                hard_active_bins
            ),

            "lte_token_decisions": (
                total_token_decisions
            ),

            "lte_layer_stats": (
                layer_metrics
            ),
        }


    # ========================================================
    # CHUNKED FORWARD WITH PERSISTENT RECURRENT STATE
    # ========================================================

    def forward_chunk(
        self,
        input_ids,
        labels=None,
        state_bank=None,
    ):

        self._reset_lte_metrics()

        backbone = self.model.backbone
        layers = backbone.layers

        hidden = backbone.embeddings(
            input_ids
        )

        layer_index = 0

        # ----------------------------------------------------
        # UNIQUE INPUT
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
        # DEPTH ROUTER
        # ----------------------------------------------------

        routing = self._route(hidden)

        router_probs = routing[
            "probs"
        ]

        depths = routing[
            "depths"
        ]

        depth_values = torch.arange(
            self.min_recursions,
            self.max_recursions + 1,
            device=hidden.device,
            dtype=torch.long,
        )

        if state_bank is None:
            state_bank = (
                self.new_state_bank()
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

            cache = state_bank[
                recursion_index
            ]

            candidate = hidden

            for j in range(
                self.cfg.shared_middle_layers
            ):

                candidate = self._run_layer(
                    layers[
                        shared_start + j
                    ],
                    candidate,
                    cache,
                )

            hard_active = (
                depths
                >= recursion_number
            ).to(hidden.dtype)

            valid_depths = (
                depth_values
                >= recursion_number
            )

            soft_active = (
                router_probs[
                    ...,
                    valid_depths
                ].sum(dim=-1)
            )

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
        # UNIQUE OUTPUT
        # ----------------------------------------------------

        output_start = (
            shared_start
            + self.cfg.shared_middle_layers
        )

        for j in range(
            self.cfg.output_layers
        ):

            hidden = self._run_layer(
                layers[
                    output_start + j
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

        router_aux_loss = (
            self.routing_cfg.get(
                "balance_loss_coeff",
                0.0,
            )
            * routing[
                "balance_loss"
            ]
            + self.routing_cfg.get(
                "z_loss_coeff",
                0.0,
            )
            * routing[
                "z_loss"
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
            )

        # ----------------------------------------------------
        # LTE LOSS
        # ----------------------------------------------------

        lte = (
            self._collect_lte_metrics()
        )

        lte_aux_loss = (
            self.lte_cfg.get(
                "efficiency_loss_coeff",
                0.0,
            )
            * lte[
                "lte_efficiency_loss"
            ]
        )

        if total_loss is not None:

            total_loss = (
                total_loss
                + lte_aux_loss
            )

        return {
            "logits": logits,

            "loss": total_loss,
            "lm_loss": lm_loss,

            "avg_depth": (
                depths.float().mean()
            ),

            "avg_recursion_depth": (
                depths.float().mean()
            ),

            "routing_depths": (
                depths.detach()
            ),

            "routing_entropy": (
                routing[
                    "entropy"
                ].detach()
            ),

            "router_balance_loss": (
                routing[
                    "balance_loss"
                ]
            ),

            "router_z_loss": (
                routing[
                    "z_loss"
                ]
            ),

            "router_aux_loss": (
                router_aux_loss
            ),

            "lte_efficiency_loss": (
                lte[
                    "lte_efficiency_loss"
                ]
            ),

            "lte_aux_loss": (
                lte_aux_loss
            ),

            "lte_average_active_bins": (
                lte[
                    "lte_average_active_bins"
                ].detach()
            ),

            "lte_active_fraction": (
                lte[
                    "lte_active_fraction"
                ].detach()
            ),

            "lte_soft_active_fraction": (
                lte[
                    "lte_soft_active_fraction"
                ].detach()
            ),

            "lte_active_bin_count": (
                lte[
                    "lte_active_bin_count"
                ].detach()
            ),

            "lte_token_decisions": (
                lte[
                    "lte_token_decisions"
                ]
            ),

            "lte_layer_stats": (
                lte[
                    "lte_layer_stats"
                ]
            ),

            "state_bank": state_bank,
        }


    def forward(
        self,
        input_ids,
        labels=None,
    ):

        output = self.forward_chunk(
            input_ids,
            labels=labels,
            state_bank=None,
        )

        output.pop(
            "state_bank",
            None,
        )

        return output


    # ========================================================
    # QUANTIZATION SERIALIZATION
    # ========================================================

    def quantization_state_dict(self):

        state = {}

        for (
            recursion_index,
            layer_index,
        ), quantizer in (
            self._state_quantizers.items()
        ):

            key = (
                f"r{recursion_index}"
                f"_l{layer_index}"
            )

            state[key] = (
                quantizer.export_state()
            )

        return state


    def load_quantization_state_dict(
        self,
        state,
    ):

        for (
            recursion_index,
            layer_index,
        ), quantizer in (
            self._state_quantizers.items()
        ):

            key = (
                f"r{recursion_index}"
                f"_l{layer_index}"
            )

            if key not in state:
                raise KeyError(
                    f"Missing quantizer state: {key}"
                )

            quantizer.load_state(
                state[key]
            )

        self.set_state_quantization_mode(
            "quantized"
        )


    # ========================================================
    # COMPRESSION REPORT
    # ========================================================

    def state_compression_report(
        self,
        batch_size,
        dtype=torch.float32,
    ):

        bytes_per_float = (
            torch.tensor(
                [],
                dtype=dtype,
            ).element_size()
        )

        intermediate = (
            self.cfg.d_model
            * self.cfg.expand
        )

        banks = (
            self.max_recursions
        )

        shared_layers = (
            self.cfg.shared_middle_layers
        )

        recurrent_elements = (
            batch_size
            * banks
            * shared_layers
            * intermediate
            * self.cfg.d_state
        )

        conv_elements = (
            batch_size
            * banks
            * shared_layers
            * intermediate
            * self.cfg.d_conv
        )

        full_recurrent_bytes = (
            recurrent_elements
            * bytes_per_float
        )

        conv_bytes = (
            conv_elements
            * bytes_per_float
        )

        # INT8 physical tensor container.
        quantized_recurrent_bytes = (
            recurrent_elements
        )

        scale_count = (
            banks
            * shared_layers
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
            scale_count
            * 4
        )

        rotation_metadata_bytes = (
            banks
            * shared_layers
            * self.cfg.d_state
        )

        full_total_bytes = (
            full_recurrent_bytes
            + conv_bytes
        )

        compressed_total_bytes = (
            quantized_recurrent_bytes
            + conv_bytes
            + scale_bytes
            + rotation_metadata_bytes
        )

        return {
            "target": (
                "recurrent_ssm_state"
            ),

            "bits": (
                self.state_quant_cfg[
                    "bits"
                ]
            ),

            "original_dtype": (
                str(dtype)
            ),

            "recurrent_state_elements": (
                recurrent_elements
            ),

            "conv_state_elements": (
                conv_elements
            ),

            "full_precision_recurrent_mb": (
                full_recurrent_bytes
                / (1024 ** 2)
            ),

            "quantized_recurrent_payload_mb": (
                quantized_recurrent_bytes
                / (1024 ** 2)
            ),

            "scale_metadata_mb": (
                scale_bytes
                / (1024 ** 2)
            ),

            "rotation_metadata_mb": (
                rotation_metadata_bytes
                / (1024 ** 2)
            ),

            "conv_state_mb": (
                conv_bytes
                / (1024 ** 2)
            ),

            "full_state_total_mb": (
                full_total_bytes
                / (1024 ** 2)
            ),

            "compressed_state_total_mb": (
                compressed_total_bytes
                / (1024 ** 2)
            ),

            "recurrent_payload_compression_ratio": (
                full_recurrent_bytes
                / quantized_recurrent_bytes
            ),

            "total_state_compression_ratio": (
                full_total_bytes
                / compressed_total_bytes
            ),

            "total_state_memory_reduction": (
                1.0
                - compressed_total_bytes
                / full_total_bytes
            ),

            "conv_state_quantized": False,

            "quantized_storage_realized": True,

            "note": (
                "Only recurrent SSM state is quantized. "
                "Convolution state remains full precision."
            ),
        }


    # ========================================================
    # REPORTS
    # ========================================================

    def parameter_report(self):

        report = super().parameter_report()

        report.update(
            {
                "state_quantization_parameters": 0,
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
                "architecture": (
                    "turboquant_grouped_lte_dynamic_mamba"
                ),

                "turboquant_state_compression": True,

                "state_quantization_target": (
                    "recurrent_ssm_state"
                ),

                "state_quantization_bits": (
                    self.state_quant_cfg[
                        "bits"
                    ]
                ),

                "state_quantization_rotation": (
                    self.state_quant_cfg[
                        "rotation"
                    ]
                ),

                "state_quantization_per_channel": (
                    self.state_quant_cfg[
                        "per_channel"
                    ]
                ),

                "conv_state_quantized": False,

                "training_required": False,
            }
        )

        return report