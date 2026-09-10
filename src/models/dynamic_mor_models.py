from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MambaConfig as HFMambaConfig
from transformers import MambaForCausalLM
from transformers.cache_utils import DynamicCache


# ============================================================
# CONFIG
# ============================================================

@dataclass
class DynamicMoRModelConfig:
    vocab_size: int
    d_model: int

    input_layers: int
    shared_middle_layers: int
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


# ============================================================
# DYNAMIC MoR MAMBA
# ============================================================

class DynamicMoRCausalLM(nn.Module):

    def __init__(self, config, recursion_cfg, state_cfg, routing_cfg):
        super().__init__()

        self.cfg = config
        self.recursion_cfg = recursion_cfg
        self.state_cfg = state_cfg
        self.routing_cfg = routing_cfg

        self.min_recursions = recursion_cfg["min_recursions"]
        self.max_recursions = recursion_cfg["max_recursions"]
        self.num_depth_choices = self.max_recursions - self.min_recursions + 1

        if routing_cfg["router_type"] != "token_choice":
            raise ValueError("S04 requires token_choice routing.")

        if state_cfg.get("share_state_across_recursions", False):
            raise ValueError("S04 requires independent recursion-wise states.")

        if self.min_recursions < 1:
            raise ValueError("min_recursions must be >= 1.")

        if self.max_recursions < self.min_recursions:
            raise ValueError("max_recursions must be >= min_recursions.")

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

        router_hidden = config.d_model * routing_cfg.get("hidden_multiplier", 1)

        self.router = nn.Sequential(
            nn.Linear(config.d_model, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, self.num_depth_choices),
        )

        self._initialize_router()


    # ========================================================
    # ROUTER INITIALIZATION
    # ========================================================

    def _initialize_router(self):

        if not self.routing_cfg.get("initialize_to_parent_depth", True):
            return

        final_layer = self.router[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        with torch.no_grad():
            final_layer.bias[-1] = self.routing_cfg.get(
                "max_depth_bias",
                0.01,
            )


    # ========================================================
    # RECURSION-WISE STATE
    # ========================================================

    def _new_cache(self, hidden):

        try:
            cache = DynamicCache(config=self.model.config)
        except TypeError:
            cache = DynamicCache()

        return cache


    def _run_layer(self, layer, hidden, cache=None):

        output = layer(
            hidden,
            cache_params=cache,
        )

        if isinstance(output, (tuple, list)):
            return output[0]

        return output


    # ========================================================
    # TOKEN-CHOICE ROUTER
    # ========================================================

    def _route(self, hidden):

        router_logits = self.router(hidden)

        temperature = self.routing_cfg.get(
            "temperature",
            1.0,
        )

        if temperature <= 0:
            raise ValueError("Router temperature must be > 0.")

        router_probs = F.softmax(
            router_logits / temperature,
            dim=-1,
        )

        depth_values = torch.arange(
            self.min_recursions,
            self.max_recursions + 1,
            device=hidden.device,
            dtype=torch.long,
        )

        selected_index = router_probs.argmax(dim=-1)
        selected_depths = depth_values[selected_index]

        # ----------------------------------------------------
        # ROUTER BALANCE LOSS
        # ----------------------------------------------------

        mean_probs = router_probs.mean(dim=(0, 1))

        uniform_target = torch.full_like(
            mean_probs,
            1.0 / self.num_depth_choices,
        )

        balance_loss = (
            (mean_probs - uniform_target) ** 2
        ).mean()

        # ----------------------------------------------------
        # ROUTER Z-LOSS
        # ----------------------------------------------------

        z_loss = (
            torch.logsumexp(router_logits, dim=-1) ** 2
        ).mean()

        # ----------------------------------------------------
        # ROUTER ENTROPY
        # ----------------------------------------------------

        entropy = -(
            router_probs
            * torch.log(router_probs.clamp_min(1e-9))
        ).sum(dim=-1).mean()

        return {
            "logits": router_logits,
            "probs": router_probs,
            "depths": selected_depths,
            "balance_loss": balance_loss,
            "z_loss": z_loss,
            "entropy": entropy,
        }


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, input_ids, labels=None):

        backbone = self.model.backbone
        layers = backbone.layers

        hidden = backbone.embeddings(input_ids)

        layer_index = 0

        # ----------------------------------------------------
        # UNIQUE INPUT LAYER(S)
        # ----------------------------------------------------

        for _ in range(self.cfg.input_layers):

            hidden = self._run_layer(
                layers[layer_index],
                hidden,
            )

            layer_index += 1

        shared_start = layer_index

        # ----------------------------------------------------
        # ROUTER
        # ----------------------------------------------------

        routing = self._route(hidden)

        router_probs = routing["probs"]
        depths = routing["depths"]

        depth_values = torch.arange(
            self.min_recursions,
            self.max_recursions + 1,
            device=hidden.device,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # RECURSION-WISE STATE BANK
        # ----------------------------------------------------

        state_bank = [
            self._new_cache(hidden)
            for _ in range(self.max_recursions)
        ]

        # ----------------------------------------------------
        # DYNAMIC RECURSIVE MIDDLE
        # ----------------------------------------------------

        for recursion_index in range(self.max_recursions):

            recursion_number = recursion_index + 1
            cache = state_bank[recursion_index]

            candidate = hidden

            # Shared Mamba blocks.
            for j in range(self.cfg.shared_middle_layers):

                candidate = self._run_layer(
                    layers[shared_start + j],
                    candidate,
                    cache,
                )

            # ------------------------------------------------
            # HARD TOKEN ACTIVATION
            # ------------------------------------------------

            hard_active = (
                depths >= recursion_number
            ).to(hidden.dtype)

            # ------------------------------------------------
            # SOFT TOKEN ACTIVATION
            # ------------------------------------------------

            valid_depths = depth_values >= recursion_number

            soft_active = router_probs[
                ...,
                valid_depths
            ].sum(dim=-1)

            # ------------------------------------------------
            # STRAIGHT-THROUGH ROUTING
            # ------------------------------------------------

            if self.routing_cfg.get("hard_routing", True):

                gate = hard_active

                if self.routing_cfg.get("straight_through", True):
                    gate = (
                        hard_active
                        + soft_active
                        - soft_active.detach()
                    )

            else:
                gate = soft_active

            # Token-wise update.
            hidden = hidden + gate.unsqueeze(-1) * (
                candidate - hidden
            )

        # ----------------------------------------------------
        # UNIQUE OUTPUT LAYER(S)
        # ----------------------------------------------------

        output_start = (
            shared_start
            + self.cfg.shared_middle_layers
        )

        for j in range(self.cfg.output_layers):

            hidden = self._run_layer(
                layers[output_start + j],
                hidden,
            )

        # ----------------------------------------------------
        # FINAL NORM + LM HEAD
        # ----------------------------------------------------

        hidden = backbone.norm_f(hidden)
        logits = self.model.lm_head(hidden)

        # ----------------------------------------------------
        # LOSSES
        # ----------------------------------------------------

        lm_loss = None
        total_loss = None

        balance_coeff = self.routing_cfg.get(
            "balance_loss_coeff",
            0.0,
        )

        z_coeff = self.routing_cfg.get(
            "z_loss_coeff",
            0.0,
        )

        router_aux_loss = (
            balance_coeff * routing["balance_loss"]
            + z_coeff * routing["z_loss"]
        )

        if labels is not None:

            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            total_loss = lm_loss + router_aux_loss

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

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
        }


    # ========================================================
    # LOAD S03 PARENT
    # ========================================================

    def load_parent_state_dict(self, parent_state):

        target_state = self.state_dict()
        mapped_state = {}

        target_parameters = dict(
            self.named_parameters(remove_duplicate=False)
        )

        transferred_ids = set()
        transferred_count = 0

        for key, target_tensor in target_state.items():

            # Router is new in S04.
            if key.startswith("router."):
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

            parameter = target_parameters.get(key)

            if parameter is not None:

                parameter_id = id(parameter)

                if parameter_id not in transferred_ids:
                    transferred_ids.add(parameter_id)
                    transferred_count += parameter.numel()

        missing, unexpected = self.load_state_dict(
            mapped_state,
            strict=False,
        )

        invalid_missing = [
            key
            for key in missing
            if not key.startswith("router.")
        ]

        if invalid_missing:
            raise RuntimeError(
                "Missing parent parameters:\n"
                + "\n".join(invalid_missing)
            )

        total_parameters = sum(
            p.numel()
            for p in self.parameters()
        )

        router_parameters = sum(
            p.numel()
            for p in self.router.parameters()
        )

        return {
            "strategy": "copy_parent_init_router",
            "transferred_parameter_count": transferred_count,
            "new_router_parameter_count": router_parameters,
            "total_parameter_count": total_parameters,
            "transferred_parameter_fraction": (
                transferred_count / total_parameters
            ),
            "missing_router_keys": [
                key
                for key in missing
                if key.startswith("router.")
            ],
            "unexpected_keys": list(unexpected),
        }


    # ========================================================
    # STATE REPORT
    # ========================================================

    def state_report(
        self,
        batch_size=1,
        dtype=torch.float32,
    ):

        intermediate = (
            self.cfg.d_model
            * self.cfg.expand
        )

        bytes_per_element = torch.tensor(
            [],
            dtype=dtype,
        ).element_size()

        # State associated with the recursive shared blocks.
        used_elements = (
            batch_size
            * self.max_recursions
            * self.cfg.shared_middle_layers
            * intermediate
            * (self.cfg.d_state + self.cfg.d_conv)
        )

        # DynamicCache may allocate for all physical blocks.
        allocated_elements = (
            batch_size
            * self.max_recursions
            * self.cfg.physical_layers
            * intermediate
            * (self.cfg.d_state + self.cfg.d_conv)
        )

        used_memory_mb = (
            used_elements
            * bytes_per_element
            / (1024 ** 2)
        )

        allocated_memory_mb = (
            allocated_elements
            * bytes_per_element
            / (1024 ** 2)
        )

        return {
            "mode": "recursion_wise",
            "state_banks": self.max_recursions,
            "share_block_weights": True,
            "share_state_across_recursions": False,
            "reset_state_between_sequences": True,
            "used_shared_state_memory_mb": used_memory_mb,
            "allocated_cache_memory_mb": allocated_memory_mb,
        }


    # ========================================================
    # PARAMETER REPORT
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

        router = sum(
            p.numel()
            for p in self.router.parameters()
        )

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embedding,
            "router_parameters": router,
            "non_embedding_parameters": total - embedding,
        }


    # ========================================================
    # ARCHITECTURE REPORT
    # ========================================================

    def architecture_report(self) -> Dict:

        min_effective_layers = (
            self.cfg.input_layers
            + self.cfg.shared_middle_layers
            * self.min_recursions
            + self.cfg.output_layers
        )

        max_effective_layers = (
            self.cfg.input_layers
            + self.cfg.shared_middle_layers
            * self.max_recursions
            + self.cfg.output_layers
        )

        return {
            "architecture": "dynamic_mor_mamba",

            "physical_layers": self.cfg.physical_layers,

            "min_effective_layers": min_effective_layers,
            "max_effective_layers": max_effective_layers,

            "input_layers": self.cfg.input_layers,
            "shared_middle_layers": self.cfg.shared_middle_layers,
            "output_layers": self.cfg.output_layers,

            "min_recursions": self.min_recursions,
            "max_recursions": self.max_recursions,

            "router_type": "token_choice",

            "recursive": True,
            "dynamic_routing": True,
            "recursion_wise_state": True,

            # The current implementation evaluates all recursive
            # cycles before masking inactive token updates.
            "dense_routing_fallback": True,
        }