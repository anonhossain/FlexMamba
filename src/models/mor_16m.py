# src/models/mor_16m.py

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoRModelConfig:
    vocab_size: int = 49152
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 768

    n_shared_layers: int = 3
    num_recursions: int = 3

    max_seq_len: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1.0e-5

    router_type: str = "token_choice"
    router_hidden_mult: int = 1
    router_balance_coeff: float = 0.01
    router_z_loss_coeff: float = 0.001

    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")

        if self.num_recursions < 1:
            raise ValueError("num_recursions must be >= 1.")

        if self.router_type != "token_choice":
            raise ValueError("MoR 16M currently supports router_type='token_choice'.")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def physical_layers(self):
        return 2 + self.n_shared_layers

    @property
    def max_effective_layers(self):
        return 2 + self.n_shared_layers * self.num_recursions


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return x_norm.to(x.dtype) * self.weight


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached",
            emb.cos()[None, None, :, :],
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            emb.sin()[None, None, :, :],
            persistent=False,
        )

    def forward(self, q, k):
        seq_len = q.shape[-2]

        cos = self.cos_cached[:, :, :seq_len].to(
            dtype=q.dtype,
            device=q.device,
        )

        sin = self.sin_cached[:, :, :seq_len].to(
            dtype=q.dtype,
            device=q.device,
        )

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        return q, k


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_repeat = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(
            cfg.d_model,
            cfg.n_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            cfg.d_model,
            cfg.n_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            cfg.d_model,
            cfg.n_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            cfg.n_heads * self.head_dim,
            cfg.d_model,
            bias=False,
        )

        self.rope = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_seq_len,
            cfg.rope_theta,
        )

    def forward(self, x, active_mask=None):
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(
            bsz, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            bsz, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            bsz, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / (self.head_dim ** 0.5)

        causal = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        allowed = causal[None, None].expand(
            bsz,
            1,
            seq_len,
            seq_len,
        )

        if active_mask is not None:
            key_active = active_mask[:, None, None, :]
            allowed = allowed & key_active

            eye = torch.eye(
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )[None, None]

            inactive_q = (~active_mask)[:, None, :, None]
            allowed = allowed | (inactive_q & eye)

        scores = scores.masked_fill(
            ~allowed,
            torch.finfo(scores.dtype).min,
        )

        attn = F.softmax(
            scores.float(),
            dim=-1,
        ).to(scores.dtype)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(
            bsz,
            seq_len,
            self.n_heads * self.head_dim,
        )

        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gate_proj = nn.Linear(
            cfg.d_model,
            cfg.d_ff,
            bias=False,
        )

        self.up_proj = nn.Linear(
            cfg.d_model,
            cfg.d_ff,
            bias=False,
        )

        self.down_proj = nn.Linear(
            cfg.d_ff,
            cfg.d_model,
            bias=False,
        )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x))
            * self.up_proj(x)
        )


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn_norm = RMSNorm(
            cfg.d_model,
            cfg.rms_norm_eps,
        )

        self.attn = GroupedQueryAttention(cfg)

        self.ffn_norm = RMSNorm(
            cfg.d_model,
            cfg.rms_norm_eps,
        )

        self.ffn = SwiGLU(cfg)

    def forward(self, x, active_mask=None):
        h = x + self.attn(
            self.attn_norm(x),
            active_mask=active_mask,
        )

        return h + self.ffn(
            self.ffn_norm(h)
        )


class TokenChoiceRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        hidden_dim = (
            cfg.d_model
            * cfg.router_hidden_mult
        )

        if cfg.router_hidden_mult == 1:
            self.net = nn.Linear(
                cfg.d_model,
                cfg.num_recursions,
                bias=False,
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(
                    cfg.d_model,
                    hidden_dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.Linear(
                    hidden_dim,
                    cfg.num_recursions,
                    bias=False,
                ),
            )

    def forward(self, hidden):
        logits = self.net(hidden)
        probs = F.softmax(logits, dim=-1)

        depth_index = probs.argmax(dim=-1)
        depths = depth_index + 1

        return logits, probs, depths


class MoRCausalLM(nn.Module):
    def __init__(self, config: MoRModelConfig):
        super().__init__()

        self.cfg = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        self.first_block = TransformerBlock(config)

        self.shared_core = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_shared_layers)
            ]
        )

        self.last_block = TransformerBlock(config)
        self.router = TokenChoiceRouter(config)

        self.final_norm = RMSNorm(
            config.d_model,
            config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def router_losses(
        self,
        router_logits,
        router_probs,
        depths,
    ):
        nr = self.cfg.num_recursions

        selected = F.one_hot(
            depths - 1,
            num_classes=nr,
        ).float()

        fraction = selected.mean(
            dim=(0, 1)
        ).detach()

        mean_prob = router_probs.mean(
            dim=(0, 1)
        )

        balance_loss = nr * torch.sum(
            fraction * mean_prob
        )

        z_loss = torch.logsumexp(
            router_logits.float(),
            dim=-1,
        ).pow(2).mean()

        return balance_loss, z_loss

    def forward(
        self,
        input_ids,
        labels=None,
    ):
        hidden = self.embed_tokens(input_ids)

        # Unique input block
        first_hidden = self.first_block(hidden)

        router_logits, router_probs, depths = (
            self.router(first_hidden)
        )

        current = first_hidden
        final_recursive = torch.zeros_like(first_hidden)
        finished = torch.zeros_like(
            depths,
            dtype=torch.bool,
        )

        # Dynamic recursive shared core
        for recursion in range(
            1,
            self.cfg.num_recursions + 1,
        ):
            active = depths >= recursion
            core_hidden = current

            for block in self.shared_core:
                block_output = block(
                    core_hidden,
                    active_mask=active,
                )

                core_hidden = torch.where(
                    active.unsqueeze(-1),
                    block_output,
                    core_hidden,
                )

            gate = router_probs[
                ...,
                recursion - 1,
            ].unsqueeze(-1)

            routed = gate * core_hidden

            exits_now = depths == recursion
            exit_hidden = routed + first_hidden

            final_recursive = torch.where(
                exits_now.unsqueeze(-1),
                exit_hidden,
                final_recursive,
            )

            finished = finished | exits_now

            continue_mask = depths > recursion

            current = torch.where(
                continue_mask.unsqueeze(-1),
                routed,
                current,
            )

        final_recursive = torch.where(
            finished.unsqueeze(-1),
            final_recursive,
            current,
        )

        # Unique output block
        hidden = self.last_block(final_recursive)
        hidden = self.final_norm(hidden)

        logits = self.lm_head(hidden)

        avg_depth = depths.float().mean()

        output = {
            "logits": logits,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "depths": depths,
            "avg_depth": avg_depth.detach(),
            "avg_recursion_depth": avg_depth.detach(),
        }

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                labels.reshape(-1),
                ignore_index=-100,
            )

            balance_loss, z_loss = self.router_losses(
                router_logits,
                router_probs,
                depths,
            )

            loss = (
                lm_loss
                + self.cfg.router_balance_coeff * balance_loss
                + self.cfg.router_z_loss_coeff * z_loss
            )

            output.update(
                {
                    "loss": loss,
                    "lm_loss": lm_loss.detach(),
                    "balance_loss": balance_loss.detach(),
                    "z_loss": z_loss.detach(),
                }
            )

        return output

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

        embedding = self.embed_tokens.weight.numel()

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embedding,
            "non_embedding_parameters": total - embedding,
        }

    def architecture_report(self) -> Dict:
        return {
            "architecture": "mor",
            "physical_layers": self.cfg.physical_layers,
            "max_effective_layers": self.cfg.max_effective_layers,
            "n_shared_layers": self.cfg.n_shared_layers,
            "num_recursions": self.cfg.num_recursions,
            "router_type": self.cfg.router_type,
            "recursive": True,
            "dynamic_routing": True,
        }