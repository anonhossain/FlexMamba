# will be deleted
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoRConfig:
    vocab_size: int = 49152
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 768
    n_shared_layers: int = 3
    num_recursions: int = 3
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    router_type: str = "token_choice"
    router_hidden_mult: int = 1
    router_balance_coeff: float = 0.01
    router_z_loss_coeff: float = 0.001
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.num_recursions >= 1

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (x_norm.to(x.dtype) * self.weight)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos_cached[:, :, :seq_len].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len].to(dtype=q.dtype, device=q.device)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_repeat = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        allowed = causal[None, None, :, :].expand(bsz, 1, seq_len, seq_len)

        # Recursion-wise masking: at deeper recursion steps, only active tokens
        # are allowed to contribute as keys. Inactive queries are later bypassed.
        if active_mask is not None:
            key_active = active_mask[:, None, None, :]
            allowed = allowed & key_active

            # Avoid all-masked rows for inactive queries; their outputs are discarded anyway.
            eye = torch.eye(seq_len, device=x.device, dtype=torch.bool)[None, None, :, :]
            inactive_q = (~active_mask)[:, None, :, None]
            allowed = allowed | (inactive_q & eye)

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), active_mask=active_mask)
        h = h + self.ffn(self.ffn_norm(h))
        return h


class TokenChoiceRouter(nn.Module):
    """Lightweight MoR token-choice router.

    Each token receives one recursion depth in {1, ..., Nr} from one routing
    decision made after the unique first layer.
    """

    def __init__(self, cfg: MoRConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.router_hidden_mult
        if cfg.router_hidden_mult == 1:
            self.net = nn.Linear(cfg.d_model, cfg.num_recursions, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(cfg.d_model, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, cfg.num_recursions, bias=False),
            )
        self.num_recursions = cfg.num_recursions

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(h)
        probs = F.softmax(logits, dim=-1)
        depth_idx = probs.argmax(dim=-1)            # 0 ... Nr-1
        depths = depth_idx + 1                      # 1 ... Nr
        return logits, probs, depths


class MoRCausalLM(nn.Module):
    """Small Llama-style Mixture-of-Recursions language model.

    Architecture:
        Embedding -> unique first block -> token-choice router
        -> shared Middle-Cycle core repeated dynamically -> unique last block
        -> RMSNorm -> tied LM head

    The core contains n_shared_layers unique blocks. Repeating the same pool
    num_recursions times creates an effective maximum depth of:
        1 + n_shared_layers * num_recursions + 1.
    """

    def __init__(self, cfg: MoRConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.first_block = TransformerBlock(cfg)
        self.shared_core = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_shared_layers)])
        self.last_block = TransformerBlock(cfg)
        self.router = TokenChoiceRouter(cfg)
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def router_losses(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        depths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nr = self.cfg.num_recursions
        total_tokens = depths.numel()

        # Paper's token-choice balancing objective: Nr * sum(f_i * p_i).
        one_hot = F.one_hot(depths - 1, num_classes=nr).float()
        fraction = one_hot.mean(dim=(0, 1)).detach()
        mean_prob = probs.mean(dim=(0, 1))
        balance = nr * torch.sum(fraction * mean_prob)

        # Standard router z-loss used by the paper's router ablations.
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        return balance, z

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.embed_tokens(input_ids)
        h1 = self.first_block(h)

        router_logits, router_probs, depths = self.router(h1)
        current = h1
        final_recursive = torch.zeros_like(h1)
        finished = torch.zeros_like(depths, dtype=torch.bool)

        for r in range(1, self.cfg.num_recursions + 1):
            active = depths >= r
            core_out = current
            for block in self.shared_core:
                block_out = block(core_out, active_mask=active)
                # Only active tokens update at this recursion depth.
                core_out = torch.where(active.unsqueeze(-1), block_out, core_out)

            gate = router_probs[..., r - 1].unsqueeze(-1)
            routed = gate * core_out

            # Eq. (2)-style exit: when a token reaches its assigned recursion,
            # add back H1 and freeze that token's recursive representation.
            exits_now = depths == r
            exit_value = routed + h1
            final_recursive = torch.where(exits_now.unsqueeze(-1), exit_value, final_recursive)
            finished = finished | exits_now

            # Tokens routed deeper continue from the gated representation.
            continue_mask = depths > r
            current = torch.where(continue_mask.unsqueeze(-1), routed, current)

        # Safety fallback (normally every token exits exactly once).
        final_recursive = torch.where(finished.unsqueeze(-1), final_recursive, current)
        h = self.last_block(final_recursive)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        output: Dict[str, torch.Tensor] = {
            "logits": logits,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "depths": depths,
        }

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            balance_loss, z_loss = self.router_losses(router_logits, router_probs, depths)
            total_loss = (
                lm_loss
                + self.cfg.router_balance_coeff * balance_loss
                + self.cfg.router_z_loss_coeff * z_loss
            )
            output.update(
                {
                    "loss": total_loss,
                    "lm_loss": lm_loss.detach(),
                    "balance_loss": balance_loss.detach(),
                    "z_loss": z_loss.detach(),
                    "avg_depth": depths.float().mean().detach(),
                }
            )

        return output

    def parameter_report(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = self.embed_tokens.weight.numel()
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embedding,
            "non_embedding_parameters": total - embedding,
        }
