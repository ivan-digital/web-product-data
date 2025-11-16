#!/usr/bin/env python3
"""
Lightweight embedding classifier implemented purely in PyTorch.

The module mirrors the architectural blocks popularised by the
Qwen embedding models (token embedding, rotary multi-head
self-attention with grouped key/value heads, SwiGLU feed-forward,
and RMSNorm).  The defaults describe a compact configuration that
trades capacity for speed and memory, but every hyper-parameter can
be overridden through ``EmbeddingClassifierConfig``.

Usage
-----
```python
from category_classification.embedding_classifier import EmbeddingClassifierConfig, EmbeddingClassifier

config = EmbeddingClassifierConfig(
    vocab_size=len(tokenizer),
    num_labels=6,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    ffn_hidden_size=2048,
)
model = EmbeddingClassifier(config)
out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
loss, logits = out["loss"], out["logits"]
```
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Classic RoPE implementation that generates cos/sin tables on the fly."""

    def __init__(self, dim: int, base: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        cos = cos.squeeze(0).squeeze(0)
        sin = sin.squeeze(0).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos[:, :, : q.shape[2], :]
        sin = sin[:, :, : q.shape[2], :]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm used by Qwen."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """Feed-forward block with SwiGLU gating."""

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DropPath(nn.Module):
    """Stochastic depth regularisation."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x * (random_tensor / keep_prob)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingClassifierConfig:
    """
    Minimal configuration record for `EmbeddingClassifier`.

    Set `vocab_size` to the tokenizer vocabulary length (typically
    `len(tokenizer)` from HuggingFace or any custom tokenizer).
    """
    vocab_size: int = 151_936
    hidden_size: int = 384
    ffn_hidden_size: int = 1536
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    num_hidden_layers: int = 10
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 4096
    num_labels: int = 6
    pad_token_id: int = 0
    initializer_range: float = 0.02
    max_drop_path: float = 0.1


class EmbeddingAttention(nn.Module):
    def __init__(self, config: EmbeddingClassifierConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        if (self.head_dim * self.num_heads) != config.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=config.rope_theta)

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int, n_heads: int) -> torch.Tensor:
        return (
            x.view(bsz, seq_len, n_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query = self._shape(self.q_proj(hidden_states), bsz, seq_len, self.num_heads)
        key = self._shape(self.k_proj(hidden_states), bsz, seq_len, self.num_kv_heads)
        value = self._shape(self.v_proj(hidden_states), bsz, seq_len, self.num_kv_heads)

        cos, sin = self.rotary_emb(seq_len, hidden_states.device)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            key = key[:, :, None, :, :].expand(bsz, self.num_kv_heads, repeat_factor, seq_len, self.head_dim)
            value = value[:, :, None, :, :].expand_as(key)
            key = key.reshape(bsz, self.num_heads, seq_len, self.head_dim)
            value = value.reshape(bsz, self.num_heads, seq_len, self.head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class EmbeddingBlock(nn.Module):
    def __init__(self, config: EmbeddingClassifierConfig, drop_path: float = 0.0) -> None:
        super().__init__()
        self.self_attn = EmbeddingAttention(config)
        self.mlp = SwiGLU(config.hidden_size, config.ffn_hidden_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.drop_path_attn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_mlp = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        init_eps = 1e-2
        self.gamma_attn = nn.Parameter(torch.ones(config.hidden_size) * init_eps)
        self.gamma_mlp = nn.Parameter(torch.ones(config.hidden_size) * init_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.drop_path_attn(hidden_states)
        hidden_states = residual + self.gamma_attn * hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.drop_path_mlp(hidden_states)
        return residual + self.gamma_mlp * hidden_states


class EmbeddingBackbone(nn.Module):
    def __init__(self, config: EmbeddingClassifierConfig) -> None:
        super().__init__()
        self.config = config
        # `config.vocab_size` should match the tokenizer length (len(tokenizer))
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        dpr = torch.linspace(0, config.max_drop_path, steps=config.num_hidden_layers).tolist()
        self.layers = nn.ModuleList(
            [EmbeddingBlock(config, drop_path=dpr[i]) for i in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), device=input_ids.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * math.sqrt(self.config.hidden_size)
        hidden_states = self.embed_layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


class EmbeddingClassifier(nn.Module):
    """End-to-end classifier wrapper."""

    def __init__(self, config: EmbeddingClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.backbone = EmbeddingBackbone(config)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, config.num_labels, bias=True),
        )

    def pool(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        masked_hidden = hidden_states * mask
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        return masked_hidden.sum(dim=1) / lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pooled = self.pool(hidden_states, attention_mask=attention_mask)
        logits = self.classifier(pooled)

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        if return_embeddings:
            result["pooled_embeddings"] = pooled
            result["token_embeddings"] = hidden_states
        return result


def build_embedding_classifier(**overrides) -> EmbeddingClassifier:
    """Instantiate ``EmbeddingClassifier`` with overriding kwargs."""
    config = EmbeddingClassifierConfig(**overrides)
    return EmbeddingClassifier(config)


__all__ = [
    "EmbeddingClassifierConfig",
    "EmbeddingBackbone",
    "EmbeddingClassifier",
    "build_embedding_classifier",
]
