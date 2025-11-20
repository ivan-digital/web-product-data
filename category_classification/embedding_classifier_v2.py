#!/usr/bin/env python3
"""
Enhanced embedding classifier with a regularised classifier head.

This module reuses the transformer-style backbone from
`embedding_classifier.py` but swaps the final classification head for a
LayerNorm + residual MLP with adjustable dropout / bottleneck width.

It enables quick experimentation with heavier heads and is meant to be
paired with `train_embedding_classifier_v2.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_classifier import (
    EmbeddingBackbone,
    EmbeddingClassifierConfig,
    RMSNorm,
)


@dataclass
class EmbeddingClassifierV2Config(EmbeddingClassifierConfig):
    """Extends the baseline config with classifier-head controls."""

    classifier_hidden_size: int = 512
    classifier_dropout: float = 0.2
    classifier_layernorm_eps: float = 1e-6
    classifier_residual: bool = True
    pooler: str = "mean"  # ["mean", "cls-first-token"]


class ResidualClassifierHead(nn.Module):
    """
    LayerNorm -> (Linear + GELU + Dropout) x2 -> residual -> LayerNorm -> Linear.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        bottleneck_size: int,
        dropout: float,
        eps: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.input_norm = RMSNorm(hidden_size, eps)
        self.output_norm = RMSNorm(hidden_size, eps)
        self.fc_in = nn.Linear(hidden_size, bottleneck_size)
        self.fc_hidden = nn.Linear(bottleneck_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.input_norm(x)
        hidden = self.fc_in(residual)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc_hidden(hidden)
        hidden = self.dropout(hidden)
        if self.use_residual:
            hidden = hidden + residual
        hidden = self.output_norm(hidden)
        hidden = self.final_dropout(hidden)
        return self.out_proj(hidden)


class EmbeddingClassifierV2(nn.Module):
    """Backbone + enhanced classifier head."""

    def __init__(self, config: EmbeddingClassifierV2Config) -> None:
        super().__init__()
        self.config = config
        self.backbone = EmbeddingBackbone(config)
        self.classifier = ResidualClassifierHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            bottleneck_size=config.classifier_hidden_size or config.hidden_size,
            dropout=config.classifier_dropout,
            eps=config.classifier_layernorm_eps,
            use_residual=config.classifier_residual,
        )

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.pooler == "cls-first-token":
            return hidden_states[:, 0]
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


def build_embedding_classifier_v2(**overrides) -> EmbeddingClassifierV2:
    """Instantiate EmbeddingClassifierV2 with overriding kwargs."""
    config = EmbeddingClassifierV2Config(**overrides)
    return EmbeddingClassifierV2(config)


__all__ = [
    "EmbeddingClassifierV2Config",
    "EmbeddingClassifierV2",
    "build_embedding_classifier_v2",
]
