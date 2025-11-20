#!/usr/bin/env python3
"""
Embedding classifier v3 with an additional contrastive projection head.

This module reuses the transformer backbone + residual classifier head
from v2 and adds a projection MLP that can be used with SimCSE-style
contrastive objectives. The forward pass returns both logits and the
normalised projection so trainers can combine cross-entropy and
contrastive losses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding_classifier import EmbeddingBackbone, EmbeddingClassifierConfig, RMSNorm
from .embedding_classifier_v2 import (
    EmbeddingClassifierV2Config,
    ResidualClassifierHead,
)


@dataclass
class EmbeddingClassifierV3Config(EmbeddingClassifierV2Config):
    contrastive_dim: int = 256
    projection_dropout: float = 0.1


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, dropout: float) -> None:
        super().__init__()
        proj_dim = proj_dim or input_dim
        self.net = nn.Sequential(
            RMSNorm(input_dim),
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.net(x)
        return F.normalize(proj, p=2, dim=-1)


class EmbeddingClassifierV3(nn.Module):
    def __init__(self, config: EmbeddingClassifierV3Config) -> None:
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
        self.projection_head: Optional[ProjectionHead]
        if config.contrastive_dim and config.contrastive_dim > 0:
            self.projection_head = ProjectionHead(
                input_dim=config.hidden_size,
                proj_dim=config.contrastive_dim,
                dropout=config.projection_dropout,
            )
        else:
            self.projection_head = None

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
        if self.projection_head is not None:
            result["projection"] = self.projection_head(pooled)
        if return_embeddings:
            result["pooled_embeddings"] = pooled
            result["token_embeddings"] = hidden_states
        return result


def build_embedding_classifier_v3(**overrides) -> EmbeddingClassifierV3:
    config = EmbeddingClassifierV3Config(**overrides)
    return EmbeddingClassifierV3(config)


__all__ = [
    "EmbeddingClassifierV3Config",
    "EmbeddingClassifierV3",
    "build_embedding_classifier_v3",
]
