from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ppisifter.attention import BidirectionalCrossAttention


class MaskedAttentivePooling(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class ProteinFastFilter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.pool = MaskedAttentivePooling(input_dim=input_dim, dropout=dropout)
        fusion_dim = input_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def fuse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

    def forward(self, emb_a: torch.Tensor, mask_a: torch.Tensor, emb_b: torch.Tensor, mask_b: torch.Tensor):
        pooled_a, weights_a = self.pool(emb_a, mask_a)
        pooled_b, weights_b = self.pool(emb_b, mask_b)
        fused = self.fuse(pooled_a, pooled_b)
        logits = self.mlp(fused)
        return logits, pooled_a, pooled_b, weights_a, weights_b


class ResidueAttentiveReranker(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, pair_hidden_dim: int, attention_dim: int, attention_heads: int, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = BidirectionalCrossAttention(
            input_dim=proj_dim,
            attention_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.pool = MaskedAttentivePooling(input_dim=proj_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 4, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden_dim, 1),
        )

    def forward(self, emb_a: torch.Tensor, mask_a: torch.Tensor, emb_b: torch.Tensor, mask_b: torch.Tensor):
        emb_a = self.proj(emb_a)
        emb_b = self.proj(emb_b)
        ctx_a, ctx_b, attn_ab, attn_ba = self.cross_attn(emb_a, mask_a, emb_b, mask_b)
        pooled_a, _ = self.pool(ctx_a, mask_a)
        pooled_b, _ = self.pool(ctx_b, mask_b)
        fused = torch.cat([pooled_a, pooled_b, torch.abs(pooled_a - pooled_b), pooled_a * pooled_b], dim=-1)
        logits = self.classifier(fused)
        return logits, attn_ab, attn_ba, ctx_a, ctx_b


class PPISifterModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        pair_hidden_dim: int,
        attention_dim: int,
        attention_heads: int,
        dropout: float = 0.2,
        fast_filter_threshold: float = 0.25,
    ):
        super().__init__()
        self.fast_filter = ProteinFastFilter(input_dim=input_dim, hidden_dim=pair_hidden_dim, dropout=dropout)
        self.reranker = ResidueAttentiveReranker(
            input_dim=input_dim,
            proj_dim=proj_dim,
            pair_hidden_dim=pair_hidden_dim,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            dropout=dropout,
        )
        self.fast_filter_threshold = fast_filter_threshold

    def forward(self, emb_a: torch.Tensor, mask_a: torch.Tensor, emb_b: torch.Tensor, mask_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        fast_logits, pooled_a, pooled_b, pool_weights_a, pool_weights_b = self.fast_filter(emb_a, mask_a, emb_b, mask_b)
        rerank_logits, attn_ab, attn_ba, ctx_a, ctx_b = self.reranker(emb_a, mask_a, emb_b, mask_b)
        final_logits = 0.5 * fast_logits + 0.5 * rerank_logits
        return {
            'fast_logits': fast_logits,
            'rerank_logits': rerank_logits,
            'logits': final_logits,
            'attention_ab': attn_ab,
            'attention_ba': attn_ba,
            'context_a': ctx_a,
            'context_b': ctx_b,
            'pool_weights_a': pool_weights_a,
            'pool_weights_b': pool_weights_b,
            'pooled_a': pooled_a,
            'pooled_b': pooled_b,
        }
