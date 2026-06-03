from __future__ import annotations

import torch
from torch import nn


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.query_a = nn.Linear(input_dim, attention_dim)
        self.key_a = nn.Linear(input_dim, attention_dim)
        self.value_a = nn.Linear(input_dim, attention_dim)
        self.query_b = nn.Linear(input_dim, attention_dim)
        self.key_b = nn.Linear(input_dim, attention_dim)
        self.value_b = nn.Linear(input_dim, attention_dim)
        self.scale = attention_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.out_a = nn.Linear(attention_dim, input_dim)
        self.out_b = nn.Linear(attention_dim, input_dim)
        self.num_heads = num_heads

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return weights

    def forward(self, emb_a: torch.Tensor, mask_a: torch.Tensor, emb_b: torch.Tensor, mask_b: torch.Tensor):
        q_a = self.query_a(emb_a)
        k_b = self.key_b(emb_b)
        v_b = self.value_b(emb_b)
        q_b = self.query_b(emb_b)
        k_a = self.key_a(emb_a)
        v_a = self.value_a(emb_a)

        logits_ab = torch.matmul(q_a, k_b.transpose(-1, -2)) * self.scale
        logits_ba = torch.matmul(q_b, k_a.transpose(-1, -2)) * self.scale

        mask_ab = mask_a.unsqueeze(-1) & mask_b.unsqueeze(1)
        mask_ba = mask_b.unsqueeze(-1) & mask_a.unsqueeze(1)

        attn_ab = self._masked_softmax(logits_ab, mask_ab)
        attn_ba = self._masked_softmax(logits_ba, mask_ba)

        ctx_a = torch.matmul(self.dropout(attn_ab), v_b)
        ctx_b = torch.matmul(self.dropout(attn_ba), v_a)

        out_a = self.out_a(ctx_a) + emb_a
        out_b = self.out_b(ctx_b) + emb_b
        return out_a, out_b, attn_ab, attn_ba
