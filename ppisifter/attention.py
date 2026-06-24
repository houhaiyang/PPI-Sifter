"""
PPI-Sifter 注意力模块
功能: 双向 cross-attention、Gated FFN、Attention Pooling
参考: B-PPI 双向 cross-attention 主干设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class BidirectionalCrossAttention(nn.Module):
    """
    双向 multi-head cross-attention 模块。

    方向 A->B: A 作为 query，B 作为 key/value
    方向 B->A: B 作为 query，A 作为 key/value
    两个方向共享 QKV 投影权重（B-PPI 风格对称设计）。

    参数:
        d_model:  特征维度
        n_heads:  注意力头数
        dropout:  dropout 比率
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 共享 Q/K/V 投影（A->B 与 B->A 复用同一组权重）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        单方向 scaled dot-product attention。

        参数:
            query:    (B, Lq, D)
            key:      (B, Lk, D)
            value:    (B, Lk, D)
            key_mask: (B, Lk) True=有效位置

        返回:
            out:      (B, Lq, D)
            attn_w:   (B, H, Lq, Lk)  注意力权重
        """
        B, Lq, D = query.shape
        Lk = key.size(1)

        # 线性投影并拆分多头 (B, H, L, d_head)
        Q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product  (B, H, Lq, Lk)
        scale = self.d_head ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Padding mask：将 pad 位置设为 -inf
        if key_mask is not None:
            # key_mask: (B, Lk) -> (B, 1, 1, Lk)
            pad_mask = ~key_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        attn_w = F.softmax(attn_scores, dim=-1)
        # 若整列全为 -inf（全 padding），softmax 输出 NaN，强制置 0
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.attn_drop(attn_w)

        # 加权求和 (B, H, Lq, d_head) -> (B, Lq, D)
        ctx = torch.matmul(attn_w, V).transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(ctx)
        return out, attn_w

    def forward(
        self,
        h_a: Tensor,
        h_b: Tensor,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        双向 cross-attention 前向传播。

        参数:
            h_a:    (B, La, D)
            h_b:    (B, Lb, D)
            mask_a: (B, La) True=有效
            mask_b: (B, Lb) True=有效

        返回:
            h_a_new:  (B, La, D)  更新后的 A 表示
            h_b_new:  (B, Lb, D)  更新后的 B 表示
            attn_ab:  (B, H, La, Lb)  A->B 注意力权重
            attn_ba:  (B, H, Lb, La)  B->A 注意力权重
        """
        # A -> B
        ctx_a, attn_ab = self._attend(h_a, h_b, h_b, mask_b)
        h_a_new = self.norm_a(h_a + self.resid_drop(ctx_a))

        # B -> A
        ctx_b, attn_ba = self._attend(h_b, h_a, h_a, mask_a)
        h_b_new = self.norm_b(h_b + self.resid_drop(ctx_b))

        return h_a_new, h_b_new, attn_ab, attn_ba


class GatedFFN(nn.Module):
    """
    门控前馈网络（Gated FFN），对齐 B-PPI 设计。

    结构: Linear -> GLU(gate * tanh(value)) -> Linear + 残差
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        d_ff = d_model * expansion
        self.fc1 = nn.Linear(d_model, d_ff * 2)   # 输出分两路：gate + value
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        参数: x (B, L, D)
        返回: (B, L, D)
        """
        gate, value = self.fc1(x).chunk(2, dim=-1)  # 各 (B, L, d_ff)
        hidden = F.silu(gate) * value               # SwiGLU 变体
        out = self.fc2(self.drop(hidden))
        return self.norm(x + self.drop(out))


class AttentionPooling(nn.Module):
    """
    注意力池化：将可变长度残基表示压缩为固定向量。

    参数:
        d_model: 特征维度
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        参数:
            h:    (B, L, D)
            mask: (B, L) True=有效残基

        返回:
            pooled: (B, D)
        """
        # 计算残基重要性分数 (B, L, 1)
        scores = self.score(h)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = F.softmax(scores, dim=1)          # (B, L, 1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights * h).sum(dim=1)           # (B, D)
        return pooled
