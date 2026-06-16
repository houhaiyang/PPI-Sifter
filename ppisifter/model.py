"""
PPI-Sifter 主模型
结构: 输入投影 -> 双向 cross-attention x N -> gated FFN -> attention pooling
      -> 对称共享表示 -> MLP 分类头
      -> 导出 residue-level attention map & top-k residue pairs
参考: B-PPI bidirectional cross-attention backbone
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple

from ppisifter.attention import BidirectionalCrossAttention, GatedFFN, AttentionPooling


class PPISifter(nn.Module):
    """
    PPI-Sifter: B-PPI 风格判别主干 + residue-level 可解释输出。

    参数:
        d_in:          输入 embedding 维度（ESM-C 输出维度）
        d_model:       内部投影维度
        n_heads:       cross-attention 头数
        n_layers:      cross-attention 堆叠层数
        ffn_expansion: gated FFN 扩展倍率
        dropout:       dropout 比率
        attn_pool_heads: attention pooling 参数（当前固定 1 头）
    """

    def __init__(
        self,
        d_in: int = 1152,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        attn_pool_heads: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # --- 5.2 输入投影（共享权重，A/B 使用同一投影层）---
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
        )

        # --- 5.3 双向 cross-attention (N 层) ---
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # --- 5.4 gated FFN (每层各一个，A/B 共享同一 FFN 参数) ---
        self.ffn_layers = nn.ModuleList([
            GatedFFN(d_model, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])

        # --- 5.5 attention pooling ---
        self.pool_a = AttentionPooling(d_model)
        self.pool_b = AttentionPooling(d_model)

        # --- 5.7 MLP 分类头 (输入维度 = 2 * d_model * 2 = 4 * d_model) ---
        mlp_in = d_model * 4   # [s_a+s_b || |s_a-s_b|] 各 d_model，concat -> 2*d_model * 2
        self.classifier = nn.Sequential(
            nn.Linear(mlp_in, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化线性层。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        前向传播。

        参数:
            emb_a:            (B, La, d_in)  蛋白 A residue embeddings
            emb_b:            (B, Lb, d_in)  蛋白 B residue embeddings
            mask_a:           (B, La)  True=有效残基
            mask_b:           (B, Lb)  True=有效残基
            return_attention: 是否返回 attention map（推理/解释模式）

        返回 dict:
            logits:      (B,)    pair-level interaction logit
            prob:        (B,)    interaction 概率 (sigmoid)
            attn_map:    (B, La, Lb) 对称化 attention map（仅 return_attention=True）
        """
        # --- 输入投影 ---
        # h_a: (B, La, d_model)  h_b: (B, Lb, d_model)
        h_a = self.input_proj(emb_a)
        h_b = self.input_proj(emb_b)

        # --- 多层双向 cross-attention + gated FFN ---
        all_attn_ab = []
        all_attn_ba = []
        for i in range(self.n_layers):
            h_a, h_b, attn_ab, attn_ba = self.cross_attn_layers[i](
                h_a, h_b, mask_a, mask_b
            )
            # Gated FFN（共享参数，A/B 分别处理）
            h_a = self.ffn_layers[i](h_a)
            h_b = self.ffn_layers[i](h_b)
            all_attn_ab.append(attn_ab)  # (B, H, La, Lb)
            all_attn_ba.append(attn_ba)  # (B, H, Lb, La)

        # --- Attention pooling ---
        s_a = self.pool_a(h_a, mask_a)   # (B, d_model)
        s_b = self.pool_b(h_b, mask_b)   # (B, d_model)

        # --- 5.6 对称共享表示: [(s_a+s_b) || |s_a-s_b|] ---
        v = torch.cat([s_a + s_b, torch.abs(s_a - s_b)], dim=-1)  # (B, 2*d_model)

        # --- 分类头 ---
        # 注意: v shape (B, 2*d_model)，但 mlp_in=4*d_model，需扩展
        # 实际拼接方式: [s_a, s_b, s_a+s_b, |s_a-s_b|] 均衡信息
        v_full = torch.cat([s_a, s_b, s_a + s_b, torch.abs(s_a - s_b)], dim=-1)  # (B, 4*d_model)
        logits = self.classifier(v_full).squeeze(-1)   # (B,)
        prob = torch.sigmoid(logits)

        output = {"logits": logits, "prob": prob}

        if return_attention:
            # --- 6.1 Residue-level attention map 导出 ---
            # 取最后一层，对 H 个头做平均 -> (B, La, Lb)
            attn_ab_last = all_attn_ab[-1].mean(dim=1)   # (B, La, Lb)
            attn_ba_last = all_attn_ba[-1].mean(dim=1)   # (B, Lb, La)
            # 对称融合: (A->B + transpose(B->A)) / 2
            attn_map = (attn_ab_last + attn_ba_last.transpose(-1, -2)) / 2.0
            output["attn_map"] = attn_map   # (B, La, Lb)
            output["attn_ab"] = attn_ab_last
            output["attn_ba"] = attn_ba_last

        return output

    @torch.no_grad()
    def get_topk_residue_pairs(
        self,
        attn_map: Tensor,
        k: int = 20,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
    ) -> list:
        """
        从 attention map 提取 top-k residue pairs。

        参数:
            attn_map: (1, La, Lb) 或 (La, Lb)  单样本 attention map
            k:        提取数量
            mask_a:   (1, La) 或 (La,)
            mask_b:   (1, Lb) 或 (Lb,)

        返回:
            list of (i, j, score)，按 score 降序排列
        """
        if attn_map.dim() == 3:
            attn_map = attn_map.squeeze(0)   # (La, Lb)

        la, lb = attn_map.shape
        if mask_a is not None:
            ma = mask_a.squeeze(0) if mask_a.dim() == 2 else mask_a
            valid_a = ma.sum().item()
        else:
            valid_a = la
        if mask_b is not None:
            mb = mask_b.squeeze(0) if mask_b.dim() == 2 else mask_b
            valid_b = mb.sum().item()
        else:
            valid_b = lb

        # 只在有效残基范围内取 top-k
        valid_map = attn_map[:int(valid_a), :int(valid_b)]
        flat = valid_map.flatten()
        k = min(k, flat.numel())
        top_vals, top_idx = torch.topk(flat, k)
        pairs = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            i = idx // int(valid_b)
            j = idx % int(valid_b)
            pairs.append((i, j, val))
        return pairs
