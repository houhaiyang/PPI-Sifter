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
from typing import Dict, List, Optional, Tuple

from ppisifter.attention import BidirectionalCrossAttention, GatedFFN, AttentionPooling


class PPISifter(nn.Module):
    """
    PPI-Sifter: B-PPI 风格判别主干 + residue-level 可解释输出。

    参数:
        d_in:          输入 embedding 维度（ESM-C 600M=1152，300M=960）
        d_model:       内部投影维度
        n_heads:       cross-attention 头数（d_model 必须能被 n_heads 整除）
        n_layers:      cross-attention + gated FFN 堆叠层数
        ffn_expansion: gated FFN 扩展倍率
        dropout:       dropout 比率
    """

    def __init__(
        self,
        d_in:          int   = 1152,
        d_model:       int   = 256,
        n_heads:       int   = 8,
        n_layers:      int   = 2,
        ffn_expansion: int   = 4,
        dropout:       float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers

        # 5.2 输入投影（共享权重，A/B 使用同一投影层）
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
        )

        # 5.3 双向 cross-attention（N 层）
        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 5.4 gated FFN（N 层，A/B 同层内共享 FFN 参数）
        self.ffn_layers = nn.ModuleList([
            GatedFFN(d_model, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])

        # 5.5 attention pooling（A/B 各一个）
        self.pool_a = AttentionPooling(d_model)
        self.pool_b = AttentionPooling(d_model)

        # 5.7 MLP 分类头
        # 对称共享表示: [s_a, s_b, s_a+s_b, |s_a-s_b|]，共 4 * d_model
        mlp_in = d_model * 4
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
        """Xavier 均匀初始化线性层权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        emb_a:            Tensor,
        emb_b:            Tensor,
        mask_a:           Optional[Tensor] = None,
        mask_b:           Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Tensor]:
        """
        前向传播。

        参数:
            emb_a:            (B, La, d_in)  蛋白 A residue embeddings
            emb_b:            (B, Lb, d_in)  蛋白 B residue embeddings
            mask_a:           (B, La)  True=有效残基，False=padding
            mask_b:           (B, Lb)  True=有效残基，False=padding
            return_attention: 是否同时返回 attention map

        返回 dict:
            logits:   (B,)       pair-level interaction logit
            prob:     (B,)       interaction 概率
            attn_map: (B, La, Lb) 对称化 attention map（return_attention=True 时）
            attn_ab:  (B, La, Lb) A->B 权重（return_attention=True 时）
            attn_ba:  (B, Lb, La) B->A 权重（return_attention=True 时）
        """
        # 5.2 输入投影  h_a: (B, La, d_model)  h_b: (B, Lb, d_model)
        h_a = self.input_proj(emb_a)
        h_b = self.input_proj(emb_b)

        # 5.3 + 5.4 多层双向 cross-attention + gated FFN
        all_attn_ab: List[Tensor] = []
        all_attn_ba: List[Tensor] = []
        for i in range(self.n_layers):
            h_a, h_b, attn_ab, attn_ba = self.cross_attn_layers[i](
                h_a, h_b, mask_a, mask_b
            )
            h_a = self.ffn_layers[i](h_a)
            h_b = self.ffn_layers[i](h_b)
            all_attn_ab.append(attn_ab)   # (B, H, La, Lb)
            all_attn_ba.append(attn_ba)   # (B, H, Lb, La)

        # 5.5 Attention pooling
        s_a = self.pool_a(h_a, mask_a)   # (B, d_model)
        s_b = self.pool_b(h_b, mask_b)   # (B, d_model)

        # 5.6 对称共享表示: [s_a, s_b, s_a+s_b, |s_a-s_b|] -> (B, 4*d_model)
        v_full = torch.cat(
            [s_a, s_b, s_a + s_b, torch.abs(s_a - s_b)], dim=-1
        )

        # 5.7 分类头
        logits = self.classifier(v_full).squeeze(-1)   # (B,)
        prob   = torch.sigmoid(logits)

        output: Dict[str, Tensor] = {"logits": logits, "prob": prob}

        if return_attention:
            # 6.1 取最后一层，对 H 个头取平均 -> (B, La, Lb)
            attn_ab_last = all_attn_ab[-1].mean(dim=1)
            attn_ba_last = all_attn_ba[-1].mean(dim=1)
            # 对称融合
            attn_map = (attn_ab_last + attn_ba_last.transpose(-1, -2)) / 2.0
            output["attn_map"] = attn_map
            output["attn_ab"]  = attn_ab_last
            output["attn_ba"]  = attn_ba_last

        return output

    @torch.no_grad()
    def get_topk_residue_pairs(
        self,
        attn_map: Tensor,
        k:        int = 20,
        mask_a:   Optional[Tensor] = None,
        mask_b:   Optional[Tensor] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        从 attention map 提取 top-k residue pairs。

        参数:
            attn_map: (1, La, Lb) 或 (La, Lb)
            k:        提取数量
            mask_a/b: 有效残基 mask

        返回:
            list of (i, j, score)，按 score 降序
        """
        if attn_map.dim() == 3:
            attn_map = attn_map.squeeze(0)

        la, lb  = attn_map.shape
        valid_a = int(mask_a.squeeze(0).sum().item()) if mask_a is not None else la
        valid_b = int(mask_b.squeeze(0).sum().item()) if mask_b is not None else lb

        valid_map = attn_map[:valid_a, :valid_b]
        flat      = valid_map.flatten()
        k         = min(k, flat.numel())
        top_vals, top_idx = torch.topk(flat, k)

        pairs: List[Tuple[int, int, float]] = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            i = idx // valid_b
            j = idx  % valid_b
            pairs.append((i, j, val))
        return pairs