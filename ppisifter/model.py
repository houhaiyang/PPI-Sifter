"""
PPI-Sifter 主模型
结构: 输入投影 -> 双向 cross-attention x N -> gated FFN -> attention pooling
      -> 对称共享表示 -> MLP 分类头
      -> 导出 residue-level attention map & top-k residue pairs
      -> [对比学习版] 导出 layer-wise pair representations
参考: B-PPI bidirectional cross-attention backbone
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from ppisifter.attention import BidirectionalCrossAttention, GatedFFN, AttentionPooling


class PPISifter(nn.Module):
    """
    PPI-Sifter: B-PPI 风格判别主干 + residue-level 可解释输出 + layer-wise 对比表示导出。

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

        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LayerNorm(d_model),
        )

        self.cross_attn_layers = nn.ModuleList([
            BidirectionalCrossAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            GatedFFN(d_model, ffn_expansion, dropout)
            for _ in range(n_layers)
        ])

        self.pool_a = AttentionPooling(d_model)
        self.pool_b = AttentionPooling(d_model)

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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_pair_repr(self, s_a: Tensor, s_b: Tensor) -> Tensor:
        """构造对称 pair representation: [sa, sb, sa+sb, |sa-sb|] -> (B, 4*d_model)"""
        return torch.cat([s_a, s_b, s_a + s_b, torch.abs(s_a - s_b)], dim=-1)

    def forward(
        self,
        emb_a:              Tensor,
        emb_b:              Tensor,
        mask_a:             Optional[Tensor] = None,
        mask_b:             Optional[Tensor] = None,
        return_attention:   bool = False,
        return_layer_reprs: bool = False,
    ) -> Dict[str, Tensor]:
        """
        前向传播。

        参数:
            emb_a:              (B, La, d_in)
            emb_b:              (B, Lb, d_in)
            mask_a:             (B, La)  True=有效残基
            mask_b:             (B, Lb)  True=有效残基
            return_attention:   是否导出 attention map
            return_layer_reprs: 是否导出每层 pair representation（对比学习用）

        返回 dict:
            logits:       (B,)
            prob:         (B,)
            pair_repr:    (B, 4*d_model)  最终对称表示
            layer_reprs:  dict {layer_idx: (B, 4*d_model)}  各层中间 pair repr
            attn_map:     (B, La, Lb)  对称化 attention（return_attention=True）
            attn_ab:      (B, La, Lb)
            attn_ba:      (B, Lb, La)
            attn_stats:   dict，每层 entropy/symmetry 统计（return_attention=True）
        """
        h_a = self.input_proj(emb_a)   # (B, La, d_model)
        h_b = self.input_proj(emb_b)   # (B, Lb, d_model)

        all_attn_ab: List[Tensor] = []
        all_attn_ba: List[Tensor] = []
        layer_reprs: Dict[int, Tensor] = {}

        for i in range(self.n_layers):
            h_a, h_b, attn_ab, attn_ba = self.cross_attn_layers[i](h_a, h_b, mask_a, mask_b)
            h_a = self.ffn_layers[i](h_a)
            h_b = self.ffn_layers[i](h_b)
            all_attn_ab.append(attn_ab)
            all_attn_ba.append(attn_ba)

            if return_layer_reprs:
                # 每层导出 pooled pair repr（detach 可选，训练时不 detach）
                s_a_l = self.pool_a(h_a, mask_a)   # (B, d_model)
                s_b_l = self.pool_b(h_b, mask_b)   # (B, d_model)
                layer_reprs[i] = self._make_pair_repr(s_a_l, s_b_l)  # (B, 4*d_model)

        s_a = self.pool_a(h_a, mask_a)
        s_b = self.pool_b(h_b, mask_b)
        pair_repr = self._make_pair_repr(s_a, s_b)  # (B, 4*d_model)

        logits = self.classifier(pair_repr).squeeze(-1)
        prob   = torch.sigmoid(logits)

        output: Dict = {
            "logits":    logits,
            "prob":      prob,
            "pair_repr": pair_repr,
        }

        if return_layer_reprs:
            output["layer_reprs"] = layer_reprs

        if return_attention:
            attn_ab_last = all_attn_ab[-1].mean(dim=1)   # (B, La, Lb)
            attn_ba_last = all_attn_ba[-1].mean(dim=1)   # (B, Lb, La)
            attn_map = (attn_ab_last + attn_ba_last.transpose(-1, -2)) / 2.0
            output["attn_map"] = attn_map
            output["attn_ab"]  = attn_ab_last
            output["attn_ba"]  = attn_ba_last

            # per-layer attention statistics（entropy, symmetry error）
            attn_stats = {}
            for l_idx, (ab, ba) in enumerate(zip(all_attn_ab, all_attn_ba)):
                ab_mean = ab.mean(dim=1)   # (B, La, Lb)
                ba_mean = ba.mean(dim=1)   # (B, Lb, La)
                eps = 1e-8
                entropy_ab = -(ab_mean * (ab_mean + eps).log()).sum(dim=-1).mean()
                sym_err = (ab_mean - ba_mean.transpose(-1, -2)).pow(2).mean()
                attn_stats[l_idx] = {
                    "entropy_ab": entropy_ab.item(),
                    "sym_err":    sym_err.item(),
                }
            output["attn_stats"] = attn_stats

        return output

    @torch.no_grad()
    def get_topk_residue_pairs(
        self,
        attn_map: Tensor,
        k:        int = 20,
        mask_a:   Optional[Tensor] = None,
        mask_b:   Optional[Tensor] = None,
    ) -> List[Tuple[int, int, float]]:
        if attn_map.dim() == 3:
            attn_map = attn_map.squeeze(0)
        la, lb  = attn_map.shape
        valid_a = int(mask_a.squeeze(0).sum().item()) if mask_a is not None else la
        valid_b = int(mask_b.squeeze(0).sum().item()) if mask_b is not None else lb
        valid_map = attn_map[:valid_a, :valid_b]
        flat      = valid_map.flatten()
        k         = min(k, flat.numel())
        top_vals, top_idx = torch.topk(flat, k)
        pairs = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            i = idx // valid_b
            j = idx  % valid_b
            pairs.append((i, j, val))
        return pairs