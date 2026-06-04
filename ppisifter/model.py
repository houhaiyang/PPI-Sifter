import torch
import torch.nn as nn
from .modules import (
    InputProjection,
    BidirectionalCrossAttention,
    GatedFFN,
    AttentionPooling,
    ClassificationHead,
    fuse_symmetric,
    build_attention_map,
)


class PPISifter(nn.Module):
    def __init__(self, input_dim=1152, proj_dim=256, num_heads=2, dropout=0.25, attn_dropout=0.1, ffn_hidden=4096, cls_hidden=(256, 64), score_temperature=1.0, topk_default=20):
        super().__init__()
        self.input_proj = InputProjection(input_dim, proj_dim, dropout=dropout)
        self.cross = BidirectionalCrossAttention(proj_dim, num_heads=num_heads, dropout=attn_dropout)
        self.ffn_a = GatedFFN(proj_dim, hidden_dim=ffn_hidden, dropout=dropout)
        self.ffn_b = GatedFFN(proj_dim, hidden_dim=ffn_hidden, dropout=dropout)
        self.pool = AttentionPooling(proj_dim)
        self.cls = ClassificationHead(proj_dim, hidden=cls_hidden, dropout=dropout)
        self.score_temperature = score_temperature
        self.topk_default = topk_default

    def forward(self, a, b, mask_a, mask_b):
        a = self.input_proj(a)
        b = self.input_proj(b)
        a, b, attn_ab, attn_ba = self.cross(a, b, mask_a, mask_b)
        a = self.ffn_a(a)
        b = self.ffn_b(b)
        sa, wa = self.pool(a, mask_a)
        sb, wb = self.pool(b, mask_b)
        v = fuse_symmetric(sa, sb)
        logit = self.cls(v)
        attn_map, valid = build_attention_map(attn_ab, attn_ba, mask_a, mask_b)
        return {
            'logit': logit,
            'prob': torch.sigmoid(logit),
            'pooled_a': sa,
            'pooled_b': sb,
            'pool_weights_a': wa,
            'pool_weights_b': wb,
            'attn_map': attn_map / self.score_temperature,
            'valid_mask': valid,
        }
