import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputProjection(nn.Module):
    def __init__(self, input_dim, proj_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=2, dropout=0.1):
        super().__init__()
        self.ab = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ba = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, b, mask_a, mask_b):
        a_out, attn_ab = self.ab(a, b, b, key_padding_mask=~mask_b, need_weights=True, average_attn_weights=False)
        b_out, attn_ba = self.ba(b, a, a, key_padding_mask=~mask_a, need_weights=True, average_attn_weights=False)
        a = self.norm_a(a + self.dropout(a_out))
        b = self.norm_b(b + self.dropout(b_out))
        return a, b, attn_ab, attn_ba


class GatedFFN(nn.Module):
    def __init__(self, d_model, hidden_dim=4096, dropout=0.25):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim // 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.dropout(self.up(x))
        a, b = z.chunk(2, dim=-1)
        gated = a * torch.sigmoid(b)
        out = self.dropout(self.down(gated))
        return self.norm(x + out)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)
        s = s.masked_fill(~mask, float('-inf'))
        w = F.softmax(s, dim=-1)
        pooled = torch.bmm(w.unsqueeze(1), x).squeeze(1)
        return pooled, w


class ClassificationHead(nn.Module):
    def __init__(self, d_model, hidden=(256, 64), dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, v):
        return self.net(v).squeeze(-1)


def fuse_symmetric(sa, sb):
    return torch.cat([sa + sb, torch.abs(sa - sb)], dim=-1)


def build_attention_map(attn_ab, attn_ba, mask_a, mask_b):
    ab = attn_ab.mean(dim=1)
    ba = attn_ba.mean(dim=1).transpose(1, 2)
    score = 0.5 * (ab + ba)
    valid = mask_a.unsqueeze(2) & mask_b.unsqueeze(1)
    score = score.masked_fill(~valid, 0.0)
    return score, valid
