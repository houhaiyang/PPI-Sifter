"""
PPI-Sifter 对比学习模块
功能: 投影头 + Supervised Contrastive Loss + Anchor-Partner Triplet Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class ContrastProjectionHead(nn.Module):
    """
    将 pair_repr 投影到对比学习的低维超球面空间。
    输入: (B, d_in) -> 输出: (B, d_proj) L2 归一化
    """

    def __init__(self, d_in: int, d_proj: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.LayerNorm(d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_proj),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d_in) -> (B, d_proj) L2 normalized
        return F.normalize(self.net(x), dim=-1)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss（Khosla et al. 2020）。
    同标签样本为 positive，异标签为 negative。

    参数:
        temperature: 温度参数
        base_temperature: 归一化基温（通常与 temperature 相同）
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        参数:
            features: (B, d_proj) L2 归一化表示
            labels:   (B,) int64，0 或 1
        返回:
            scalar loss
        """
        B = features.size(0)
        device = features.device

        # 相似度矩阵 (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # 对角线 mask（自身不参与）
        mask_self = torch.eye(B, dtype=torch.bool, device=device)

        # 正样本 mask：相同标签且不是自身
        labels = labels.view(-1, 1)  # (B, 1)
        mask_pos = (labels == labels.T) & ~mask_self  # (B, B)

        # 无正样本的行直接跳过（避免 NaN）
        has_pos = mask_pos.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # log_softmax（排除自身）
        sim_no_self = sim.masked_fill(mask_self, float("-inf"))
        log_prob = F.log_softmax(sim_no_self, dim=-1)

        # 仅对有正样本的行计算损失
        mean_log_prob_pos = (mask_pos.float() * log_prob).sum(dim=1) / (
            mask_pos.float().sum(dim=1).clamp(min=1)
        )
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[has_pos].mean()
        return loss


class TripletPartnerLoss(nn.Module):
    """
    Anchor-Partner Triplet Loss。
    固定 anchor protein（来自 pair_repr），真实 partner 视为 positive，
    负样本对视为 negative。

    参数:
        margin: triplet margin
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """
        参数:
            anchor/positive/negative: (B, d_proj) L2 归一化
        返回:
            scalar loss
        """
        d_pos = (anchor - positive).pow(2).sum(dim=-1)  # (B,)
        d_neg = (anchor - negative).pow(2).sum(dim=-1)  # (B,)
        loss = F.relu(d_pos - d_neg + self.margin).mean()
        return loss


class LayerwiseContrastHead(nn.Module):
    """
    为每一层 cross-attention 中间表示分别建立独立投影头。
    支持 layer-wise 消融：可以只激活其中某些层。

    参数:
        n_layers:   cross-attention 层数
        d_in:       每层 pair_repr 的维度
        d_proj:     投影后对比空间维度
        active_layers: 参与对比学习的层 index 列表，None 表示全部
    """

    def __init__(
        self,
        n_layers: int,
        d_in: int,
        d_proj: int = 128,
        active_layers: Optional[list] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.active_layers = active_layers if active_layers is not None else list(range(n_layers))
        self.heads = nn.ModuleDict({
            str(i): ContrastProjectionHead(d_in, d_proj, dropout)
            for i in self.active_layers
        })
        self.loss_fn = SupervisedContrastiveLoss()

    def forward(
        self,
        layer_reprs: dict,
        labels: Tensor,
        layer_weights: Optional[dict] = None,
    ) -> Tensor:
        """
        参数:
            layer_reprs:   dict {layer_idx (int): Tensor (B, d_in)}
            labels:        (B,)
            layer_weights: dict {layer_idx: float}，默认各层等权

        返回:
            total_contrast_loss: scalar
        """
        total = torch.tensor(0.0, device=labels.device)
        count = 0
        for i in self.active_layers:
            if i not in layer_reprs:
                continue
            proj = self.heads[str(i)](layer_reprs[i])  # (B, d_proj)
            l = self.loss_fn(proj, labels)
            w = layer_weights[i] if (layer_weights and i in layer_weights) else 1.0
            total = total + w * l
            count += 1
        return total / max(count, 1)