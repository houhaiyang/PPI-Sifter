"""
PPI-Sifter 对比学习模块
功能: 投影头 + Supervised Contrastive Loss（数值稳定版）+ Anchor-Partner Triplet Loss

修复：
  - SupervisedContrastiveLoss 使用 log-sum-exp trick 彻底消除 nan
  - 强制 FP32 计算相似度，不受 fp16 影响
  - 增加 batch 内类别数量检查，单类 batch 直接返回 0
"""

import warnings
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
        x = x.float()
        z = self.net(x)                                   # (B, d_proj)
        # eps=1e-6 防止零向量时梯度爆炸（F.normalize 默认 eps=1e-12 太小）
        return F.normalize(z, dim=-1, eps=1e-6)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss（Khosla et al. 2020），数值稳定版。

    修复要点：
        1. 手动实现 log-sum-exp trick，避免 F.log_softmax 在极值下的梯度 nan
        2. 强制 FP32 计算全程（兼容 fp16 训练）
        3. batch 内只有单一类别时直接返回 0，不参与梯度

    参数:
        temperature:      温度参数（建议 0.1，不要低于 0.07）
        base_temperature: 归一化基温
    """

    def __init__(
        self,
        temperature: float = 0.1,
        base_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        参数:
            features: (B, d_proj) L2 归一化，建议已经是 FP32
            labels:   (B,) int64，0 或 1
        返回:
            scalar loss（FP32）
        """
        # 强制 FP32，彻底隔离 fp16 数值问题
        features = features.float()
        device = features.device
        B = features.size(0)

        # batch 内至少要有两种标签才有意义
        unique_labels = labels.unique()
        if unique_labels.numel() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ── 相似度矩阵 ────────────────────────────────────────────────────
        # features 已 L2 归一化，matmul 结果值域 [-1, 1]
        # 除以 temperature 后值域 [-1/t, 1/t]
        features = F.normalize(features, dim=-1, eps=1e-6)  # 双保险归一化
        sim = torch.matmul(features, features.T)            # (B, B)，值域 [-1,1]
        sim = sim / self.temperature                         # 放大到 [-1/t, 1/t]

        # ── mask 构造 ─────────────────────────────────────────────────────
        # mask
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        labels_2d = labels.view(-1, 1)
        mask_pos = (labels_2d == labels_2d.T) & ~mask_self

        has_pos = mask_pos.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # sim
        features = F.normalize(features.float(), dim=-1, eps=1e-6)
        sim = torch.matmul(features, features.T) / self.temperature

        # 排除自身
        sim_no_self = sim.masked_fill(mask_self, float("-inf"))

        # 稳定 logsumexp
        log_partition = torch.logsumexp(sim_no_self, dim=1, keepdim=True)
        log_prob = sim_no_self - log_partition

        # 关键：不要再用 mask * log_prob，避免 0 * -inf = nan
        pos_count = mask_pos.float().sum(dim=1).clamp(min=1.0)
        log_prob_masked = log_prob.masked_fill(~mask_pos, 0.0)
        mean_log_prob_pos = log_prob_masked.sum(dim=1) / pos_count

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[has_pos].mean()

        if not torch.isfinite(loss):
            warnings.warn(
                f"[SupervisedContrastiveLoss] loss={loss.item():.4f}，已置 0。"
                f" B={B}, temperature={self.temperature},"
                f" unique_labels={unique_labels.tolist()},"
                f" sim_range=[{sim.min().item():.2f}, {sim.max().item():.2f}]"
            )
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss


class TripletPartnerLoss(nn.Module):
    """
    Anchor-Partner Triplet Loss。

    参数:
        margin: triplet margin
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor:   Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """
        参数:
            anchor/positive/negative: (B, d_proj) L2 归一化
        返回:
            scalar loss
        """
        d_pos = (anchor - positive).pow(2).sum(dim=-1)
        d_neg = (anchor - negative).pow(2).sum(dim=-1)
        return F.relu(d_pos - d_neg + self.margin).mean()


class LayerwiseContrastHead(nn.Module):
    """
    为每一层 cross-attention 中间表示分别建立独立投影头。

    参数:
        n_layers:       cross-attention 层数
        d_in:           每层 pair_repr 的维度（= d_model * 4）
        d_proj:         投影后对比空间维度
        active_layers:  参与对比学习的层 index 列表，None 表示全部
        loss_type:      supcon | triplet | both
        triplet_margin: triplet margin
    """

    def __init__(
        self,
        n_layers:       int,
        d_in:           int,
        d_proj:         int   = 128,
        active_layers:  Optional[list] = None,
        dropout:        float = 0.1,
        loss_type:      str   = "supcon",
        triplet_margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_layers     = n_layers
        self.active_layers = (
            active_layers if active_layers is not None else list(range(n_layers))
        )
        self.heads = nn.ModuleDict({
            str(i): ContrastProjectionHead(d_in, d_proj, dropout)
            for i in self.active_layers
        })
        self.loss_fn    = SupervisedContrastiveLoss()
        self.loss_type  = loss_type
        self.triplet_fn = (
            TripletPartnerLoss(margin=triplet_margin)
            if loss_type != "supcon" else None
        )

    def forward(
        self,
        layer_reprs:  dict,
        labels:       Tensor,
        layer_weights: Optional[dict] = None,
    ) -> Tensor:
        """
        参数:
            layer_reprs:   dict {layer_idx (int): Tensor (B, d_in)}
            labels:        (B,) int64
            layer_weights: dict {layer_idx: float}，默认等权

        返回:
            total_contrast_loss: scalar
        """
        total = torch.tensor(0.0, device=labels.device)
        count = 0
        for i in self.active_layers:
            if i not in layer_reprs:
                continue
            proj = self.heads[str(i)](layer_reprs[i])       # (B, d_proj)，L2 归一化
            layer_loss = self.loss_fn(proj, labels)
            w = float(layer_weights[i]) if (layer_weights and i in layer_weights) else 1.0
            total = total + w * layer_loss
            count += 1
        return total / max(count, 1)