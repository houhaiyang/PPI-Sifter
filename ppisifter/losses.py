"""
PPI-Sifter 损失函数
结构: weighted BCE + focal loss + sparse regularization + symmetry consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss，聚焦难分类样本。"""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


class PPILoss(nn.Module):
    """
    PPI-Sifter 总损失函数。

    L = L_wbce + lambda_focal * L_focal + lambda_sparse * L_sparse + lambda_sym * L_sym

    参数:
        pos_weight:     BCE 正样本权重（推荐等于 neg_pos_ratio）
        focal_gamma:    focal loss gamma
        lambda_focal:   focal loss 权重
        lambda_sparse:  attention 稀疏正则权重
        lambda_sym:     attention 对称一致性正则权重
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        focal_gamma: float = 2.0,
        lambda_focal: float = 0.5,
        lambda_sparse: float = 1e-3,
        lambda_sym: float = 1e-2,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.focal = FocalLoss(gamma=focal_gamma)
        self.lambda_focal = lambda_focal
        self.lambda_sparse = lambda_sparse
        self.lambda_sym = lambda_sym

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        attn_ab: Optional[Tensor] = None,
        attn_ba: Optional[Tensor] = None,
    ) -> Tensor:
        """
        参数:
            logits:   (B,)
            targets:  (B,)
            attn_ab:  (B, La, Lb)  A->B 注意力权重（可选）
            attn_ba:  (B, Lb, La)  B->A 注意力权重（可选）

        返回:
            total_loss: scalar Tensor
        """
        # weighted BCE
        l_wbce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight
        )

        # focal loss
        l_focal = self.focal(logits, targets)

        total = l_wbce + self.lambda_focal * l_focal

        # 注意力正则（仅在有 attention 时计算）
        if attn_ab is not None and attn_ba is not None:
            # 稀疏正则: 鼓励 attention 集中，用 entropy 近似
            # L_sparse = mean(- sum(p * log(p+eps)))，值越小表示越集中
            eps = 1e-8
            l_sparse = (-attn_ab * torch.log(attn_ab + eps)).mean()
            l_sym = F.mse_loss(attn_ab, attn_ba.transpose(-1, -2))
            total = total + self.lambda_sparse * l_sparse + self.lambda_sym * l_sym

        return total
