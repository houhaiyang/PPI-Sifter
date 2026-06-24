"""
PPI-Sifter 损失函数
结构: weighted BCE + focal loss + sparse regularization + symmetry consistency
     + [对比学习版] layerwise supervised contrastive loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean() if self.reduction == "mean" else focal.sum()


class PPILoss(nn.Module):
    """
    PPI-Sifter 总损失函数（对比学习版）。

    L = L_wbce + λ_focal * L_focal
      + λ_sparse * L_sparse + λ_sym * L_sym
      + λ_contrast * L_contrast

    参数:
        pos_weight:       BCE 正样本权重
        focal_gamma:      focal gamma
        lambda_focal:     focal 权重
        lambda_sparse:    attention 稀疏正则权重
        lambda_sym:       attention 对称一致性权重
        lambda_contrast:  对比损失权重（0 则关闭）
    """

    def __init__(
        self,
        pos_weight:      float = 10.0,
        focal_gamma:     float = 2.0,
        lambda_focal:    float = 0.5,
        lambda_sparse:   float = 1e-3,
        lambda_sym:      float = 1e-2,
        lambda_contrast: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.focal           = FocalLoss(gamma=focal_gamma)
        self.lambda_focal    = lambda_focal
        self.lambda_sparse   = lambda_sparse
        self.lambda_sym      = lambda_sym
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        logits:       Tensor,
        targets:      Tensor,
        attn_ab:      Optional[Tensor] = None,
        attn_ba:      Optional[Tensor] = None,
        contrast_loss: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        参数:
            logits:        (B,)
            targets:       (B,)
            attn_ab:       (B, La, Lb)  可选
            attn_ba:       (B, Lb, La)  可选
            contrast_loss: 预先计算好的对比损失 scalar，可选

        返回:
            dict，包含 total 和各分项（便于 TensorBoard 记录）
        """
        l_wbce  = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        l_focal = self.focal(logits, targets)
        total   = l_wbce + self.lambda_focal * l_focal

        l_sparse = torch.tensor(0.0, device=logits.device)
        l_sym    = torch.tensor(0.0, device=logits.device)

        if attn_ab is not None and attn_ba is not None:
            eps = 1e-8
            l_sparse = (-attn_ab * torch.log(attn_ab + eps)).mean()
            l_sym    = F.mse_loss(attn_ab, attn_ba.transpose(-1, -2))
            total    = total + self.lambda_sparse * l_sparse + self.lambda_sym * l_sym

        l_contrast = torch.tensor(0.0, device=logits.device)
        if contrast_loss is not None and self.lambda_contrast > 0:
            l_contrast = contrast_loss
            total      = total + self.lambda_contrast * l_contrast

        return {
            "total":      total,
            "wbce":       l_wbce.detach(),
            "focal":      l_focal.detach(),
            "sparse":     l_sparse.detach(),
            "sym":        l_sym.detach(),
            "contrast":   l_contrast.detach(),
        }