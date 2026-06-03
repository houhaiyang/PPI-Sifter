from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class LossOutput:
    total: torch.Tensor
    items: Dict[str, float]


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class PPISifterLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 10.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_sparse: float = 1e-4,
        lambda_sym: float = 1e-3,
        lambda_hotspot: float = 0.0,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lambda_focal = lambda_focal
        self.lambda_sparse = lambda_sparse
        self.lambda_sym = lambda_sym
        self.lambda_hotspot = lambda_hotspot

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_ab: Optional[torch.Tensor] = None,
        attention_ba: Optional[torch.Tensor] = None,
        hotspot_targets: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        labels = labels.float().view_as(logits)
        bce_loss = self.bce(logits, labels)
        focal_loss = self.focal(logits, labels)

        sparse_loss = torch.tensor(0.0, device=logits.device)
        sym_loss = torch.tensor(0.0, device=logits.device)
        hotspot_loss = torch.tensor(0.0, device=logits.device)

        if attention_ab is not None:
            sparse_loss = attention_ab.mean()
        if attention_ab is not None and attention_ba is not None:
            sym_loss = torch.mean(torch.abs(attention_ab - attention_ba.transpose(-1, -2)))
        if attention_ab is not None and hotspot_targets is not None and self.lambda_hotspot > 0:
            hotspot_loss = F.mse_loss(attention_ab, hotspot_targets)

        total = bce_loss + self.lambda_focal * focal_loss + self.lambda_sparse * sparse_loss + self.lambda_sym * sym_loss + self.lambda_hotspot * hotspot_loss
        items = {
            'loss_total': float(total.detach().cpu()),
            'loss_bce': float(bce_loss.detach().cpu()),
            'loss_focal': float(focal_loss.detach().cpu()),
            'loss_sparse': float(sparse_loss.detach().cpu()),
            'loss_sym': float(sym_loss.detach().cpu()),
            'loss_hotspot': float(hotspot_loss.detach().cpu()),
        }
        return LossOutput(total=total, items=items)
