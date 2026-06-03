from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def extract_topk_residue_pairs(attention_ab: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor, top_k: int = 50) -> List[List[Dict[str, float]]]:
    results: List[List[Dict[str, float]]] = []
    batch_size = attention_ab.size(0)
    for batch_idx in range(batch_size):
        valid = (mask_a[batch_idx].unsqueeze(-1) & mask_b[batch_idx].unsqueeze(0)).cpu().numpy()
        matrix = attention_ab[batch_idx].detach().cpu().numpy()
        matrix = np.where(valid, matrix, -np.inf)
        flat_idx = np.argpartition(matrix.reshape(-1), -min(top_k, matrix.size))[-min(top_k, matrix.size):]
        ranked = flat_idx[np.argsort(matrix.reshape(-1)[flat_idx])[::-1]]
        items = []
        for idx in ranked:
            i, j = np.unravel_index(idx, matrix.shape)
            if not np.isfinite(matrix[i, j]):
                continue
            items.append({'residue_a': int(i + 1), 'residue_b': int(j + 1), 'score': float(matrix[i, j])})
        results.append(items)
    return results


@torch.no_grad()
def compute_attention_statistics(attention_ab: torch.Tensor, attention_ba: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor) -> Dict[str, float]:
    sym_gap = torch.mean(torch.abs(attention_ab - attention_ba.transpose(-1, -2))).item()
    valid_mask = mask_a.unsqueeze(-1) & mask_b.unsqueeze(1)
    valid_attn = torch.where(valid_mask, attention_ab, torch.zeros_like(attention_ab))
    eps = 1e-8
    entropy = -(valid_attn * torch.log(valid_attn + eps)).sum(dim=(-1, -2)).mean().item()
    max_score = valid_attn.max().item()
    return {
        'symmetry_gap': float(sym_gap),
        'attention_entropy': float(entropy),
        'attention_max': float(max_score),
    }


def topk_records_to_frame(pair_ids: List[str], topk_items: List[List[Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for pair_id, items in zip(pair_ids, topk_items):
        for rank, item in enumerate(items, start=1):
            rows.append({
                'pair_id': pair_id,
                'rank': rank,
                'residue_a': item['residue_a'],
                'residue_b': item['residue_b'],
                'score': item['score'],
            })
    return pd.DataFrame(rows)
