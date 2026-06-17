"""
PPI-Sifter 解释性模块
功能: 导出 residue-level attention map 热图、top-k residue pairs CSV
依赖: matplotlib, seaborn, pandas, torch
入口: AttentionInterpreter(model, device)
"""

import os
import csv
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import List, Tuple, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # 无头环境兼容
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


class AttentionInterpreter:
    """
    从模型输出中抽取 attention map，生成热图与 top-k residue pairs。

    参数:
        model:  PPISifter 实例（已加载权重）
        device: 推理设备
    """

    def __init__(self, model, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model.to(device).eval()

    @torch.no_grad()
    def explain_pair(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
        topk: int = 20,
    ) -> dict:
        """
        对单个蛋白对进行解释。

        参数:
            emb_a:  (1, La, D)
            emb_b:  (1, Lb, D)
            mask_a: (1, La)
            mask_b: (1, Lb)
            topk:   提取的 top-k 残基对数量

        返回 dict:
            prob:       float  interaction 概率
            attn_map:   (La, Lb) numpy array
            topk_pairs: list of (i, j, score)
        """
        emb_a = emb_a.to(self.device)
        emb_b = emb_b.to(self.device)
        if mask_a is not None:
            mask_a = mask_a.to(self.device)
        if mask_b is not None:
            mask_b = mask_b.to(self.device)

        out = self.model(emb_a, emb_b, mask_a, mask_b, return_attention=True)
        prob = out["prob"].item()
        attn_map = out["attn_map"].squeeze(0).cpu().numpy()   # (La, Lb)
        topk_pairs = self.model.get_topk_residue_pairs(
            out["attn_map"], k=topk, mask_a=mask_a, mask_b=mask_b
        )
        return {"prob": prob, "attn_map": attn_map, "topk_pairs": topk_pairs}

    def save_attn_heatmap(
        self,
        attn_map: np.ndarray,
        save_path: str,
        title: str = "Residue-Level Attention Map",
    ) -> None:
        """
        保存 attention map 热图。

        参数:
            attn_map:  (La, Lb) numpy array
            save_path: 输出图像路径 (.png)
            title:     图标题
        """
        if not HAS_VIZ:
            print("[警告] matplotlib/seaborn 未安装，跳过热图生成")
            return
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(min(attn_map.shape[1] / 10 + 2, 20),
                                        min(attn_map.shape[0] / 10 + 2, 20)))
        sns.heatmap(attn_map, ax=ax, cmap="YlOrRd", cbar=True,
                    xticklabels=False, yticklabels=False)
        ax.set_title(title)
        ax.set_xlabel("Protein B residues")
        ax.set_ylabel("Protein A residues")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def save_topk_pairs(
        self,
        topk_pairs: List[Tuple],
        save_path: str,
        protein_a_id: str = "A",
        protein_b_id: str = "B",
    ) -> None:
        """
        保存 top-k residue pairs 为 CSV。

        输出列: protein_a, residue_a_idx, protein_b, residue_b_idx, score
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["protein_a", "residue_a_idx",
                             "protein_b", "residue_b_idx", "attn_score"])
            for i, j, score in topk_pairs:
                writer.writerow([protein_a_id, i, protein_b_id, j, f"{score:.6f}"])
