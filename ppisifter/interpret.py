"""
PPI-Sifter 解释性模块
功能: 导出 residue-level attention map 热图、top-k residue pairs CSV
     [对比学习版] 新增 layer-wise 投影空间可视化（UMAP/t-SNE）
依赖: matplotlib, seaborn, pandas, torch
入口: AttentionInterpreter(model, device)

[对比学习版改动]
  1. explain_pair() 增加 return_layer_reprs 参数，支持同时返回各层 pair repr
  2. 新增 plot_layer_umap()：将各层 pair_repr 在 2D 投影空间中绘图
  3. model.forward() 调用处增加 return_layer_reprs 参数传递
  4. 无结构性改动，向下兼容原接口
"""

import os
import csv
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import List, Tuple, Optional, Dict

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
    [对比学习版] 额外支持 layer-wise pair repr 导出与可视化。

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
        return_layer_reprs: bool = False,
    ) -> dict:
        """
        对单个蛋白对进行解释。

        参数:
            emb_a:              (1, La, D)
            emb_b:              (1, Lb, D)
            mask_a:             (1, La)
            mask_b:             (1, Lb)
            topk:               提取的 top-k 残基对数量
            return_layer_reprs: [新增] 是否同时返回各层 pair repr，供对比分析使用

        返回 dict:
            prob:         float  interaction 概率
            attn_map:     (La, Lb) numpy array
            topk_pairs:   list of (i, j, score)
            layer_reprs:  dict {layer_idx: (1, 4*d_model) Tensor}（仅 return_layer_reprs=True 时存在）
        """
        emb_a = emb_a.to(self.device)
        emb_b = emb_b.to(self.device)
        if mask_a is not None:
            mask_a = mask_a.to(self.device)
        if mask_b is not None:
            mask_b = mask_b.to(self.device)

        # [改动 1] 传入 return_layer_reprs 参数
        out = self.model(
            emb_a, emb_b, mask_a, mask_b,
            return_attention=True,
            return_layer_reprs=return_layer_reprs,
        )

        prob     = out["prob"].item()
        attn_map = out["attn_map"].squeeze(0).cpu().numpy()   # (La, Lb)
        topk_pairs = self.model.get_topk_residue_pairs(
            out["attn_map"], k=topk, mask_a=mask_a, mask_b=mask_b
        )

        result = {"prob": prob, "attn_map": attn_map, "topk_pairs": topk_pairs}

        # [改动 2] 如果请求了 layer_reprs，附加到返回 dict
        if return_layer_reprs and "layer_reprs" in out:
            result["layer_reprs"] = {
                k: v.cpu() for k, v in out["layer_reprs"].items()
            }

        return result

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

    # -------------------------------------------------------------------------
    # [新增] 对比学习版可视化方法
    # -------------------------------------------------------------------------

    def plot_layer_umap(
        self,
        layer_reprs: Dict[int, Tensor],
        labels: List[int],
        save_dir: str,
        method: str = "tsne",
        n_components: int = 2,
    ) -> None:
        """
        将各层 pair_repr 用 t-SNE 或 UMAP 降维后可视化，
        区分正样本（label=1，蓝色）与负样本（label=0，橙色）。

        参数:
            layer_reprs: dict {layer_idx: (N, d)} Tensor，来自批量推理的 layer_reprs_all
            labels:      长度 N 的 list，0 或 1
            save_dir:    图片保存目录
            method:      "tsne" 或 "umap"（需安装 umap-learn）
            n_components: 降维维度，通常 2
        """
        if not HAS_VIZ:
            print("[警告] matplotlib 未安装，跳过 UMAP/t-SNE 绘图")
            return

        os.makedirs(save_dir, exist_ok=True)
        labels_np = np.array(labels)

        for layer_idx, reprs in sorted(layer_reprs.items()):
            X = reprs.numpy() if isinstance(reprs, Tensor) else reprs
            X = X.astype(np.float32)

            try:
                if method == "umap":
                    import umap as umap_lib
                    reducer = umap_lib.UMAP(n_components=n_components, random_state=42)
                    emb = reducer.fit_transform(X)
                else:
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X) - 1))
                    emb = reducer.fit_transform(X)
            except Exception as e:
                print(f"[警告] Layer {layer_idx} 降维失败: {e}")
                continue

            fig, ax = plt.subplots(figsize=(7, 6))
            colors = ["#e07b39" if l == 0 else "#4a90d9" for l in labels_np]
            ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=12, alpha=0.7, linewidths=0)

            # 手动图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#e07b39", label="Negative (label=0)"),
                Patch(facecolor="#4a90d9", label="Positive (label=1)"),
            ]
            ax.legend(handles=legend_elements, loc="best", fontsize=9)
            ax.set_title(f"Layer {layer_idx} pair repr ({method.upper()})", fontsize=11)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"layer{layer_idx}_{method}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"[interpret] Layer {layer_idx} {method.upper()} 图已保存: {save_path}")

    def plot_entropy_gap_profile(
        self,
        attn_stats_profile: dict,
        save_path: str,
    ) -> None:
        """
        绘制各层 attention entropy gap 折线图（正样本均值 - 负样本均值）。

        参数:
            attn_stats_profile: dict，结构：
                {
                  "pos": {layer_idx: {"mean_entropy": float, ...}, ...},
                  "neg": {layer_idx: {"mean_entropy": float, ...}, ...}
                }
            save_path: 图片保存路径
        """
        if not HAS_VIZ:
            print("[警告] matplotlib 未安装，跳过 entropy gap 绘图")
            return

        pos_stats = attn_stats_profile.get("pos", {})
        neg_stats = attn_stats_profile.get("neg", {})
        layers = sorted(set(list(pos_stats.keys()) + list(neg_stats.keys())))

        pos_entropy = [pos_stats.get(l, {}).get("mean_entropy", float("nan")) for l in layers]
        neg_entropy = [neg_stats.get(l, {}).get("mean_entropy", float("nan")) for l in layers]
        gap = [p - n for p, n in zip(pos_entropy, neg_entropy)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 左图：各层正/负 entropy 均值对比
        ax = axes[0]
        ax.plot(layers, pos_entropy, "o-", color="#4a90d9", label="Positive")
        ax.plot(layers, neg_entropy, "s-", color="#e07b39", label="Negative")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Attention Entropy")
        ax.set_title("Attention Entropy by Layer")
        ax.legend()
        ax.set_xticks(layers)

        # 右图：entropy gap
        ax2 = axes[1]
        bar_colors = ["#4a90d9" if g >= 0 else "#e07b39" for g in gap]
        ax2.bar(layers, gap, color=bar_colors, alpha=0.85)
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Entropy Gap (Pos - Neg)")
        ax2.set_title("Entropy Gap Profile")
        ax2.set_xticks(layers)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[interpret] Entropy gap profile 已保存: {save_path}")
