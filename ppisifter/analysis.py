"""
PPI-Sifter 对比学习机制分析模块
功能:
  1. layer_separability      - 各层线性可分性（线性 probe AUROC）
  2. partner_shift_analysis  - 固定 anchor，真实/负 partner 导致的表示偏移
  3. attention_stats_profile - 各层 entropy gap、symmetry error
  4. seed_stability          - 跨 seed top-k 稳定性
"""

import torch
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA


def layer_separability(
    layer_reprs: Dict[int, np.ndarray],
    labels: np.ndarray,
    max_iter: int = 200,
) -> Dict[int, float]:
    """
    对每一层 pair repr 训练线性 Logistic Regression probe，
    报告 AUROC 作为该层表示的可分性度量。

    参数:
        layer_reprs: {layer_idx: (N, d)} numpy array
        labels:      (N,) 0/1
        max_iter:    logistic regression 最大迭代次数

    返回:
        {layer_idx: auroc}
    """
    results = {}
    for l_idx, reprs in layer_reprs.items():
        scaler = StandardScaler()
        X = scaler.fit_transform(reprs)
        try:
            clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs")
            clf.fit(X, labels)
            probs = clf.predict_proba(X)[:, 1]
            auroc = roc_auc_score(labels, probs)
        except Exception:
            auroc = float("nan")
        results[l_idx] = auroc
    return results


def partner_shift_analysis(
    layer_reprs_pos: Dict[int, np.ndarray],
    layer_reprs_neg: Dict[int, np.ndarray],
) -> Dict[int, Dict[str, float]]:
    """
    计算每一层正/负 partner 导致的 anchor 表示偏移量。
    layer_reprs_pos 与 layer_reprs_neg 需行对齐（同一 anchor，不同 partner）。

    返回:
        {layer_idx: {"mean_shift": float, "std_shift": float}}
    """
    results = {}
    for l_idx in layer_reprs_pos:
        if l_idx not in layer_reprs_neg:
            continue
        pos = layer_reprs_pos[l_idx]   # (N, d)
        neg = layer_reprs_neg[l_idx]   # (N, d)
        shifts = np.linalg.norm(pos - neg, axis=-1)  # (N,)
        results[l_idx] = {
            "mean_shift": float(shifts.mean()),
            "std_shift":  float(shifts.std()),
        }
    return results


def attention_stats_profile(
    attn_stats_list: List[Dict[int, Dict[str, float]]],
    labels: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    汇总多个样本的 per-layer attention statistics，
    分别计算正负样本的 entropy 均值，报告 entropy gap。

    参数:
        attn_stats_list: 每个样本的 attn_stats dict 列表
        labels:          (N,) 0/1

    返回:
        {layer_idx: {"entropy_gap": float, "sym_err_mean": float}}
    """
    if not attn_stats_list:
        return {}

    n_layers = max(attn_stats_list[0].keys()) + 1
    results = {}
    for l_idx in range(n_layers):
        ent_pos, ent_neg, sym_errs = [], [], []
        for stats, lbl in zip(attn_stats_list, labels):
            if l_idx not in stats:
                continue
            e = stats[l_idx]["entropy_ab"]
            s = stats[l_idx]["sym_err"]
            (ent_pos if lbl == 1 else ent_neg).append(e)
            sym_errs.append(s)
        gap = float(np.mean(ent_pos) - np.mean(ent_neg)) if ent_pos and ent_neg else float("nan")
        results[l_idx] = {
            "entropy_pos":  float(np.mean(ent_pos)) if ent_pos else float("nan"),
            "entropy_neg":  float(np.mean(ent_neg)) if ent_neg else float("nan"),
            "entropy_gap":  gap,
            "sym_err_mean": float(np.mean(sym_errs)) if sym_errs else float("nan"),
        }
    return results


def compute_topk_overlap(
    topk_a: List[Tuple[int, int, float]],
    topk_b: List[Tuple[int, int, float]],
) -> float:
    """计算两个 top-k residue pairs 列表的 Jaccard overlap（不含 score）。"""
    set_a = {(i, j) for i, j, _ in topk_a}
    set_b = {(i, j) for i, j, _ in topk_b}
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def seed_stability(
    topk_results_per_seed: Dict[int, List[List[Tuple[int, int, float]]]],
) -> Dict[str, float]:
    """
    计算不同 seed 下 top-k pairs 的平均 Jaccard overlap。

    参数:
        topk_results_per_seed: {seed: [topk_pairs_sample_0, topk_pairs_sample_1, ...]}

    返回:
        {"mean_overlap": float, "std_overlap": float}
    """
    seeds = list(topk_results_per_seed.keys())
    if len(seeds) < 2:
        return {"mean_overlap": 1.0, "std_overlap": 0.0}

    overlaps = []
    n_samples = len(topk_results_per_seed[seeds[0]])
    for i in range(n_samples):
        for s1 in range(len(seeds)):
            for s2 in range(s1 + 1, len(seeds)):
                ov = compute_topk_overlap(
                    topk_results_per_seed[seeds[s1]][i],
                    topk_results_per_seed[seeds[s2]][i],
                )
                overlaps.append(ov)
    return {
        "mean_overlap": float(np.mean(overlaps)),
        "std_overlap":  float(np.std(overlaps)),
    }


def pca_layer_repr(
    layer_reprs: np.ndarray,
    n_components: int = 50,
) -> np.ndarray:
    """
    在高维 pair_repr 上先做 PCA 降维，再传给下游分析（节省内存）。
    """
    if layer_reprs.shape[1] <= n_components:
        return layer_reprs
    pca = PCA(n_components=n_components)
    return pca.fit_transform(layer_reprs)