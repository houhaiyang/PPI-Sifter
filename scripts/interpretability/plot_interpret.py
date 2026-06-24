"""
脚本: plot_interpret.py
功能: 可视化可解释性结果
     1. 每个蛋白对的 attention heatmap（按 prob 降序取 top-N）
     2. Entropy 正/负样本分布对比图（violin + box）
     3. Symmetry score 分布图
     4. Top-k residue pair attention score scatter
运行: python scripts/interpretability/plot_interpret.py
     （所有参数在 configs/interpret.yaml 中配置）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ppisifter.utils import set_seed, get_logger

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "interpret.yaml",
)


def plot_heatmap(
    attn_matrix_path: str,
    save_path: str,
    title: str,
    cmap: str = "YlOrRd",
    dpi: int = 200,
    max_len: int = 200,
) -> None:
    """绘制单对蛋白的 attention map 热图。"""
    mat = pd.read_csv(attn_matrix_path, header=None).values.astype(np.float32)
    # 截断显示（不影响计算，仅控制图像大小）
    mat = mat[:max_len, :max_len]
    la, lb = mat.shape
    fig_w = max(4, min(lb / 20 + 2, 16))
    fig_h = max(3, min(la / 20 + 2, 14))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        mat, ax=ax, cmap=cmap, cbar=True,
        xticklabels=False, yticklabels=False,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Protein B residues", fontsize=8)
    ax.set_ylabel("Protein A residues", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_entropy_comparison(
    result_df: pd.DataFrame,
    save_path: str,
    dpi: int = 200,
) -> None:
    """正/负样本 attention entropy 分布对比（violin + stripplot）。"""
    df = result_df[["label", "attn_entropy"]].copy()
    df["group"] = df["label"].map({1: "Positive (label=1)", 0: "Negative (label=0)"})
    df = df.dropna(subset=["attn_entropy"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data=df, x="group", y="attn_entropy",
        palette={"Positive (label=1)": "#e64c4c", "Negative (label=0)": "#4c7ae6"},
        inner="box", ax=ax, cut=0,
    )
    sns.stripplot(
        data=df, x="group", y="attn_entropy",
        color="black", alpha=0.25, size=2, jitter=True, ax=ax,
    )
    ax.set_title("Attention Entropy: Positive vs Negative", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized Entropy", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_symmetry_comparison(
    result_df: pd.DataFrame,
    save_path: str,
    dpi: int = 200,
) -> None:
    """正/负样本 symmetry score 分布对比。"""
    df = result_df[["label", "symmetry_score"]].copy()
    df["group"] = df["label"].map({1: "Positive (label=1)", 0: "Negative (label=0)"})
    df = df.dropna(subset=["symmetry_score"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data=df, x="group", y="symmetry_score",
        palette={"Positive (label=1)": "#e64c4c", "Negative (label=0)": "#4c7ae6"},
        inner="box", ax=ax, cut=0,
    )
    sns.stripplot(
        data=df, x="group", y="attn_entropy" if "attn_entropy" in df.columns else "symmetry_score",
        color="black", alpha=0.25, size=2, jitter=True, ax=ax,
    )
    # 修正 stripplot y 轴
    ax.set_title("Symmetry Score: Positive vs Negative", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Symmetry Score (lower = more symmetric)", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_symmetry_distribution(
    result_df: pd.DataFrame,
    save_path: str,
    dpi: int = 200,
) -> None:
    """修正版：正/负 symmetry score 分布（violin only）。"""
    df = result_df[["label", "symmetry_score"]].copy()
    df["group"] = df["label"].map({1: "Positive (label=1)", 0: "Negative (label=0)"})
    df = df.dropna(subset=["symmetry_score"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data=df, x="group", y="symmetry_score",
        palette={"Positive (label=1)": "#e64c4c", "Negative (label=0)": "#4c7ae6"},
        inner="box", ax=ax, cut=0,
    )
    sns.stripplot(
        data=df, x="group", y="symmetry_score",
        color="black", alpha=0.25, size=2, jitter=True, ax=ax,
    )
    ax.set_title("Attention Symmetry Score: Positive vs Negative", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Symmetry Score (Frobenius / size)", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_topk_scatter(
    interp_dir: str,
    summary_df: pd.DataFrame,
    save_path: str,
    dpi: int = 200,
    max_pairs: int = 50,
) -> None:
    """
    Top-k residue pair attention score 散点图：
    x = residue index in A，y = residue index in B，color = attn_score，
    每个蛋白对一个子图（最多 max_pairs 对）。
    """
    pos_df = summary_df[summary_df["label"] == 1].head(max_pairs)
    n = len(pos_df)
    if n == 0:
        return
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax_i, (_, sr) in enumerate(pos_df.iterrows()):
        ax = axes[ax_i]
        topk_path = sr["topk_pairs_path"]
        if not os.path.isfile(str(topk_path)):
            ax.axis("off")
            continue
        tk = pd.read_csv(topk_path)
        sc = ax.scatter(
            tk["res_a_idx"], tk["res_b_idx"],
            c=tk["attn_score"], cmap="YlOrRd", s=30, edgecolors="gray", linewidths=0.3,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{sr['protein_a'][:6]}|{sr['protein_b'][:6]}\np={sr['prob']:.2f}",
                     fontsize=7)
        ax.set_xlabel("Res A idx", fontsize=7)
        ax.set_ylabel("Res B idx", fontsize=7)
        ax.tick_params(labelsize=6)

    # 关闭多余子图
    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")

    plt.suptitle("Top-k Residue Pair Attention Scores (Positive Pairs)", fontsize=10, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("plot_interpret", log_dir=cfg["paths"]["log_dir"])

    interp_dir  = cfg["paths"]["interpret_dir"]
    figures_dir = cfg["paths"]["figures_dir"]
    p_cfg       = cfg["plot"]
    os.makedirs(figures_dir, exist_ok=True)

    # ── 读取 summary 和量化结果 ──────────────────────────────────────────────
    summary_path = os.path.join(interp_dir, "summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"summary.csv 不存在: {summary_path}\n"
            "请先运行 run_interpret.py"
        )
    summary_df = pd.read_csv(summary_path)

    quant_csv = cfg["quantify"]["output_csv"]
    if not os.path.isfile(quant_csv):
        raise FileNotFoundError(
            f"quantify_results.csv 不存在: {quant_csv}\n"
            "请先运行 quantify_interpret.py"
        )
    result_df = pd.read_csv(quant_csv)
    # topk_precision 列可能含空字符串
    result_df["topk_precision"] = pd.to_numeric(result_df["topk_precision"], errors="coerce")

    # ── 1. Attention heatmaps ────────────────────────────────────────────────
    heatmap_top_n = p_cfg.get("heatmap_top_n", 10)
    cmap          = p_cfg.get("heatmap_cmap", "YlOrRd")
    dpi           = p_cfg.get("heatmap_dpi", 200)
    max_len       = p_cfg.get("heatmap_max_len", 200)

    # 按 prob 降序取正例前 N
    heat_df = summary_df[summary_df["label"] == 1].sort_values("prob", ascending=False).head(heatmap_top_n)
    logger.info(f"绘制热图: {len(heat_df)} 张")
    for _, sr in tqdm(heat_df.iterrows(), desc="热图", total=len(heat_df)):
        mat_path = sr["attn_matrix_path"]
        if not os.path.isfile(str(mat_path)):
            continue
        pid = f"{sr['protein_a']}__{sr['protein_b']}"
        save_path = os.path.join(figures_dir, f"{pid}__heatmap.png")
        plot_heatmap(
            attn_matrix_path=mat_path,
            save_path=save_path,
            title=f"{sr['protein_a']} × {sr['protein_b']}  prob={sr['prob']:.3f}",
            cmap=cmap, dpi=dpi, max_len=max_len,
        )
    logger.info(f"热图写入: {figures_dir}/")

    # ── 2. Entropy 分布对比 ──────────────────────────────────────────────────
    entropy_fig = p_cfg.get("entropy_fig", os.path.join(figures_dir, "entropy_comparison.png"))
    plot_entropy_comparison(result_df, entropy_fig, dpi=dpi)
    logger.info(f"Entropy 分布图: {entropy_fig}")

    # ── 3. Symmetry 分布 ─────────────────────────────────────────────────────
    symmetry_fig = p_cfg.get("symmetry_fig", os.path.join(figures_dir, "symmetry_comparison.png"))
    plot_symmetry_distribution(result_df, symmetry_fig, dpi=dpi)
    logger.info(f"Symmetry 分布图: {symmetry_fig}")

    # ── 4. Top-k scatter ─────────────────────────────────────────────────────
    topk_fig = p_cfg.get("topk_scatter_fig", os.path.join(figures_dir, "topk_scatter.png"))
    plot_topk_scatter(
        interp_dir=interp_dir,
        summary_df=summary_df,
        save_path=topk_fig,
        dpi=dpi,
        max_pairs=heatmap_top_n * 2,
    )
    logger.info(f"Top-k scatter 图: {topk_fig}")
    logger.info("可视化全部完成。")


if __name__ == "__main__":
    main()