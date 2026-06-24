"""
脚本: quantify_interpret.py
功能: 对 run_interpret.py 的输出计算定量可解释性指标：
     1. Attention entropy（正 vs 负样本对比）
     2. Symmetry consistency score（SAB 与 SBA^T 的差异）
     3. Top-k residue pair precision（需提供 PDB 界面标注 CSV）
运行: python scripts/interpretability/quantify_interpret.py
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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "interpret.yaml",
)


def attn_entropy(attn_map: np.ndarray) -> float:
    """计算 attention map 的归一化 Shannon entropy（值越小=越聚焦）。"""
    flat = attn_map.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-12)
    flat = flat[flat > 0]
    entropy = -np.sum(flat * np.log(flat + 1e-12))
    # 归一化到 [0,1]
    max_entropy = np.log(len(attn_map.flatten()))
    return float(entropy / (max_entropy + 1e-12))


def symmetry_score(attn_ab: np.ndarray, attn_ba: np.ndarray) -> float:
    """
    计算 A->B 与 B->A attention map 的对称一致性。
    attn_ab: (La, Lb)  attn_ba: (Lb, La)
    返回 Frobenius 距离 / (La*Lb)，越小越对称。
    """
    diff = attn_ab - attn_ba.T
    return float(np.linalg.norm(diff, "fro") / (diff.size + 1e-12))


def topk_precision(
    topk_df: pd.DataFrame,
    interface_set: set,
) -> float:
    """
    计算 top-k residue pairs 中命中已知界面残基对的比例。
    interface_set: set of (res_a_idx, res_b_idx) 整数元组
    """
    if len(topk_df) == 0 or len(interface_set) == 0:
        return float("nan")
    hits = sum(
        1 for _, r in topk_df.iterrows()
        if (int(r["res_a_idx"]), int(r["res_b_idx"])) in interface_set
    )
    return hits / len(topk_df)


def main() -> None:
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("quantify_interpret", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_cfg     = cfg["quantify"]
    interp_dir = cfg["paths"]["interpret_dir"]
    out_csv    = q_cfg["output_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # ── 读取 summary ────────────────────────────────────────────────────────
    summary_path = os.path.join(interp_dir, "summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"summary.csv 不存在: {summary_path}\n"
            "请先运行 scripts/interpretability/run_interpret.py"
        )
    summary_df = pd.read_csv(summary_path)
    logger.info(f"读取 summary.csv，共 {len(summary_df)} 对")

    # ── 加载模型（用于同时导出 attn_ab / attn_ba 做对称性计算）──────────────
    m = cfg["model"]
    model = PPISifter(
        d_in=m["d_in"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_expansion=m["ffn_expansion"],
        dropout=m["dropout"],
    ).to(device)
    load_checkpoint(cfg["checkpoint"], model, device=device)
    model.eval()

    data_cfg = cfg["data"]
    split    = data_cfg["split"]
    csv_path = os.path.join(data_cfg["splits_dir"], f"{split}.csv")
    dataset  = PPIDataset(
        csv_path=csv_path,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
        cache_size=0,
        inference=False,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ── 加载 PDB 界面标注（可选）────────────────────────────────────────────
    pdb_csv = q_cfg.get("pdb_interface_csv", "")
    pdb_interface_map: dict = {}    # (prot_a, prot_b) -> set of (res_a, res_b)
    if pdb_csv and os.path.isfile(pdb_csv):
        pdb_df = pd.read_csv(pdb_csv)
        for _, r in pdb_df.iterrows():
            key = (str(r["protein_a"]), str(r["protein_b"]))
            pdb_interface_map.setdefault(key, set()).add(
                (int(r["res_a"]), int(r["res_b"]))
            )
        logger.info(f"加载 PDB 界面标注: {len(pdb_interface_map)} 个蛋白对")
    else:
        logger.info("未提供 PDB 界面标注，跳过 top-k overlap 计算")

    # ── 逐对计算指标 ─────────────────────────────────────────────────────────
    rows = []
    with torch.no_grad():
        for idx, (emb_a, emb_b, mask_a, mask_b, label) in enumerate(
            tqdm(loader, desc="量化中", total=len(loader))
        ):
            if idx >= len(summary_df):
                break
            sr = summary_df.iloc[idx]

            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)

            out    = model(emb_a, emb_b, mask_a, mask_b, return_attention=True)
            prob   = float(out["prob"][0].cpu())

            valid_a = int(mask_a[0].sum().item())
            valid_b = int(mask_b[0].sum().item())

            # attn_map（对称融合版）
            attn_map = out["attn_map"][0].cpu().numpy()[:valid_a, :valid_b]
            # 原始单向（用于对称性计算）
            attn_ab  = out["attn_ab"][0].cpu().numpy()[:valid_a, :valid_b]
            attn_ba  = out["attn_ba"][0].cpu().numpy()[:valid_b, :valid_a]

            # 1. Entropy
            ent = attn_entropy(attn_map) if q_cfg.get("entropy_compare", True) else float("nan")

            # 2. Symmetry
            sym = (
                symmetry_score(attn_ab, attn_ba)
                if q_cfg.get("symmetry_check", True)
                else float("nan")
            )

            # 3. Top-k precision（若有 PDB 标注）
            prot_a = str(sr["protein_a"])
            prot_b = str(sr["protein_b"])
            topk_path = sr["topk_pairs_path"]
            topk_prec = float("nan")
            if pdb_interface_map and os.path.isfile(str(topk_path)):
                iface = pdb_interface_map.get((prot_a, prot_b), set())
                if not iface:
                    iface = pdb_interface_map.get((prot_b, prot_a), set())
                if iface:
                    tk_df = pd.read_csv(topk_path)
                    topk_prec = topk_precision(tk_df, iface)

            rows.append({
                "protein_a":      prot_a,
                "protein_b":      prot_b,
                "label":          int(label[0].item()),
                "prob":           round(prob, 6),
                "valid_len_a":    valid_a,
                "valid_len_b":    valid_b,
                "attn_entropy":   round(ent, 6),
                "symmetry_score": round(sym, 6),
                "topk_precision": round(topk_prec, 6) if not np.isnan(topk_prec) else "",
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False)
    logger.info(f"量化结果写入: {out_csv}")

    # ── 统计摘要 ─────────────────────────────────────────────────────────────
    pos = result_df[result_df["label"] == 1]
    neg = result_df[result_df["label"] == 0]
    logger.info("=== Attention Entropy ===")
    logger.info(f"  正样本: mean={pos['attn_entropy'].mean():.4f}  std={pos['attn_entropy'].std():.4f}  n={len(pos)}")
    logger.info(f"  负样本: mean={neg['attn_entropy'].mean():.4f}  std={neg['attn_entropy'].std():.4f}  n={len(neg)}")
    if len(pos) > 0 and len(neg) > 0:
        stat, pval = scipy.stats.mannwhitneyu(
            pos["attn_entropy"].dropna(), neg["attn_entropy"].dropna(),
            alternative="less"
        )
        logger.info(f"  Mann-Whitney U 检验 (正<负): U={stat:.1f}  p={pval:.4e}")

    logger.info("=== Symmetry Score ===")
    logger.info(f"  正样本: mean={pos['symmetry_score'].mean():.4f}  std={pos['symmetry_score'].std():.4f}")
    logger.info(f"  负样本: mean={neg['symmetry_score'].mean():.4f}  std={neg['symmetry_score'].std():.4f}")

    if pdb_interface_map:
        valid_prec = result_df["topk_precision"].replace("", float("nan")).astype(float).dropna()
        logger.info(f"=== Top-k Precision（有 PDB 标注的对）===")
        logger.info(f"  mean={valid_prec.mean():.4f}  std={valid_prec.std():.4f}  n={len(valid_prec)}")


if __name__ == "__main__":
    main()