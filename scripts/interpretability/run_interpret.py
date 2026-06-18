"""
脚本: run_interpret.py
功能: 批量对测试集蛋白对导出 residue-level attention map 和 top-k residue pairs
     结果写入 outputs/interpret/<pair_id>_attn_matrix.csv
                               <pair_id>_topk_pairs.csv
                               summary.csv（所有对的 prob + 路径索引）
运行: python scripts/interpretability/run_interpret.py
     （所有参数在 configs/interpret.yaml 中配置，无命令行传参）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint

# ── 配置路径硬编码，所有参数来自 yaml ──────────────────────────────────────
_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "interpret.yaml",
)


def _pair_id(row: pd.Series) -> str:
    """生成蛋白对文件名前缀。"""
    return f"{row['protein_a']}__{row['protein_b']}"


def main() -> None:
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("run_interpret", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ── 数据 ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    interp_cfg = cfg["interpret"]
    split = data_cfg["split"]
    csv_path = os.path.join(data_cfg["splits_dir"], f"{split}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Split CSV 不存在: {csv_path}")

    pairs_df = pd.read_csv(csv_path)
    max_pairs = interp_cfg.get("max_pairs", -1)
    if max_pairs > 0:
        pairs_df = pairs_df.head(max_pairs)
        logger.info(f"debug 模式：只处理前 {max_pairs} 对")

    logger.info(f"共 {len(pairs_df)} 个蛋白对，split={split}")

    dataset = PPIDataset(
        csv_path=csv_path,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
        cache_size=0,          # 逐对推理，不需要大缓存
        inference=False,
    )
    # 若 max_pairs 截断了，对应截断 dataset
    if max_pairs > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(max_pairs, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=1,          # attention 导出固定 bs=1
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ── 模型 ───────────────────────────────────────────────────────────────
    m = cfg["model"]
    model = PPISifter(
        d_in=m["d_in"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_expansion=m["ffn_expansion"],
        dropout=m["dropout"],
    ).to(device)

    ckpt_path = cfg["checkpoint"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    load_checkpoint(ckpt_path, model, device=device)
    logger.info(f"加载 checkpoint: {ckpt_path}")
    model.eval()

    # ── 输出目录 ────────────────────────────────────────────────────────────
    out_dir = cfg["paths"]["interpret_dir"]
    os.makedirs(out_dir, exist_ok=True)

    topk = interp_cfg.get("topk", 20)
    pos_threshold = interp_cfg.get("pos_threshold", 0.5)
    save_matrix = interp_cfg.get("save_attn_matrix", True)

    summary_rows = []

    with torch.no_grad():
        for idx, (emb_a, emb_b, mask_a, mask_b, label) in enumerate(
            tqdm(loader, desc="导出 attention", total=len(loader))
        ):
            row = pairs_df.iloc[idx]
            pid = _pair_id(row)

            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)

            out = model(emb_a, emb_b, mask_a, mask_b, return_attention=True)
            prob = float(out["prob"][0].cpu())

            # ── attention map ─────────────────────────────────────────────
            attn_map = out["attn_map"][0].cpu().numpy()   # (La, Lb)

            # 只保留有效残基区域（去 padding）
            valid_a = int(mask_a[0].sum().item())
            valid_b = int(mask_b[0].sum().item())
            attn_crop = attn_map[:valid_a, :valid_b]

            attn_matrix_path = ""
            if save_matrix:
                attn_matrix_path = os.path.join(out_dir, f"{pid}__attn_matrix.csv")
                pd.DataFrame(attn_crop).to_csv(attn_matrix_path, index=False, header=False)

            # ── top-k residue pairs ───────────────────────────────────────
            topk_pairs = model.get_topk_residue_pairs(
                out["attn_map"][0].unsqueeze(0),
                k=topk,
                mask_a=mask_a,
                mask_b=mask_b,
            )
            topk_path = os.path.join(out_dir, f"{pid}__topk_pairs.csv")
            topk_df = pd.DataFrame(topk_pairs, columns=["res_a_idx", "res_b_idx", "attn_score"])
            topk_df.insert(0, "protein_b", row["protein_b"])
            topk_df.insert(0, "protein_a", row["protein_a"])
            topk_df.to_csv(topk_path, index=False)

            summary_rows.append({
                "protein_a":        row["protein_a"],
                "protein_b":        row["protein_b"],
                "label":            int(label[0].item()) if not model.training else -1,
                "prob":             round(prob, 6),
                "pred":             int(prob >= pos_threshold),
                "valid_len_a":      valid_a,
                "valid_len_b":      valid_b,
                "attn_matrix_path": attn_matrix_path,
                "topk_pairs_path":  topk_path,
            })

    # ── summary CSV ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"导出完成，共 {len(summary_df)} 对。summary -> {summary_path}")


if __name__ == "__main__":
    main()