"""
脚本: export_attention.py
功能: 与 run_interpret.py 的轻量版本；单独导出指定蛋白对的 attention map，
     供论文 case study 使用（直接修改 yaml 中的 interpret.max_pairs 控制对数）。
     本文件是原 export_attention.py 的对齐重写版，完全兼容当前 model/data API。
运行: python scripts/interpretability/export_attention.py
     （所有参数在 configs/interpret.yaml 中配置）
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "interpret.yaml",
)


def save_heatmap(mat: np.ndarray, path: str, title: str, cmap: str = "YlOrRd", dpi: int = 200):
    la, lb = mat.shape
    fw = max(4, min(lb / 20 + 2, 16))
    fh = max(3, min(la / 20 + 2, 14))
    fig, ax = plt.subplots(figsize=(fw, fh))
    sns.heatmap(mat, ax=ax, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Protein B residues", fontsize=8)
    ax.set_ylabel("Protein A residues", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("export_attention", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg   = cfg["data"]
    interp_cfg = cfg["interpret"]
    split      = data_cfg["split"]
    csv_path   = os.path.join(data_cfg["splits_dir"], f"{split}.csv")
    pairs_df   = pd.read_csv(csv_path)

    max_pairs = interp_cfg.get("max_pairs", 20)
    if max_pairs > 0:
        pairs_df = pairs_df.head(max_pairs)

    dataset = PPIDataset(
        csv_path=csv_path,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
        cache_size=0,
        inference=False,
    )
    if max_pairs > 0:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(max_pairs, len(dataset)))))

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn,
        num_workers=data_cfg.get("num_workers", 0),
    )

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
    logger.info(f"加载 checkpoint: {cfg['checkpoint']}")

    out_dir    = cfg["paths"]["interpret_dir"]
    fig_dir    = cfg["paths"]["figures_dir"]
    topk       = interp_cfg.get("topk", 20)
    cmap       = cfg["plot"].get("heatmap_cmap", "YlOrRd")
    dpi        = cfg["plot"].get("heatmap_dpi", 200)
    max_len    = cfg["plot"].get("heatmap_max_len", 200)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (emb_a, emb_b, mask_a, mask_b, label) in enumerate(
            tqdm(loader, desc="export attention", total=len(loader))
        ):
            row  = pairs_df.iloc[idx]
            prot_a = str(row["protein_a"])
            prot_b = str(row["protein_b"])
            pid  = f"{prot_a}__{prot_b}"

            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)

            out  = model(emb_a, emb_b, mask_a, mask_b, return_attention=True)
            prob = float(out["prob"][0].cpu())

            valid_a = int(mask_a[0].sum().item())
            valid_b = int(mask_b[0].sum().item())

            attn_map = out["attn_map"][0].cpu().numpy()[:valid_a, :valid_b]

            # 保存 attn matrix CSV
            mat_path = os.path.join(out_dir, f"{pid}__attn_matrix.csv")
            pd.DataFrame(attn_map).to_csv(mat_path, index=False, header=False)

            # 保存 top-k pairs CSV
            topk_pairs = model.get_topk_residue_pairs(
                out["attn_map"][0].unsqueeze(0), k=topk,
                mask_a=mask_a, mask_b=mask_b,
            )
            tk_df = pd.DataFrame(topk_pairs, columns=["res_a_idx", "res_b_idx", "attn_score"])
            tk_df.insert(0, "protein_b", prot_b)
            tk_df.insert(0, "protein_a", prot_a)
            tk_df.to_csv(os.path.join(out_dir, f"{pid}__topk_pairs.csv"), index=False)

            # 保存热图
            display_mat = attn_map[:max_len, :max_len]
            fig_path = os.path.join(fig_dir, f"{pid}__heatmap.png")
            save_heatmap(
                display_mat, fig_path,
                title=f"{prot_a} × {prot_b}  prob={prob:.3f}  label={int(label[0].item())}",
                cmap=cmap, dpi=dpi,
            )
            logger.info(f"[{idx+1}] {pid}  prob={prob:.3f}  La={valid_a}  Lb={valid_b}")

    logger.info(f"导出完成 → {out_dir} / {fig_dir}")


if __name__ == "__main__":
    main()