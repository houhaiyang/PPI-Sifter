"""
脚本: infer.py
功能: 批量推理，可选导出 attention map 与 top-k residue pairs
依赖: torch, h5py, pandas, pyyaml, tqdm, matplotlib
运行: python scripts/train/infer.py
     （pairs CSV / checkpoint / 输出目录均在 configs/default.yaml 中配置）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint
from ppisifter.interpret import AttentionInterpreter

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


def main() -> None:
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("infer", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg  = cfg["data"]
    infer_cfg = cfg["infer"]

    split     = infer_cfg.get("split", "test")
    pairs_csv = os.path.join(data_cfg["splits_dir"], f"{split}.csv")
    if not os.path.isfile(pairs_csv):
        raise FileNotFoundError(f"Pairs CSV 不存在: {pairs_csv}")

    dataset = PPIDataset(
        csv_path=pairs_csv,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
        inference=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=infer_cfg["batch_size"],
        shuffle=False,
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

    ckpt_path = infer_cfg.get("checkpoint", "")
    if not ckpt_path:
        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best_auprc.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    logger.info(f"加载 checkpoint: {ckpt_path}，推理对数: {len(dataset)}")

    threshold = infer_cfg.get("threshold", 0.5)
    out_dir   = cfg["paths"]["pred_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for emb_a, emb_b, mask_a, mask_b, _ in tqdm(loader, desc="推理中"):
            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)
            out   = model(emb_a, emb_b, mask_a, mask_b)
            probs = out["prob"].cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend([int(p >= threshold) for p in probs])

    pairs_df = pd.read_csv(pairs_csv)
    pairs_df["prob"] = all_probs
    pairs_df["pred"] = all_preds
    out_csv = os.path.join(out_dir, "predictions.csv")
    pairs_df.to_csv(out_csv, index=False)
    logger.info(f"推理完成，结果保存至: {out_csv}")

    # export_attn: true 时逐样本导出 attention map
    if infer_cfg.get("export_attn", False):
        topk        = infer_cfg.get("topk_residue_pairs", 20)
        attn_dir    = cfg["paths"]["interpret_dir"]
        os.makedirs(attn_dir, exist_ok=True)
        interpreter = AttentionInterpreter(model, device=str(device))

        single_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        for idx, (emb_a, emb_b, mask_a, mask_b, _) in enumerate(
            tqdm(single_loader, desc="导出 attention")
        ):
            row    = pairs_df.iloc[idx]
            pid_a  = str(row["protein_a"])
            pid_b  = str(row["protein_b"])
            result = interpreter.explain_pair(
                emb_a, emb_b, mask_a, mask_b, topk=topk
            )
            pair_id = f"{pid_a}__{pid_b}"
            interpreter.save_attn_heatmap(
                result["attn_map"],
                os.path.join(attn_dir, f"{pair_id}.png"),
                title=f"{pid_a} vs {pid_b} | prob={result['prob']:.4f}",
            )
            interpreter.save_topk_pairs(
                result["topk_pairs"],
                os.path.join(attn_dir, f"{pair_id}_topk.csv"),
                protein_a_id=pid_a,
                protein_b_id=pid_b,
            )
        logger.info(f"Attention 导出完成，保存至: {attn_dir}")


if __name__ == "__main__":
    main()