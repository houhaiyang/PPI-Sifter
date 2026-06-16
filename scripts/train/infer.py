"""
脚本: infer.py
功能: 批量推理并可选导出 attention map 与 top-k residue pairs
依赖: torch, h5py, pandas, yaml
运行: python scripts/train/infer.py \
          --config    configs/default.yaml \
          --checkpoint outputs/checkpoints/best_auprc.pt \
          --pairs     data/BIOGRID/pairs/test_pairs.csv \
          --out_dir   outputs/preds
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint
from ppisifter.interpret import AttentionInterpreter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs",      required=True)
    parser.add_argument("--out_dir",    default="outputs/preds")
    parser.add_argument("--export_attn", action="store_true",
                        help="是否导出 attention map 与 top-k residue pairs")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("infer", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    dataset = PPIDataset(
        csv_path=args.pairs,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
        inference=True,
    )
    loader = DataLoader(
        dataset, batch_size=cfg["infer"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=data_cfg["num_workers"],
    )

    m = cfg["model"]
    model = PPISifter(
        d_in=m["d_in"], d_model=m["d_model"], n_heads=m["n_heads"],
        n_layers=m["n_layers"], ffn_expansion=m["ffn_expansion"],
        dropout=m["dropout"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    pairs_df = pd.read_csv(args.pairs)
    threshold = cfg["infer"]["threshold"]
    os.makedirs(args.out_dir, exist_ok=True)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for emb_a, emb_b, mask_a, mask_b, _ in tqdm(loader, desc="推理中"):
            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)
            out = model(emb_a, emb_b, mask_a, mask_b)
            probs = out["prob"].cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend([int(p >= threshold) for p in probs])

    pairs_df["prob"]  = all_probs
    pairs_df["pred"]  = all_preds
    out_csv = os.path.join(args.out_dir, "predictions.csv")
    pairs_df.to_csv(out_csv, index=False)
    logger.info(f"推理完成，结果保存至: {out_csv}")

    # 可选：导出 attention map（逐样本）
    if args.export_attn:
        interpreter = AttentionInterpreter(model, device=str(device))
        topk = cfg["infer"]["topk_residue_pairs"]
        attn_dir = cfg["paths"]["interpret_dir"]
        os.makedirs(attn_dir, exist_ok=True)

        single_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        for idx, (emb_a, emb_b, mask_a, mask_b, _) in enumerate(
            tqdm(single_loader, desc="导出 attention")
        ):
            row = pairs_df.iloc[idx]
            pid_a = str(row["protein_a"])
            pid_b = str(row["protein_b"])
            result = interpreter.explain_pair(
                emb_a, emb_b, mask_a, mask_b, topk=topk
            )
            pair_id = f"{pid_a}__{pid_b}"
            # 保存热图
            interpreter.save_attn_heatmap(
                result["attn_map"],
                os.path.join(attn_dir, f"{pair_id}.png"),
                title=f"{pid_a} vs {pid_b} | prob={result['prob']:.4f}",
            )
            # 保存 top-k pairs
            interpreter.save_topk_pairs(
                result["topk_pairs"],
                os.path.join(attn_dir, f"{pair_id}_topk.csv"),
                protein_a_id=pid_a, protein_b_id=pid_b,
            )
        logger.info(f"Attention 导出完成，保存至: {attn_dir}")


if __name__ == "__main__":
    main()
