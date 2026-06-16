"""
脚本: eval.py
功能: 在指定数据集上评估 PPI-Sifter，计算 AUPRC/AUROC/F1/MCC 等指标
依赖: sklearn, torch, tqdm
运行: python scripts/train/eval.py \
          --config configs/default.yaml \
          --checkpoint outputs/checkpoints/best_auprc.pt \
          --split test
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, matthews_corrcoef, precision_score, recall_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint


def evaluate(model, data_loader, device, threshold=0.5) -> dict:
    """
    在 data_loader 上推理并计算指标。

    返回 dict: auprc, auroc, f1, mcc, precision, recall
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for emb_a, emb_b, mask_a, mask_b, labels in tqdm(
            data_loader, desc="评估中", leave=False
        ):
            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)
            out = model(emb_a, emb_b, mask_a, mask_b)
            all_probs.extend(out["prob"].cpu().tolist())
            all_labels.extend(labels.tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)
    f1    = f1_score(labels, preds, zero_division=0)
    mcc   = matthews_corrcoef(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)

    return {
        "auprc": auprc, "auroc": auroc,
        "f1": f1, "mcc": mcc,
        "precision": prec, "recall": rec,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("eval", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    dataset = PPIDataset(
        csv_path=os.path.join(data_cfg["splits_dir"], f"{args.split}.csv"),
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
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

    metrics = evaluate(model, loader, device, threshold=args.threshold)
    logger.info(f"[{args.split.upper()}] 评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
