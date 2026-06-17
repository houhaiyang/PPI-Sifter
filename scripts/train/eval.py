"""
脚本: eval.py
功能: 评估 PPI-Sifter，计算 AUPRC/AUROC/F1/MCC 等指标
依赖: sklearn, torch, tqdm, pyyaml
运行: python scripts/train/eval.py
     （split / checkpoint 均在 configs/default.yaml 中配置）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger, load_checkpoint

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


def evaluate(model, data_loader, device, threshold: float = 0.5) -> dict:
    """
    在 data_loader 上推理并计算指标（供 train.py import 调用）。

    参数:
        model:       已构建好的 PPISifter 实例
        data_loader: DataLoader，collate_fn 返回 (emb_a, emb_b, mask_a, mask_b, labels)
        device:      torch.device
        threshold:   二分类阈值

    返回:
        dict: auprc, auroc, f1, mcc, precision, recall
    """
    model.eval()
    all_probs  = []
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

    return {
        "auprc":     average_precision_score(labels, probs),
        "auroc":     roc_auc_score(labels, probs),
        "f1":        f1_score(labels, preds, zero_division=0),
        "mcc":       matthews_corrcoef(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }


def main() -> None:
    """独立运行评估入口，配置完全来自 configs/default.yaml。"""
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    logger = get_logger("eval", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg  = cfg["data"]
    infer_cfg = cfg["infer"]
    split     = infer_cfg.get("split", "test")
    csv_path  = os.path.join(data_cfg["splits_dir"], f"{split}.csv")

    dataset = PPIDataset(
        csv_path=csv_path,
        hdf5_path=data_cfg["hdf5_path"],
        max_seq_len=data_cfg["max_seq_len"],
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
    logger.info(f"加载 checkpoint: {ckpt_path}")

    threshold = infer_cfg.get("threshold", 0.5)
    metrics   = evaluate(model, loader, device, threshold=threshold)
    logger.info(f"[{split.upper()}] 评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()