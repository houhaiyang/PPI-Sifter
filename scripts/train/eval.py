"""
脚本: eval.py
功能: 评估 PPI-Sifter，计算 AUPRC/AUROC/F1/MCC 等指标
依赖: sklearn, torch, tqdm, pyyaml
运行: python scripts/train/eval.py
     （split / checkpoint 均在 configs/default.yaml 中配置）

[对比学习版改动]
  1. DataLoader 迭代改为 dict 解包（与新版 collate_fn 对齐）
  2. loss_fn 调用改为取 ["total"]（PPILoss 现在返回 dict）
  3. model.forward() 增加 return_layer_reprs=False（eval 阶段不需要导出）
  4. 新增 --对比分析版-- 提示：eval 独立运行时不依赖 contrast_head
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
from ppisifter.utils import set_seed, get_logger
from ppisifter.config import load_config

_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


def evaluate(model, data_loader, device, threshold: float = 0.5) -> dict:
    """
    在 data_loader 上推理并计算指标（供 train.py import 调用）。

    参数:
        model:       已构建好的 PPISifter 实例
        data_loader: DataLoader，collate_fn 返回 dict 格式 batch
        device:      torch.device
        threshold:   二分类阈值

    返回:
        dict: auprc, auroc, f1, mcc, precision, recall
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        # [改动 1] batch 从 tuple 解包改为 dict 解包，与新版 collate_fn 对齐
        for batch in tqdm(data_loader, desc="评估中", leave=False):
            emb_a  = batch["emb_a"].to(device)
            emb_b  = batch["emb_b"].to(device)
            mask_a = batch["mask_a"].to(device)
            mask_b = batch["mask_b"].to(device)
            labels = batch["label"]

            # [改动 2] 增加 return_layer_reprs=False，eval 阶段不导出中间层（节省显存）
            out = model(
                emb_a, emb_b, mask_a, mask_b,
                return_attention=False,
                return_layer_reprs=False,
            )
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

    cfg = load_config(_CFG_PATH)

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

    # [改动 3] 新版 checkpoint 包含 "model" key（由 train.py 对比版保存）
    # load_checkpoint 应优先读 ckpt["model"]，若不存在则兼容旧格式整体 state_dict
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        logger.info(f"加载 checkpoint (model key): {ckpt_path}，epoch={ckpt.get('epoch', '?')}")
    else:
        # 兼容旧版直接保存 state_dict 的格式
        model.load_state_dict(ckpt)
        logger.info(f"加载 checkpoint (直接 state_dict): {ckpt_path}")

    threshold = infer_cfg.get("threshold", 0.5)
    metrics   = evaluate(model, loader, device, threshold=threshold)

    logger.info(f"[{split.upper()}] 评估结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 可选：保存 metrics 到 JSON
    import json
    pred_dir = cfg["paths"]["pred_dir"]
    os.makedirs(pred_dir, exist_ok=True)
    out_json = os.path.join(pred_dir, f"eval_{split}.json")
    with open(out_json, "w", encoding="utf-8") as fj:
        json.dump(metrics, fj, indent=2, ensure_ascii=False)
    logger.info(f"Metrics 已保存: {out_json}")


if __name__ == "__main__":
    main()
