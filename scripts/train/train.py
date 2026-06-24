"""
PPI-Sifter 训练脚本（对比学习版）
所有超参均通过 configs/default.yaml 配置，无命令行参数
"""

import os
import sys
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

# 确保 ppisifter 可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ppisifter.config import load_config
from ppisifter.model import PPISifter
from ppisifter.losses import PPILoss
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger
from ppisifter.contrast import LayerwiseContrastHead

CONFIG_PATH = "configs/default.yaml"


def build_contrast_head(cfg, model_cfg):
    """根据配置构建对比头，contrast.enabled=false 时返回 None。"""
    c = cfg.get("contrast", {})
    if not c.get("enabled", False):
        return None
    n_layers     = model_cfg["n_layers"]
    d_model      = model_cfg["d_model"]
    d_in         = d_model * 4  # pair_repr = [sa, sb, sa+sb, |sa-sb|]
    active_layers = c.get("active_layers", [n_layers - 1])
    # 支持 -1 表示最后一层
    active_layers = [l if l >= 0 else n_layers + l for l in active_layers]
    head = LayerwiseContrastHead(
        n_layers=n_layers,
        d_in=d_in,
        d_proj=c.get("d_proj", 128),
        active_layers=active_layers,
        dropout=model_cfg.get("dropout", 0.1),
    )
    head.loss_fn.temperature = c.get("temperature", 0.07)
    return head


def train_one_epoch(model, contrast_head, loss_fn, loader, optimizer, scaler, cfg, device):
    model.train()
    if contrast_head:
        contrast_head.train()

    total_loss_sum = 0.0
    all_probs, all_labels = [], []
    return_attn  = cfg["train"].get("return_attention", True)
    return_layer = cfg["train"].get("return_layer_reprs", True) and contrast_head is not None

    lambda_contrast = cfg.get("contrast", {}).get("lambda_contrast", 0.1)
    layer_weights   = cfg.get("contrast", {}).get("layer_weights", None)

    for batch in loader:
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        mask_a = batch["mask_a"].to(device)
        mask_b = batch["mask_b"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg["train"].get("fp16", False)):
            out = model(
                emb_a, emb_b, mask_a, mask_b,
                return_attention=return_attn,
                return_layer_reprs=return_layer,
            )
            # 对比损失
            contrast_loss = None
            if contrast_head is not None and return_layer:
                layer_reprs = out["layer_reprs"]
                contrast_loss = contrast_head(
                    layer_reprs, labels.long(),
                    layer_weights=layer_weights,
                )

            loss_dict = loss_fn(
                logits=out["logits"],
                targets=labels,
                attn_ab=out.get("attn_ab"),
                attn_ba=out.get("attn_ba"),
                contrast_loss=contrast_loss,
            )
            loss = loss_dict["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"].get("grad_clip", 1.0)
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item()
        all_probs.extend(out["prob"].detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    auprc = average_precision_score(all_labels, all_probs)
    return total_loss_sum / len(loader), auprc


@torch.no_grad()
def evaluate(model, contrast_head, loss_fn, loader, cfg, device):
    model.eval()
    if contrast_head:
        contrast_head.eval()

    all_probs, all_labels = [], []
    return_attn  = cfg["train"].get("return_attention", True)
    return_layer = cfg["train"].get("return_layer_reprs", True) and contrast_head is not None

    for batch in loader:
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        mask_a = batch["mask_a"].to(device)
        mask_b = batch["mask_b"].to(device)
        labels = batch["label"].float().to(device)

        out = model(emb_a, emb_b, mask_a, mask_b,
                    return_attention=return_attn,
                    return_layer_reprs=return_layer)
        all_probs.extend(out["prob"].cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    auprc = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)
    return auprc, auroc


def main():
    cfg    = load_config(CONFIG_PATH)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(cfg["project"]["seed"])
    logger = get_logger("train", log_dir=cfg["paths"]["log_dir"])
    logger.info(f"Device: {device}")

    # 数据
    train_set  = PPIDataset(cfg, split="train")
    val_set    = PPIDataset(cfg, split="valid")
    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
        prefetch_factor=cfg["data"].get("prefetch_factor", 2),
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["infer"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
    )

    # 模型
    model = PPISifter(**cfg["model"]).to(device)
    contrast_head = build_contrast_head(cfg, cfg["model"])
    if contrast_head:
        contrast_head = contrast_head.to(device)
        logger.info(f"对比头已启用，active_layers={cfg['contrast']['active_layers']}")
    else:
        logger.info("对比头未启用（contrast.enabled=false）")

    # 损失
    loss_fn = PPILoss(
        pos_weight      = cfg["train"]["pos_weight"],
        focal_gamma     = cfg["train"]["focal_gamma"],
        lambda_focal    = cfg["train"]["lambda_focal"],
        lambda_sparse   = cfg["train"]["lambda_sparse"],
        lambda_sym      = cfg["train"]["lambda_sym"],
        lambda_contrast = cfg.get("contrast", {}).get("lambda_contrast", 0.0),
    )

    # 优化器
    params = list(model.parameters())
    if contrast_head:
        params += list(contrast_head.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("fp16", False))

    # 训练循环
    ckpt_dir    = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auprc  = 0.0
    patience    = cfg["train"]["early_stop_patience"]
    no_improve  = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_auprc = train_one_epoch(
            model, contrast_head, loss_fn, train_loader, optimizer, scaler, cfg, device
        )
        scheduler.step()

        if epoch % cfg["train"].get("val_interval", 1) == 0:
            val_auprc, val_auroc = evaluate(
                model, contrast_head, loss_fn, val_loader, cfg, device
            )
            logger.info(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_auprc={train_auprc:.4f}"
                f" | val_auprc={val_auprc:.4f} val_auroc={val_auroc:.4f}"
            )

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                no_improve = 0
                ckpt = {"model": model.state_dict(), "epoch": epoch, "val_auprc": val_auprc}
                if contrast_head:
                    ckpt["contrast_head"] = contrast_head.state_dict()
                torch.save(ckpt, ckpt_dir / "best_auprc.pt")
                logger.info(f"  ✓ 保存最优 checkpoint（val_auprc={best_auprc:.4f}）")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping（patience={patience}）")
                    break

    logger.info(f"训练完成，best_val_auprc={best_auprc:.4f}")


if __name__ == "__main__":
    main()