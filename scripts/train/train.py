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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ppisifter.config import load_config
from ppisifter.model import PPISifter
from ppisifter.losses import PPILoss
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.utils import set_seed, get_logger
from ppisifter.contrast import LayerwiseContrastHead

CONFIG_PATH = "configs/default.yaml"


# ── 辅助：构建 Dataset（显式传参，不传整个 cfg）─────────────────────────────
def _build_dataset(cfg: dict, split: str, inference: bool = False) -> PPIDataset:
    data_cfg = cfg["data"]
    csv_path = os.path.join(data_cfg["splits_dir"], f"{split}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"split CSV 不存在: {csv_path}")
    return PPIDataset(
        csv_path    = csv_path,
        hdf5_path   = data_cfg["hdf5_path"],
        max_seq_len = data_cfg.get("max_seq_len", 512),
        cache_size  = data_cfg.get("cache_size", 8192),
        inference   = inference,
    )


def build_contrast_head(cfg: dict, model_cfg: dict):
    c = cfg.get("contrast", {})
    if not c.get("enabled", False):
        return None
    n_layers      = model_cfg["n_layers"]
    d_model       = model_cfg["d_model"]
    d_in          = d_model * 4
    active_layers = c.get("active_layers", [n_layers - 1])
    active_layers = [l if l >= 0 else n_layers + l for l in active_layers]
    head = LayerwiseContrastHead(
        n_layers      = n_layers,
        d_in          = d_in,
        d_proj        = c.get("d_proj", 128),
        active_layers = active_layers,
        dropout       = model_cfg.get("dropout", 0.1),
    )
    head.loss_fn.temperature = c.get("temperature", 0.07)
    return head


def train_one_epoch(
    model, contrast_head, loss_fn, loader, optimizer, scaler, cfg, device
):
    model.train()
    if contrast_head is not None:
        contrast_head.train()

    total_loss_sum = 0.0
    all_probs, all_labels = [], []

    return_attn  = cfg["train"].get("return_attention", True)
    # ★ 仅当 contrast_head 存在时才导出 layer_reprs
    return_layer = (
        cfg["train"].get("return_layer_reprs", True)
        and contrast_head is not None
    )
    layer_weights   = cfg.get("contrast", {}).get("layer_weights", None)
    grad_accum      = cfg["train"].get("grad_accum_steps", 1)

    # ★ 所有可训练参数（含 contrast_head），用于 clip_grad_norm_
    all_params = list(model.parameters())
    if contrast_head is not None:
        all_params += list(contrast_head.parameters())

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        # ★ collate_fn 已返回 dict
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        mask_a = batch["mask_a"].to(device)
        mask_b = batch["mask_b"].to(device)
        labels = batch["label"].float().to(device)

        with torch.cuda.amp.autocast(enabled=cfg["train"].get("fp16", False)):
            out = model(
                emb_a, emb_b, mask_a, mask_b,
                return_attention   = return_attn,
                return_layer_reprs = return_layer,
            )
            contrast_loss = None
            if contrast_head is not None and return_layer and "layer_reprs" in out:
                contrast_loss = contrast_head(
                    out["layer_reprs"], labels.long(),
                    layer_weights=layer_weights,
                )
            loss_dict = loss_fn(
                logits        = out["logits"],
                targets       = labels,
                attn_ab       = out.get("attn_ab"),
                attn_ba       = out.get("attn_ba"),
                contrast_loss = contrast_loss,
            )
            # ★ 梯度累积：除以 grad_accum 后 backward
            loss = loss_dict["total"] / grad_accum

        scaler.scale(loss).backward()

        # ★ 每 grad_accum 步才真正 step
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                all_params, cfg["train"].get("grad_clip", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss_sum += loss_dict["total"].item()
        all_probs.extend(out["prob"].detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    auprc = average_precision_score(all_labels, all_probs)
    return total_loss_sum / max(len(loader), 1), auprc


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    """
    验证集评估。★ 去除 contrast_head/loss_fn/cfg 参数，val 阶段不需要。
    """
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        emb_a  = batch["emb_a"].to(device)
        emb_b  = batch["emb_b"].to(device)
        mask_a = batch["mask_a"].to(device)
        mask_b = batch["mask_b"].to(device)
        labels = batch["label"].float()

        out = model(
            emb_a, emb_b, mask_a, mask_b,
            return_attention   = False,
            return_layer_reprs = False,   # ★ val 阶段不需要 layer_reprs
        )
        all_probs.extend(out["prob"].cpu().tolist())
        all_labels.extend(labels.tolist())

    auprc = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)
    return auprc, auroc


def main():
    cfg    = load_config(CONFIG_PATH)
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    set_seed(cfg["project"]["seed"])
    logger = get_logger("train", log_dir=cfg["paths"]["log_dir"])
    logger.info(f"Device: {device}")

    # ★ 显式传参构建 Dataset
    train_set    = _build_dataset(cfg, split="train")
    val_set      = _build_dataset(cfg, split="valid")
    train_loader = DataLoader(
        train_set,
        batch_size  = cfg["train"]["batch_size"],
        shuffle     = True,
        num_workers = cfg["data"].get("num_workers", 4),
        collate_fn  = collate_fn,
        pin_memory  = True,
        prefetch_factor    = cfg["data"].get("prefetch_factor", 2),
        persistent_workers = (cfg["data"].get("num_workers", 4) > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = cfg["infer"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["data"].get("num_workers", 4),
        collate_fn  = collate_fn,
        pin_memory  = True,
        persistent_workers = (cfg["data"].get("num_workers", 4) > 0),
    )

    # 模型
    m = cfg["model"]
    model = PPISifter(
        d_in          = m["d_in"],
        d_model       = m["d_model"],
        n_heads       = m["n_heads"],
        n_layers      = m["n_layers"],
        ffn_expansion = m["ffn_expansion"],
        dropout       = m["dropout"],
    ).to(device)

    contrast_head = build_contrast_head(cfg, m)
    if contrast_head is not None:
        contrast_head = contrast_head.to(device)
        logger.info(f"对比头已启用，active_layers={cfg['contrast']['active_layers']}")
    else:
        logger.info("对比头未启用（contrast.enabled=false 或 contrast 块缺失）")

    # 断点续训
    resume_path = cfg["train"].get("resume_checkpoint", "")
    start_epoch = 1
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if contrast_head is not None and "contrast_head" in ckpt:
            contrast_head.load_state_dict(ckpt["contrast_head"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"从 {resume_path} 恢复训练，从 epoch {start_epoch} 继续")

    # 损失
    c_cfg   = cfg.get("contrast", {})
    loss_fn = PPILoss(
        pos_weight      = cfg["train"].get("pos_weight", 2.0),
        focal_gamma     = cfg["train"].get("focal_gamma", 2.0),
        lambda_focal    = cfg["train"].get("lambda_focal", 0.5),
        lambda_sparse   = cfg["train"].get("lambda_sparse", 0.01),
        lambda_sym      = cfg["train"].get("lambda_sym", 0.01),
        lambda_contrast = c_cfg.get("lambda_contrast", 0.0),
    )

    # 优化器 + 调度器
    params = list(model.parameters())
    if contrast_head is not None:
        params += list(contrast_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr           = cfg["train"]["lr"],
        weight_decay = cfg["train"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("fp16", False))

    # 训练循环
    ckpt_dir   = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auprc = 0.0
    patience   = cfg["train"].get("early_stop_patience", 8)
    no_improve = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        train_loss, train_auprc = train_one_epoch(
            model, contrast_head, loss_fn, train_loader,
            optimizer, scaler, cfg, device
        )
        scheduler.step()

        if epoch % cfg["train"].get("val_interval", 1) == 0:
            # ★ 简化调用：只传 model, loader, device
            val_auprc, val_auroc = evaluate(model, val_loader, device)
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:03d} | lr={lr_now:.2e} "
                f"train_loss={train_loss:.4f} train_auprc={train_auprc:.4f} "
                f"| val_auprc={val_auprc:.4f} val_auroc={val_auroc:.4f}"
            )

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                no_improve = 0
                ckpt = {
                    "model":     model.state_dict(),
                    "epoch":     epoch,
                    "val_auprc": val_auprc,
                }
                if contrast_head is not None:
                    ckpt["contrast_head"] = contrast_head.state_dict()
                torch.save(ckpt, ckpt_dir / "best_auprc.pt")
                logger.info(f"  ✓ 保存最优 checkpoint（val_auprc={best_auprc:.4f}）")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"Early stopping（patience={patience}）")
                    break

        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")

    logger.info(f"训练完成，best_val_auprc={best_auprc:.4f}")


if __name__ == "__main__":
    main()