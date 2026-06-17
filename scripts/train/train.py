"""
脚本: train.py
功能: PPI-Sifter 训练主入口，统一通过 configs/default.yaml 配置，无命令行传参
依赖: torch, h5py, pandas, pyyaml, sklearn, tqdm
运行: python scripts/train/train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppisifter.model import PPISifter
from ppisifter.data import PPIDataset, collate_fn
from ppisifter.losses import PPILoss
from ppisifter.utils import set_seed, get_logger, save_checkpoint, load_checkpoint
from scripts.train.eval import evaluate

# 固定配置文件路径（相对项目根目录）
_CFG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


def load_config() -> dict:
    """加载 YAML 配置文件。"""
    if not os.path.exists(_CFG_PATH):
        raise FileNotFoundError(f"配置文件不存在: {_CFG_PATH}")
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> PPISifter:
    """根据配置构建模型。"""
    m = cfg["model"]
    return PPISifter(
        d_in=m["d_in"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        ffn_expansion=m["ffn_expansion"],
        dropout=m["dropout"],
    )


def main() -> None:
    cfg = load_config()

    set_seed(cfg["project"]["seed"])
    logger = get_logger("train", log_dir=cfg["paths"]["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"配置文件: {_CFG_PATH}")

    # -------- 数据集 --------
    data_cfg    = cfg["data"]
    splits_dir  = data_cfg["splits_dir"]
    hdf5_path   = data_cfg["hdf5_path"]
    max_seq_len = data_cfg["max_seq_len"]
    num_workers = data_cfg.get("num_workers", 0)

    train_set = PPIDataset(
        csv_path=os.path.join(splits_dir, "train.csv"),
        hdf5_path=hdf5_path,
        max_seq_len=max_seq_len,
    )
    valid_set = PPIDataset(
        csv_path=os.path.join(splits_dir, "valid.csv"),
        hdf5_path=hdf5_path,
        max_seq_len=max_seq_len,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=cfg["train"]["batch_size"] * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    logger.info(f"训练集: {len(train_set)} 对，验证集: {len(valid_set)} 对")

    # -------- 模型 --------
    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数量: {param_count:,}")

    # -------- 损失函数 --------
    t = cfg["train"]
    criterion = PPILoss(
        pos_weight=t["pos_weight"],
        focal_gamma=t["focal_gamma"],
        lambda_focal=t["lambda_focal"],
        lambda_sparse=t["lambda_sparse"],
        lambda_sym=t["lambda_sym"],
    )

    # -------- 优化器 & 学习率调度 --------
    optimizer = optim.AdamW(
        model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
    )
    if t["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t["epochs"], eta_min=t["lr"] * 0.01
        )
    else:
        scheduler = None

    # -------- 断点续训 --------
    start_epoch = 0
    best_auprc  = 0.0
    resume_path = t.get("resume_checkpoint", "")
    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_auprc = load_checkpoint(
            resume_path, model, optimizer, device
        )
        logger.info(f"从 epoch={start_epoch}, best_auprc={best_auprc:.4f} 恢复训练")

    # -------- 混合精度 --------
    use_fp16 = t.get("fp16", False) and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler() if use_fp16 else None

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------- Early stopping 状态 --------
    patience       = t.get("early_stop_patience", 8)
    no_improve_cnt = 0

    # -------- 训练循环 --------
    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        total_loss  = 0.0
        num_batches = 0

        for emb_a, emb_b, mask_a, mask_b, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{t['epochs']}", leave=False
        ):
            emb_a  = emb_a.to(device)
            emb_b  = emb_b.to(device)
            mask_a = mask_a.to(device)
            mask_b = mask_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    out = model(emb_a, emb_b, mask_a, mask_b, return_attention=True)
                    loss = criterion(
                        out["logits"], labels,
                        out.get("attn_ab"), out.get("attn_ba"),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(emb_a, emb_b, mask_a, mask_b, return_attention=True)
                loss = criterion(
                    out["logits"], labels,
                    out.get("attn_ab"), out.get("attn_ba"),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                optimizer.step()

            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if scheduler is not None:
            scheduler.step()

        # -------- 验证 --------
        val_interval = t.get("val_interval", 1)
        if (epoch + 1) % val_interval == 0:
            metrics    = evaluate(model, valid_loader, device)
            lr_now     = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                f"AUPRC={metrics['auprc']:.4f} | AUROC={metrics['auroc']:.4f} | "
                f"F1={metrics['f1']:.4f} | MCC={metrics['mcc']:.4f} | LR={lr_now:.2e}"
            )

            if metrics["auprc"] > best_auprc:
                best_auprc     = metrics["auprc"]
                no_improve_cnt = 0
                save_checkpoint(
                    {
                        "epoch":                epoch + 1,
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric":          best_auprc,
                        "config":               cfg,
                    },
                    ckpt_dir,
                    "best_auprc.pt",
                )
                logger.info(f"  >>> 保存最优 checkpoint: AUPRC={best_auprc:.4f}")
            else:
                no_improve_cnt += 1
                if no_improve_cnt >= patience:
                    logger.info(
                        f"Early stopping: {patience} 轮验证无提升，停止训练"
                    )
                    break

        # 每 10 epoch 定期保存
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                {
                    "epoch":                epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric":          best_auprc,
                    "config":               cfg,
                },
                ckpt_dir,
                f"epoch_{epoch+1:03d}.pt",
            )

    logger.info(f"训练完成，最优 AUPRC={best_auprc:.4f}")


if __name__ == "__main__":
    main()
