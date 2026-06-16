"""
PPI-Sifter 通用工具函数
功能: 日志初始化、随机种子固定、checkpoint 读写、指标计算
"""

import os
import random
import logging
import numpy as np
import torch
from typing import Dict, Any


def set_seed(seed: int) -> None:
    """固定全局随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    初始化日志器，同时输出到控制台和文件（若指定 log_dir）。

    参数:
        name:    日志器名称
        log_dir: 日志保存目录，None 表示只输出控制台
        level:   日志级别

    返回:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str
) -> None:
    """保存训练 checkpoint。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, device="cpu"):
    """
    加载 checkpoint，支持断点续训。

    返回:
        (start_epoch, best_metric)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", 0.0)
