# -*- coding: utf-8 -*-
"""
文件功能：IO 工具模块（文件读写、embedding 加载）
运行依赖：torch, pandas, numpy
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import torch
import pandas as pd


def load_embeddings_pt(path: str | Path) -> dict:
    """
    加载 .pt 格式 embedding 文件。

    Args:
        path: .pt 文件路径
    Returns:
        dict: accession -> Tensor
    """
    payload = torch.load(path, map_location="cpu")
    return payload.get("data", payload)


def load_embeddings_npz_dir(
    directory: str | Path, max_len: int | None = None
) -> dict:
    """
    扫描目录下所有 batch*.npz，返回 accession -> Tensor 字典。

    Args:
        directory: 含 batch*.npz 的目录路径
        max_len:   序列最大长度，超出则截断；None 表示不截断
    Returns:
        dict: accession -> torch.Tensor [L, D]
    """
    directory = Path(directory)
    result = {}
    for bf in sorted(directory.glob("batch*.npz")):
        try:
            with np.load(bf, allow_pickle=False) as npz:
                for key in npz.files:
                    acc = key[:-4] if key.endswith(".npy") else key
                    t = torch.from_numpy(npz[key].astype(np.float32))
                    if max_len and t.shape[0] > max_len:
                        t = t[:max_len]
                    result[acc] = t
        except Exception as exc:
            print(f"WARNING 加载 {bf.name} 失败: {exc}", file=sys.stderr)
    return result


def read_pairs_csv(path: str | Path) -> pd.DataFrame:
    """读取 pair CSV 文件。"""
    return pd.read_csv(path)


def ensure_dir(path: str | Path) -> None:
    """确保目录存在，不存在则创建。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(obj: dict, path: str | Path) -> None:
    """将字典写入 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")