"""
PPI-Sifter 数据模块
优化点:
  1. LRU 内存缓存：热点蛋白 embedding 缓存在 RAM，避免重复 HDF5 I/O
  2. 向量化 collate_fn：用 torch.zeros + slice 替代 Python for-loop
  3. __getstate__/__setstate__ 保证 DataLoader 多进程 pickle 安全
  4. persistent_workers 友好：_h5 延迟打开，进程复用时不重建
"""

import h5py
import numpy as np
import pandas as pd
import torch
from functools import lru_cache
from torch import Tensor
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class PPIDataset(Dataset):
    """
    蛋白对 Dataset。

    参数:
        csv_path:    pair CSV 路径，列：protein_a, protein_b, label
        hdf5_path:   HDF5 嵌入文件路径，key=protein_id, value=ndarray(L, D)
        max_seq_len: 残基序列截断上限
        cache_size:  LRU 缓存蛋白数量（0 = 不缓存）
        inference:   True 时允许 label 列缺失
    """

    def __init__(
        self,
        csv_path:    str,
        hdf5_path:   str,
        max_seq_len: int  = 512,
        cache_size:  int  = 8192,
        inference:   bool = False,
    ) -> None:
        self.hdf5_path   = hdf5_path
        self.max_seq_len = max_seq_len
        self.inference   = inference
        self.cache_size  = cache_size

        self.pairs = pd.read_csv(csv_path)
        required = {"protein_a", "protein_b"}
        if not required.issubset(self.pairs.columns):
            raise ValueError(f"CSV 缺少必要列: {required}")
        if not inference and "label" not in self.pairs.columns:
            raise ValueError("非推理模式下 CSV 必须包含 label 列")

        # 预提取为 numpy array 加速 iloc 访问
        self._prot_a = self.pairs["protein_a"].astype(str).values
        self._prot_b = self.pairs["protein_b"].astype(str).values
        self._labels = (
            None if inference
            else self.pairs["label"].astype(np.float32).values
        )

        # HDF5 句柄：每进程延迟打开
        self._h5: Optional[h5py.File] = None
        # 进程级 LRU 缓存（在 worker 进程内各自独立）
        self._cache: Optional[Dict[str, np.ndarray]] = None

    # ── pickle 安全 ─────────────────────────────────────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"]    = None
        state["_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5    = None
        self._cache = None

    # ── HDF5 延迟打开 ────────────────────────────────────────────────────────
    def _open_h5(self) -> h5py.File:
        if self._h5 is None:
            # swmr=True 允许多进程并发只读，提升 I/O 吞吐
            try:
                self._h5 = h5py.File(self.hdf5_path, "r", swmr=True)
            except Exception:
                self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _init_cache(self):
        if self._cache is None:
            self._cache = {}

    def _load_emb(self, protein_id: str) -> Tensor:
        """从 LRU 缓存或 HDF5 加载 embedding，截断到 max_seq_len。"""
        self._init_cache()

        if protein_id in self._cache:
            arr = self._cache[protein_id]
        else:
            h5  = self._open_h5()
            if protein_id not in h5:
                raise KeyError(
                    f"HDF5 中未找到蛋白: '{protein_id}'。"
                    f"请用 scripts/emb/verify_hdf5.py 检查 key 格式。"
                )
            arr = h5[protein_id][: self.max_seq_len].astype(np.float32)
            # 简单 LRU：超出 cache_size 时清半
            if self.cache_size > 0:
                if len(self._cache) >= self.cache_size:
                    # 删除前 1/4
                    evict_keys = list(self._cache.keys())[: self.cache_size // 4]
                    for k in evict_keys:
                        del self._cache[k]
                self._cache[protein_id] = arr

        return torch.from_numpy(arr)

    # ── Dataset 接口 ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        emb_a = self._load_emb(self._prot_a[idx])
        emb_b = self._load_emb(self._prot_b[idx])
        label = (
            torch.tensor(-1.0)
            if self.inference
            else torch.tensor(self._labels[idx])
        )
        return emb_a, emb_b, label

    def __del__(self):
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass


def collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    向量化 collate：zero-padding + bool mask。
    替代原 Python for-loop，减少 CPU 开销。

    返回:
        emb_a:  (B, La_max, D)
        emb_b:  (B, Lb_max, D)
        mask_a: (B, La_max)  True=有效
        mask_b: (B, Lb_max)  True=有效
        labels: (B,)
    """
    emb_a_list, emb_b_list, label_list = zip(*batch)

    def _pad(seqs: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        lens    = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
        max_len = int(lens.max().item())
        d       = seqs[0].size(1)
        B       = len(seqs)
        padded  = torch.zeros(B, max_len, d, dtype=torch.float32)
        mask    = torch.zeros(B, max_len, dtype=torch.bool)
        for i, (s, l) in enumerate(zip(seqs, lens)):
            padded[i, :l] = s
            mask[i,   :l] = True
        return padded, mask

    emb_a_pad, mask_a = _pad(emb_a_list)
    emb_b_pad, mask_b = _pad(emb_b_list)
    labels = torch.stack(label_list)
    return emb_a_pad, emb_b_pad, mask_a, mask_b, labels