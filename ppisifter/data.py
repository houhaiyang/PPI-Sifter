"""
PPI-Sifter 数据模块
功能: 基于 HDF5 的按需加载 Dataset，支持 protein-disjoint / cluster-disjoint / species-disjoint split
依赖: h5py, pandas, torch, numpy
入口: PPIDataset(csv_path, hdf5_path, max_seq_len)
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Tuple, Optional


class PPIDataset(Dataset):
    """
    蛋白对 Dataset，使用 HDF5 随机访问按需加载 residue embeddings，避免全量载入 OOM。

    参数:
        csv_path:    pair CSV 路径，列：protein_a, protein_b, label
        hdf5_path:   HDF5 嵌入文件路径，key=protein_id, value=ndarray(L, D)
        max_seq_len: 残基序列截断上限
        inference:   True 时允许 label 列缺失（推理模式）
    """

    def __init__(
        self,
        csv_path: str,
        hdf5_path: str,
        max_seq_len: int = 1024,
        inference: bool = False,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.max_seq_len = max_seq_len
        self.inference = inference

        # 加载蛋白对表格（轻量 CSV，全量读入无压力）
        self.pairs = pd.read_csv(csv_path)
        required_cols = {"protein_a", "protein_b"}
        if not required_cols.issubset(self.pairs.columns):
            raise ValueError(f"CSV 缺少必要列: {required_cols}")
        if not inference and "label" not in self.pairs.columns:
            raise ValueError("非推理模式下 CSV 必须包含 label 列")

        # HDF5 句柄延迟打开（DataLoader 多进程安全）
        self._h5: Optional[h5py.File] = None

    def _open_h5(self) -> h5py.File:
        """延迟打开 HDF5 文件（每个 worker 各自打开一次）。"""
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _load_emb(self, protein_id: str) -> Tensor:
        """
        从 HDF5 随机访问单个蛋白 embedding，截断到 max_seq_len。

        返回:
            Tensor (L, D)  L <= max_seq_len
        """
        h5 = self._open_h5()
        if protein_id not in h5:
            raise KeyError(f"HDF5 中未找到蛋白: {protein_id}")
        emb = h5[protein_id][:]                          # (L, D) numpy
        emb = emb[: self.max_seq_len]                    # 截断
        return torch.from_numpy(emb.astype(np.float32))  # -> float32 Tensor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.pairs.iloc[idx]
        emb_a = self._load_emb(str(row["protein_a"]))
        emb_b = self._load_emb(str(row["protein_b"]))
        if self.inference:
            label = torch.tensor(-1.0)
        else:
            label = torch.tensor(float(row["label"]))
        return emb_a, emb_b, label

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def collate_fn(
    batch: list,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    自定义 collate：对 residue 序列做 padding，生成 attention mask。

    返回:
        emb_a:   (B, La_max, D)
        emb_b:   (B, Lb_max, D)
        mask_a:  (B, La_max)  True=有效残基
        mask_b:  (B, Lb_max)  True=有效残基
        labels:  (B,)
    """
    emb_a_list, emb_b_list, labels = zip(*batch)

    def pad_sequence(seqs):
        max_len = max(s.size(0) for s in seqs)
        d = seqs[0].size(1)
        padded = torch.zeros(len(seqs), max_len, d)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
        for i, s in enumerate(seqs):
            l = s.size(0)
            padded[i, :l] = s
            mask[i, :l] = True
        return padded, mask

    emb_a_pad, mask_a = pad_sequence(emb_a_list)
    emb_b_pad, mask_b = pad_sequence(emb_b_list)
    labels = torch.stack(labels)
    return emb_a_pad, emb_b_pad, mask_a, mask_b, labels
