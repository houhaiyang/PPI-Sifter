#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import glob
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ResidueEmbeddingNPZStore:
    """
    从 batch_*.npz 中按 seq_id 读取 residue embedding。
    每个 batch_*.npz 是一个 zip 容器，内部成员名类似:
        Q5YLW3.npy
        P12345.npy
    每个成员保存一个 shape [L, D] 的 numpy array。
    """

    def __init__(self, embeddings_dir: str, dtype: torch.dtype = torch.float32):
        self.embeddings_dir = Path(embeddings_dir)
        self.dtype = dtype

        self.batch_files = sorted(self.embeddings_dir.glob("batch_*.npz"))
        if len(self.batch_files) == 0:
            raise FileNotFoundError(
                f"No batch_*.npz found under: {self.embeddings_dir}"
            )

        self.seqid_to_batch: Dict[str, Path] = {}
        self._zip_handles: Dict[str, zipfile.ZipFile] = {}

        self._build_index()

    def _build_index(self):
        for batch_file in self.batch_files:
            with zipfile.ZipFile(batch_file, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith(".npy"):
                        continue
                    seq_id = name[:-4]
                    self.seqid_to_batch[seq_id] = batch_file

    def __len__(self):
        return len(self.seqid_to_batch)

    def has(self, seq_id: str) -> bool:
        return seq_id in self.seqid_to_batch

    def _get_zip_handle(self, batch_file: Path) -> zipfile.ZipFile:
        key = str(batch_file)
        if key not in self._zip_handles:
            self._zip_handles[key] = zipfile.ZipFile(batch_file, "r")
        return self._zip_handles[key]

    def get_numpy(self, seq_id: str) -> np.ndarray:
        if seq_id not in self.seqid_to_batch:
            raise KeyError(f"Embedding not found for seq_id: {seq_id}")

        batch_file = self.seqid_to_batch[seq_id]
        zf = self._get_zip_handle(batch_file)
        member_name = f"{seq_id}.npy"

        with zf.open(member_name, "r") as f:
            bio = io.BytesIO(f.read())
            arr = np.load(bio, allow_pickle=False)

        if arr.ndim != 2:
            raise ValueError(f"Bad embedding shape for {seq_id}: {arr.shape}")

        return arr

    def get_tensor(self, seq_id: str) -> torch.Tensor:
        arr = self.get_numpy(seq_id)
        return torch.from_numpy(arr).to(self.dtype)

    def close(self):
        for zf in self._zip_handles.values():
            try:
                zf.close()
            except Exception:
                pass
        self._zip_handles = {}


def infer_pair_columns(df: pd.DataFrame):
    col_sets = [
        ("protein_a", "protein_b", "label"),
        ("protein1", "protein2", "label"),
        ("prot_a", "prot_b", "label"),
        ("seq_id_a", "seq_id_b", "label"),
        ("A", "B", "label"),
    ]
    for a, b, y in col_sets:
        if a in df.columns and b in df.columns and y in df.columns:
            return a, b, y

    raise ValueError(
        f"Cannot infer pair columns from dataframe columns: {list(df.columns)}"
    )


class PPIPairDatasetNPZ(Dataset):
    """
    适配 batch_*.npz residue embedding 的 PPI pair dataset。
    读取 pair csv/tsv，并按 protein A/B 取 residue embedding。
    """

    def __init__(
        self,
        pair_file: str,
        embeddings_dir: str,
        sep: Optional[str] = None,
        label_dtype: torch.dtype = torch.float32,
        emb_dtype: torch.dtype = torch.float32,
        drop_missing: bool = True,
    ):
        self.pair_file = pair_file
        self.emb_store = ResidueEmbeddingNPZStore(embeddings_dir, dtype=emb_dtype)
        self.label_dtype = label_dtype

        if sep is None:
            if pair_file.endswith(".tsv"):
                sep = "\t"
            else:
                sep = ","

        self.df = pd.read_csv(pair_file, sep=sep)
        self.col_a, self.col_b, self.col_y = infer_pair_columns(self.df)

        if drop_missing:
            keep_rows = []
            for i, row in self.df.iterrows():
                a = str(row[self.col_a])
                b = str(row[self.col_b])
                if self.emb_store.has(a) and self.emb_store.has(b):
                    keep_rows.append(i)
            self.df = self.df.loc[keep_rows].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        prot_a = str(row[self.col_a])
        prot_b = str(row[self.col_b])
        label = float(row[self.col_y])

        emb_a = self.emb_store.get_tensor(prot_a)   # [La, D]
        emb_b = self.emb_store.get_tensor(prot_b)   # [Lb, D]

        item = {
            "protein_a": prot_a,
            "protein_b": prot_b,
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": emb_a.shape[0],
            "len_b": emb_b.shape[0],
            "label": torch.tensor(label, dtype=self.label_dtype),
        }
        return item

    def close(self):
        self.emb_store.close()


def pad_residue_embeddings(
    tensors: List[torch.Tensor],
    padding_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        tensors: list of [L, D]
    输出:
        padded: [B, Lmax, D]
        mask:   [B, Lmax]  1表示有效位, 0表示padding
    """
    batch_size = len(tensors)
    max_len = max(x.shape[0] for x in tensors)
    dim = tensors[0].shape[1]
    dtype = tensors[0].dtype
    device = tensors[0].device

    padded = torch.full(
        (batch_size, max_len, dim),
        fill_value=padding_value,
        dtype=dtype,
        device=device,
    )
    mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
        device=device,
    )

    for i, x in enumerate(tensors):
        L = x.shape[0]
        padded[i, :L] = x
        mask[i, :L] = True

    return padded, mask


def ppi_pair_collate_fn(batch: List[Dict]):
    emb_a_list = [x["emb_a"] for x in batch]
    emb_b_list = [x["emb_b"] for x in batch]

    x_a, mask_a = pad_residue_embeddings(emb_a_list)
    x_b, mask_b = pad_residue_embeddings(emb_b_list)

    labels = torch.stack([x["label"] for x in batch], dim=0)
    len_a = torch.tensor([x["len_a"] for x in batch], dtype=torch.long)
    len_b = torch.tensor([x["len_b"] for x in batch], dtype=torch.long)

    out = {
        "protein_a": [x["protein_a"] for x in batch],
        "protein_b": [x["protein_b"] for x in batch],
        "x_a": x_a,           # [B, La_max, D]
        "x_b": x_b,           # [B, Lb_max, D]
        "mask_a": mask_a,     # [B, La_max]
        "mask_b": mask_b,     # [B, Lb_max]
        "len_a": len_a,       # [B]
        "len_b": len_b,       # [B]
        "label": labels,      # [B]
    }
    return out