from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PairRecord:
    pair_id: str
    protein_a: str
    protein_b: str
    label: int


class EmbeddingStore:
    def __init__(self, embeddings_dir: str, suffix: str = '.residue_emb.npy', cache_in_memory: bool = False):
        self.embeddings_dir = Path(embeddings_dir)
        self.suffix = suffix
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[str, torch.Tensor] = {}

    def path_for(self, protein_id: str) -> Path:
        return self.embeddings_dir / f'{protein_id}{self.suffix}'

    def exists(self, protein_id: str) -> bool:
        return self.path_for(protein_id).exists()

    def load(self, protein_id: str) -> torch.Tensor:
        if self.cache_in_memory and protein_id in self._cache:
            return self._cache[protein_id]
        path = self.path_for(protein_id)
        if not path.exists():
            raise FileNotFoundError(f'Missing embedding file: {path}')
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f'Embedding for {protein_id} must be 2D, got shape={array.shape}')
        tensor = torch.from_numpy(array).float()
        if self.cache_in_memory:
            self._cache[protein_id] = tensor
        return tensor


class PPIPairDataset(Dataset):
    def __init__(self, csv_path: str, embeddings_dir: str, suffix: str = '.residue_emb.npy', max_length: Optional[int] = None, cache_in_memory: bool = False):
        self.df = pd.read_csv(csv_path)
        self.store = EmbeddingStore(embeddings_dir=embeddings_dir, suffix=suffix, cache_in_memory=cache_in_memory)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def _trim(self, emb: torch.Tensor) -> torch.Tensor:
        if self.max_length is None or emb.size(0) <= self.max_length:
            return emb
        return emb[: self.max_length]

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        emb_a = self._trim(self.store.load(row['protein_a']))
        emb_b = self._trim(self.store.load(row['protein_b']))
        item = {
            'pair_id': row.get('pair_id', f"{row['protein_a']}__{row['protein_b']}"),
            'protein_a': row['protein_a'],
            'protein_b': row['protein_b'],
            'embedding_a': emb_a,
            'embedding_b': emb_b,
            'label': torch.tensor(float(row['label']), dtype=torch.float32),
        }
        return item


def pad_embeddings(tensors: List[torch.Tensor]):
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)
    dim = tensors[0].size(1)
    max_len = int(lengths.max().item())
    batch = torch.zeros(len(tensors), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        batch[i, : tensor.size(0)] = tensor
        mask[i, : tensor.size(0)] = True
    return batch, mask, lengths


def collate_pair_batch(batch: List[Dict[str, torch.Tensor]]):
    emb_a, mask_a, len_a = pad_embeddings([x['embedding_a'] for x in batch])
    emb_b, mask_b, len_b = pad_embeddings([x['embedding_b'] for x in batch])
    labels = torch.stack([x['label'] for x in batch]).view(-1, 1)
    return {
        'pair_id': [x['pair_id'] for x in batch],
        'protein_a': [x['protein_a'] for x in batch],
        'protein_b': [x['protein_b'] for x in batch],
        'embedding_a': emb_a,
        'embedding_b': emb_b,
        'mask_a': mask_a,
        'mask_b': mask_b,
        'len_a': len_a,
        'len_b': len_b,
        'label': labels,
    }
