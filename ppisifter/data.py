from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import pandas as pd


@dataclass
class Batch:
    a: torch.Tensor
    b: torch.Tensor
    mask_a: torch.Tensor
    mask_b: torch.Tensor
    labels: torch.Tensor | None
    meta: list


class PairDataset(Dataset):
    def __init__(self, pairs_csv, embeddings):
        self.df = pd.read_csv(pairs_csv)
        self.embeddings = embeddings
        self.df = self.df[
            self.df['protein_a'].isin(embeddings.keys()) &
            self.df['protein_b'].isin(embeddings.keys())
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label'] if 'label' in row.index else None
        return {
            'protein_a': row['protein_a'],
            'protein_b': row['protein_b'],
            'a': self.embeddings[row['protein_a']]['embedding'].float(),
            'b': self.embeddings[row['protein_b']]['embedding'].float(),
            'label': None if label is None else float(label),
        }


def _pad_stack(xs):
    max_len = max(x.shape[0] for x in xs)
    dim = xs[0].shape[1]
    arr = torch.zeros(len(xs), max_len, dim)
    mask = torch.zeros(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        arr[i, :x.shape[0]] = x
        mask[i, :x.shape[0]] = True
    return arr, mask


def collate_fn(items):
    a, mask_a = _pad_stack([x['a'] for x in items])
    b, mask_b = _pad_stack([x['b'] for x in items])
    labels = None
    if items[0]['label'] is not None:
        labels = torch.tensor([x['label'] for x in items], dtype=torch.float32)
    meta = [{'protein_a': x['protein_a'], 'protein_b': x['protein_b']} for x in items]
    return Batch(a=a, b=b, mask_a=mask_a, mask_b=mask_b, labels=labels, meta=meta)
