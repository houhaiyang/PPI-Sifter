#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ppisifter.utils import load_config, save_json


def main():
    parser = argparse.ArgumentParser(description='Merge residue embedding inventory for incremental generation scenario.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--output', required=True, help='Output .pt path.')
    args = parser.parse_args()

    config = load_config(args.config)
    embeddings_dir = Path(config['data']['embeddings_dir'])
    suffix = config['data'].get('embedding_suffix', '.residue_emb.npy')
    files = sorted(embeddings_dir.glob(f'*{suffix}'))

    index_rows = []
    dims = {}
    for path in files:
        protein_id = path.name[: -len(suffix)]
        try:
            arr = np.load(path, mmap_mode='r')
            if arr.ndim != 2:
                continue
            index_rows.append({
                'protein_id': protein_id,
                'path': str(path),
                'seq_len': int(arr.shape[0]),
                'emb_dim': int(arr.shape[1]),
            })
            dims[int(arr.shape[1])] = dims.get(int(arr.shape[1]), 0) + 1
        except Exception:
            continue

    index_df = pd.DataFrame(index_rows).sort_values(['protein_id']).reset_index(drop=True)
    payload = {
        'index': index_df.to_dict(orient='records'),
        'num_embeddings': int(len(index_df)),
        'dimension_histogram': dims,
        'embedding_suffix': suffix,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    index_csv = output_path.with_suffix('.index.csv')
    report_json = output_path.with_suffix('.report.json')
    index_df.to_csv(index_csv, index=False)
    save_json(payload, report_json)


if __name__ == '__main__':
    main()
