#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ppisifter.data import PPIPairDataset, collate_pair_batch
from ppisifter.model import PPISifterModel
from ppisifter.utils import get_device, load_config


def build_model(config):
    return PPISifterModel(
        input_dim=config['model']['input_dim'],
        proj_dim=config['model']['proj_dim'],
        pair_hidden_dim=config['model']['pair_hidden_dim'],
        attention_dim=config['model']['attention_dim'],
        attention_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout'],
        fast_filter_threshold=config['model']['fast_filter_threshold'],
    )


def main():
    parser = argparse.ArgumentParser(description='Export simple feature importance statistics from pooled representations.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument('--input_csv', required=True, help='Input pair CSV.')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config['project'].get('device', 'cuda'))
    out_dir = Path(config['outputs']['interpret_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = PPIPairDataset(
        csv_path=args.input_csv,
        embeddings_dir=config['data']['embeddings_dir'],
        suffix=config['data'].get('embedding_suffix', '.residue_emb.npy'),
        max_length=config['model'].get('max_length'),
        cache_in_memory=config['data'].get('cache_in_memory', False),
    )
    loader = DataLoader(dataset, batch_size=config['interpret']['batch_size'], shuffle=False, num_workers=config['data'].get('num_workers', 0), collate_fn=collate_pair_batch)

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    abs_diff_accumulator = []
    hadamard_accumulator = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch['embedding_a'], batch['mask_a'], batch['embedding_b'], batch['mask_b'])
            pooled_a = outputs['pooled_a'].detach().cpu().numpy()
            pooled_b = outputs['pooled_b'].detach().cpu().numpy()
            abs_diff_accumulator.append(np.abs(pooled_a - pooled_b))
            hadamard_accumulator.append(pooled_a * pooled_b)

    abs_diff = np.concatenate(abs_diff_accumulator, axis=0) if abs_diff_accumulator else np.empty((0, config['model']['input_dim']))
    hadamard = np.concatenate(hadamard_accumulator, axis=0) if hadamard_accumulator else np.empty((0, config['model']['input_dim']))

    importance = pd.DataFrame({
        'feature_index': np.arange(abs_diff.shape[1]) if abs_diff.size else np.array([], dtype=int),
        'mean_abs_diff': abs_diff.mean(axis=0) if abs_diff.size else np.array([]),
        'mean_hadamard': hadamard.mean(axis=0) if hadamard.size else np.array([]),
    })
    importance['combined_score'] = importance['mean_abs_diff'].abs() + importance['mean_hadamard'].abs()
    importance = importance.sort_values('combined_score', ascending=False)
    importance.to_csv(out_dir / 'feature_importance.csv', index=False)


if __name__ == '__main__':
    main()
