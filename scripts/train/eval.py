#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ppisifter.data import PPIPairDataset, collate_pair_batch
from ppisifter.losses import PPISifterLoss
from ppisifter.model import PPISifterModel
from ppisifter.utils import compute_classification_metrics, get_device, load_config, save_json


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
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on a split.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'], help='Split name.')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config['project'].get('device', 'cuda'))
    split_csv = config['data'][f'{args.split}_csv']

    dataset = PPIPairDataset(
        csv_path=split_csv,
        embeddings_dir=config['data']['embeddings_dir'],
        suffix=config['data'].get('embedding_suffix', '.residue_emb.npy'),
        max_length=config['model'].get('max_length'),
        cache_in_memory=config['data'].get('cache_in_memory', False),
    )
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train'].get('num_workers', 0), collate_fn=collate_pair_batch)

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    probs_all, labels_all, pair_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch['embedding_a'], batch['mask_a'], batch['embedding_b'], batch['mask_b'])
            probs = torch.sigmoid(outputs['logits']).detach().cpu().numpy().reshape(-1)
            labels = batch['label'].detach().cpu().numpy().reshape(-1)
            probs_all.append(probs)
            labels_all.append(labels)
            pair_ids.extend(batch['pair_id'])

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    metrics = compute_classification_metrics(labels_all, probs_all, threshold=float(config['infer'].get('threshold', 0.5)))

    out_dir = Path(config['outputs']['preds_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({'pair_id': pair_ids, 'label': labels_all, 'probability': probs_all})
    pred_df.to_csv(out_dir / f'{args.split}_eval_predictions.csv', index=False)
    save_json(metrics, out_dir / f'{args.split}_metrics.json')


if __name__ == '__main__':
    main()
