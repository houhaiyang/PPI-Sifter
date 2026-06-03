#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ppisifter.data import PPIPairDataset, collate_pair_batch
from ppisifter.interpret import compute_attention_statistics, extract_topk_residue_pairs
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
    parser = argparse.ArgumentParser(description='Run batch inference for PPI-Sifter.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument('--input_csv', required=True, help='Input pair CSV.')
    parser.add_argument('--output_csv', required=True, help='Output prediction CSV.')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config['project'].get('device', 'cuda'))

    dataset = PPIPairDataset(
        csv_path=args.input_csv,
        embeddings_dir=config['data']['embeddings_dir'],
        suffix=config['data'].get('embedding_suffix', '.residue_emb.npy'),
        max_length=config['model'].get('max_length'),
        cache_in_memory=config['data'].get('cache_in_memory', False),
    )
    loader = DataLoader(dataset, batch_size=config['infer']['batch_size'], shuffle=False, num_workers=config['data'].get('num_workers', 0), collate_fn=collate_pair_batch)

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    rows = []
    top_k = int(config['model'].get('top_k_residue_pairs', 50))
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch['embedding_a'], batch['mask_a'], batch['embedding_b'], batch['mask_b'])
            probs = torch.sigmoid(outputs['logits']).detach().cpu().numpy().reshape(-1)
            topk_items = extract_topk_residue_pairs(outputs['attention_ab'], batch['mask_a'], batch['mask_b'], top_k=top_k)
            stats = compute_attention_statistics(outputs['attention_ab'], outputs['attention_ba'], batch['mask_a'], batch['mask_b'])
            for pair_id, protein_a, protein_b, prob, items in zip(batch['pair_id'], batch['protein_a'], batch['protein_b'], probs, topk_items):
                rows.append({
                    'pair_id': pair_id,
                    'protein_a': protein_a,
                    'protein_b': protein_b,
                    'probability': float(prob),
                    'prediction': int(prob >= float(config['infer'].get('threshold', 0.5))),
                    'top1_residue_pair': f"{items[0]['residue_a']}-{items[0]['residue_b']}" if items else '',
                    'top1_attention_score': float(items[0]['score']) if items else 0.0,
                    'attention_entropy_batch': stats['attention_entropy'],
                    'symmetry_gap_batch': stats['symmetry_gap'],
                })

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
