#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ppisifter.data import PPIPairDataset, collate_pair_batch
from ppisifter.interpret import extract_topk_residue_pairs, topk_records_to_frame
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
    parser = argparse.ArgumentParser(description='Export attention maps and top-k residue pairs.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path.')
    parser.add_argument('--input_csv', required=True, help='Input pair CSV.')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config['project'].get('device', 'cuda'))
    top_k = int(config['interpret'].get('top_k', 50))
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

    all_topk_frames = []
    sample_written = False
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch['embedding_a'], batch['mask_a'], batch['embedding_b'], batch['mask_b'])
            topk_items = extract_topk_residue_pairs(outputs['attention_ab'], batch['mask_a'], batch['mask_b'], top_k=top_k)
            all_topk_frames.append(topk_records_to_frame(batch['pair_id'], topk_items))
            if not sample_written and len(batch['pair_id']) > 0:
                matrix = outputs['attention_ab'][0].detach().cpu().numpy()
                pd.DataFrame(matrix).to_csv(out_dir / 'sample_attention.csv', index=False)
                sample_written = True

    if all_topk_frames:
        pd.concat(all_topk_frames, ignore_index=True).to_csv(out_dir / 'topk_residue_pairs.csv', index=False)


if __name__ == '__main__':
    main()
