#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppisifter.data import PPIPairDataset, collate_pair_batch
from ppisifter.losses import PPISifterLoss
from ppisifter.model import PPISifterModel
from ppisifter.utils import compute_classification_metrics, ensure_dir, get_device, load_config, save_json, set_seed
from scripts.train.save_checkpoint import save_checkpoint


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


def evaluate(model, loader, criterion, device):
    model.eval()
    probs_all = []
    labels_all = []
    loss_values = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(batch['embedding_a'], batch['mask_a'], batch['embedding_b'], batch['mask_b'])
            loss_out = criterion(outputs['logits'], batch['label'], outputs['attention_ab'], outputs['attention_ba'])
            probs = torch.sigmoid(outputs['logits']).detach().cpu().numpy().reshape(-1)
            labels = batch['label'].detach().cpu().numpy().reshape(-1)
            probs_all.append(probs)
            labels_all.append(labels)
            loss_values.append(loss_out.items['loss_total'])
    probs_all = np.concatenate(probs_all) if probs_all else np.array([])
    labels_all = np.concatenate(labels_all) if labels_all else np.array([])
    metrics = compute_classification_metrics(labels_all, probs_all, threshold=0.5) if len(labels_all) else {}
    metrics['loss'] = float(np.mean(loss_values)) if loss_values else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train PPI-Sifter model.')
    parser.add_argument('--config', required=True, help='Path to YAML config.')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config['project'].get('seed', 42)))
    device = get_device(config['project'].get('device', 'cuda'))

    train_dataset = PPIPairDataset(
        csv_path=config['data']['train_csv'],
        embeddings_dir=config['data']['embeddings_dir'],
        suffix=config['data'].get('embedding_suffix', '.residue_emb.npy'),
        max_length=config['model'].get('max_length'),
        cache_in_memory=config['data'].get('cache_in_memory', False),
    )
    valid_dataset = PPIPairDataset(
        csv_path=config['data']['valid_csv'],
        embeddings_dir=config['data']['embeddings_dir'],
        suffix=config['data'].get('embedding_suffix', '.residue_emb.npy'),
        max_length=config['model'].get('max_length'),
        cache_in_memory=config['data'].get('cache_in_memory', False),
    )

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train'].get('num_workers', 0), collate_fn=collate_pair_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train'].get('num_workers', 0), collate_fn=collate_pair_batch)

    model = build_model(config).to(device)
    criterion = PPISifterLoss(
        pos_weight=config['train']['pos_weight'],
        focal_alpha=config['train']['focal_alpha'],
        focal_gamma=config['train']['focal_gamma'],
        lambda_focal=config['train']['lambda_focal'],
        lambda_sparse=config['train']['lambda_sparse'],
        lambda_sym=config['train']['lambda_sym'],
        lambda_hotspot=config['train']['lambda_hotspot'],
    )
    optimizer = AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    scaler = GradScaler(enabled=bool(config['train'].get('amp', True) and device.type == 'cuda'))

    checkpoints_dir = ensure_dir(config['outputs']['checkpoints_dir'])
    best_metric = -1.0
    patience = 0
    history = []

    for epoch in range(1, int(config['train']['epochs']) + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch in progress:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=bool(config['train'].get('amp', True) and device.type == 'cuda')):
                outputs = model(batch['embedding_a'], batch['mask_a'], batch['embedding_b'], batch['mask_b'])
                loss_out = criterion(outputs['logits'], batch['label'], outputs['attention_ab'], outputs['attention_ba'])
                loss = loss_out.total
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config['train']['grad_clip']))
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss_out.items['loss_total'])
            progress.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        valid_metrics = evaluate(model, valid_loader, criterion, device)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        epoch_record = {'epoch': epoch, 'train_loss': train_loss, **valid_metrics}
        history.append(epoch_record)

        monitor_name = config['train'].get('monitor', 'auprc')
        current_metric = float(valid_metrics.get(monitor_name, 0.0))
        save_checkpoint(str(Path(checkpoints_dir) / 'last.pt'), model, optimizer, epoch, epoch_record, config)
        if current_metric >= best_metric:
            best_metric = current_metric
            patience = 0
            save_checkpoint(str(Path(checkpoints_dir) / 'best.pt'), model, optimizer, epoch, epoch_record, config)
        else:
            patience += 1
            if patience >= int(config['train']['early_stop_patience']):
                break

    save_json({'history': history, 'best_monitor': best_metric}, Path(checkpoints_dir) / 'train_history.json')


if __name__ == '__main__':
    main()
