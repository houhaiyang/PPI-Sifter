import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ppisifter.config import load_config
from ppisifter.io import load_embeddings_pt
from ppisifter.data import PairDataset, collate_fn
from ppisifter.model import PPISifter
from ppisifter.train_utils import run_epoch, save_checkpoint, save_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
    embeddings = load_embeddings_pt(cfg['paths']['embeddings_pt'])
    train_ds = PairDataset(cfg['paths']['train_csv'], embeddings)
    valid_ds = PairDataset(cfg['paths']['valid_csv'], embeddings)
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg.get('num_workers', 0), collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 0), collate_fn=collate_fn)
    model = PPISifter(**cfg['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    best = -1.0
    patience = 0
    ckpt_dir = Path(cfg['paths']['checkpoint_dir'])
    for epoch in range(1, cfg['train']['epochs'] + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, optimizer, cfg, device, train=False)
        save_metrics({'epoch': epoch, 'train': train_metrics, 'valid': valid_metrics}, ckpt_dir / f'epoch_{epoch:03d}.metrics.json')
        score = valid_metrics.get('auprc', float('-inf'))
        if score > best:
            best = score
            patience = 0
            save_checkpoint(model, optimizer, epoch, valid_metrics, ckpt_dir / 'best.pt')
        else:
            patience += 1
        if patience >= cfg['train']['early_stop_patience']:
            break

if __name__ == '__main__':
    main()
